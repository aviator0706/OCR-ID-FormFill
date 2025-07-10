import os
import cv2
import easyocr
from ultralytics import YOLO
import re
from collections import defaultdict
from gtts import gTTS

# === Config ===
image_input_path = r"C:\Users\<username>\<your folder path>"
model_path = "mymodel.pt"
final_output_dir = "final_outputs"
idcard_output_dir = "idcard_outputs"
os.makedirs(final_output_dir, exist_ok=True)
os.makedirs(idcard_output_dir, exist_ok=True)

yolo_model = YOLO(model_path)
ocr_reader = easyocr.Reader(['en'])

# --- IOU and NMS for class 0,1,2 ---
def iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * (yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    unionArea = boxAArea + boxBArea - interArea
    return interArea / unionArea if unionArea != 0 else 0

def non_max_suppression_area(boxes, iou_thresh=0.4):
    boxes = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    final_boxes = []
    for box in boxes:
        if all(iou(box, kept) < iou_thresh for kept in final_boxes):
            final_boxes.append(box)
    return final_boxes

# --- Class mapping and field normalization ---
class_map = {
    3: "Surname", 4: "Name", 5: "Nationality", 6: "Sex", 7: "Date of Birth",
    8: "Place of Birth", 9: "Issue Date", 10: "Expiry Date", 11: "Issuing Office",
    12: "Height", 13: "Type", 14: "Country", 15: "Passport No", 16: "Personal No", 17: "Card No"
}
field_equivalents = {
    "Country": ["Country", "Code of State", "Code of Issuing State", "Codeof Issulng State", "ICode of State"],
    "Issuing Office": ["Issuing Office", "Issuing Authority", "Issuing office", "Iss. office", "Authority", "authoricy", "Iss office", "issuing authority"],
    "Passport No": ["Passport No", "Document No", "IPassport No"],
    "Personal No": ["Personal No", "National ID"],
    "Date of Birth": ["Date of Birth", "DOB", "Date ofbimn", "of birth", "ofbimn", "of pirth"],
    "Issue Date": ["Issue Date", "Date of Issue", "dale"],
    "Expiry Date": ["Expiry Date", "Date of Expiry", "of expiny"],
    "Name": ["Name", "Given Name", "Given", "nane", "Given name"],
    "Surname": ["Surname", "Last Name", "Sumname"],
    "Place of Birth": ["Place of Birth", "Place of binth"],
    "Card No": ["Card No", "card no_"]
}
equivalent_to_standard = {}
for standard, equivalents in field_equivalents.items():
    for equiv in equivalents:
        equivalent_to_standard[equiv.lower()] = standard

# --- Field cleaner ---
def clean_ocr_text(field, text):
    if not text:
        return ""

    for equiv in field_equivalents.get(field, []) + [field]:
        text = re.sub(rf"(?i)\b{re.escape(equiv)}\b", '', text)

    if "Date" in field:
        text = re.sub(r'[^0-9./a-zA-Z -]', '', text)
        match = re.search(r'\b\d{1,2}[./ -](?:\d{1,2}|[A-Za-z]{3,9})[./ -]\d{2,4}\b', text)
        return match.group() if match else ""

    if field == "Card No":
        text = re.sub(r'(?i)card no[_:\s-]*', '', text)
    if field == "Place of Birth":
        text = re.sub(r'(?i)place of binth[:\s-]*', '', text)
    if field == "Country":
        text = re.sub(r'(?i)(codeof issu.*|icode of state|code of state)', '', text)
    if field == "Passport No":
        text = re.sub(r'(?i)passport no[:\s-]*', '', text)
    if field == "Name":
        text = re.sub(r'(?i)given name[:\s-]*', '', text)
    if field == "Issuing Office":
        text = re.sub(r'(?i)(iss office|issuing authority|authority)', '', text)
    if field == "Type":
        text = re.sub(r'(?i)type[:\s-]*', '', text)

    return re.sub(r'[^A-Za-z0-9\s/-]', '', text).strip()

# --- Detect unknown new fields ---
def detect_unknown_fields(text):
    field_markers = {
        'Phone No': ['phone', 'mobile', 'tel'],
        'Driving License': ['driving', 'license', 'dl'],
        'Aadhar': ['aadhar', 'uid'],
        'Email': ['email', '@'],
        'Address': ['address', 'location', 'street']
    }
    detected = {}
    for field, markers in field_markers.items():
        if any(marker in text.lower() for marker in markers):
            detected[field] = text.split(':')[-1].strip() if ':' in text else text
    return detected

# --- Image processing ---
image_files = [os.path.join(image_input_path, f)
               for f in os.listdir(image_input_path)
               if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

for img_path in image_files:
    img_file = os.path.basename(img_path)
    base_name = os.path.splitext(img_file)[0]
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to read {img_path}")
        continue

    results = yolo_model(img)[0]
    boxes = results.boxes

    # --- First: process class 0,1,2 only ---
    raw_boxes = []
    for box in boxes:
        class_id = int(box.cls[0])
        if class_id in [0, 1, 2]:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            raw_boxes.append((x1, y1, x2, y2))
    if raw_boxes:
        filtered_boxes = non_max_suppression_area(raw_boxes, iou_thresh=0.4)
        text_output_path = os.path.join(final_output_dir, f"{base_name}.txt")
        audio_output_path = os.path.join(final_output_dir, f"{base_name}.mp3")
        img_final = img.copy()
        all_lines = []

        with open(text_output_path, "w", encoding="utf-8") as txt_file:
            for x1, y1, x2, y2 in filtered_boxes:
                crop = img[y1:y2, x1:x2]
                ocr_result = ocr_reader.readtext(crop, detail=0)
                for line in ocr_result:
                    cleaned_text = line.strip()
                    if cleaned_text:
                        txt_file.write(cleaned_text + "\n")
                        all_lines.append(cleaned_text)
                text_for_box = " | ".join(ocr_result)[:30]
                cv2.rectangle(img_final, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img_final, text_for_box, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Save annotated image
        out_img_path = os.path.join(final_output_dir, f"{base_name}.jpg")
        cv2.imwrite(out_img_path, img_final)

        # Generate and save audio
        if all_lines:
            joined_text = ". ".join(all_lines)
            tts = gTTS(joined_text)
            tts.save(audio_output_path)

    # --- Second: class 3+ fields ---
    raw_fields = defaultdict(set)
    all_texts = []
    img_idcard = img.copy()
    found_idcard = False
    for box in boxes:
        class_id = int(box.cls[0])
        if class_id not in class_map: continue
        found_idcard = True
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = img[y1:y2, x1:x2]
        ocr_result = ocr_reader.readtext(crop, detail=0)
        extracted_text = " ".join(ocr_result).strip()
        all_texts.append(extracted_text)
        field_name = class_map[class_id]
        cleaned = clean_ocr_text(field_name, extracted_text)
        raw_fields[field_name].add(cleaned)
        label = f"{field_name}: {cleaned}"
        cv2.rectangle(img_idcard, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_idcard, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

    if found_idcard:
        new_fields = {}
        for text in all_texts:
            new_fields.update(detect_unknown_fields(text))

        final_fields = {}
        for field, values in raw_fields.items():
            standard_field = equivalent_to_standard.get(field.lower(), field)
            cleaned_values = [v for v in values if len(v) >= 2 and not v.lower().startswith("iss ") and not v.lower().startswith("date of expiny")]
            if cleaned_values and standard_field not in final_fields:
                final_fields[standard_field] = max(cleaned_values, key=len)

        final_fields.update(new_fields)
        txt_path = os.path.join(idcard_output_dir, f"{base_name}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            for field, value in final_fields.items():
                f.write(f"{field}: {value}\n")
        cv2.imwrite(os.path.join(idcard_output_dir, f"{base_name}.jpg"), img_idcard)

    print(f"Processed: {img_file}")

print("All done. Check the 'final_outputs' and 'idcard_outputs' folders.")
