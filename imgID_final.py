import cv2
import easyocr
from ultralytics import YOLO
import re
from collections import defaultdict
from gtts import gTTS

# === Config ===
image_input_path = "image.png"
model_path = "mymodel.pt"

yolo_model = YOLO(model_path)
ocr_reader = easyocr.Reader(['en'])

def iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    unionArea = boxAArea + boxBArea - interArea
    return interArea / unionArea if unionArea else 0

def non_max_suppression_area(boxes, iou_thresh=0.4):
    boxes = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    final_boxes = []
    for box in boxes:
        if all(iou(box, kept) < iou_thresh for kept in final_boxes):
            final_boxes.append(box)
    return final_boxes

class_map = {
    3: "Surname", 4: "Name", 5: "Nationality", 6: "Sex", 7: "Date of Birth",
    8: "Place of Birth", 9: "Issue Date", 10: "Expiry Date", 11: "Issuing Office",
    12: "Height", 13: "Type", 14: "Country", 15: "Passport No", 16: "Personal No", 17: "Card No"
}

field_equivalents = {
    "Country": ["Country", "Code of State", "Code of Issuing State", "Codeof Issulng State", "ICode of State"],
    "Issuing Office": ["Issuing Office", "Issuing Authority", "Issuing office", "Iss. office", "Authority", "authoricy", "Iss office", "issuing authority"],
    "Passport No": ["Passport No", "Document No", "IPassport No","Passport Number"],
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

def clean_ocr_text(field, text):
    if not text:
        return ""
    for equiv in field_equivalents.get(field, []) + [field]:
        text = re.sub(rf"(?i)\b{re.escape(equiv)}\b", '', text)

    if "Date" in field:
        text = re.sub(r'[^0-9./a-zA-Z -]', '', text)
        match = re.search(r'\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b', text)
        return match.group() if match else ""

    return re.sub(r'[^A-Za-z0-9\s/-]', '', text).strip()

def detect_unknown_fields(text):
    field_markers = {
        'Phone No': ['phone', 'mobile', 'tel'],
        'Driving License': ['driving', 'license'],
        'Aadhar': ['aadhar', 'uid'],
        'Email': ['email', '@'],
        'Address': ['address', 'location', 'street']
    }
    detected = {}
    for field, markers in field_markers.items():
        if any(marker in text.lower() for marker in markers):
            detected[field] = text.split(':')[-1].strip() if ':' in text else text
    return detected

# === Processing ===
img = cv2.imread(image_input_path)
if img is None:
    print(f"❌ Failed to read {image_input_path}")
    exit()

results = yolo_model(img)[0]
boxes = results.boxes

# === Class 0/1/2 → TTS ===
raw_boxes = []
for box in boxes:
    class_id = int(box.cls[0])
    if class_id in [0, 1, 2]:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        raw_boxes.append((x1, y1, x2, y2))

if raw_boxes:
    filtered_boxes = non_max_suppression_area(raw_boxes)
    audio_text = []
    for x1, y1, x2, y2 in filtered_boxes:
        crop = img[y1:y2, x1:x2]
        ocr_result = ocr_reader.readtext(crop, detail=0)
        for line in ocr_result:
            cleaned = line.strip()
            if cleaned:
                audio_text.append(cleaned)
    if audio_text:
        try:
            tts = gTTS(text=" ".join(audio_text), lang='en')
            tts.save("audio.mp3")
            print("🔊 Saved audio as 'audio.mp3'")
        except Exception as e:
            print("❌ TTS error:", e)

# === Class 3+ → Field Extraction ===
raw_fields = defaultdict(set)
all_texts = []
found_idcard = False

for box in boxes:
    class_id = int(box.cls[0])
    if class_id not in class_map:
        continue
    found_idcard = True
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    crop = img[y1:y2, x1:x2]
    ocr_result = ocr_reader.readtext(crop, detail=0)
    extracted = " ".join(ocr_result).strip()
    field = class_map[class_id]
    cleaned_value = clean_ocr_text(field, extracted)
    all_texts.append(extracted)
    if cleaned_value:
        raw_fields[field].add(cleaned_value)

if found_idcard:
    final_fields = {}
    for field, values in raw_fields.items():
        standard_field = equivalent_to_standard.get(field.strip().lower(), field.strip())
        filtered_values = [v for v in values if len(v) > 1]
        if filtered_values:
            final_fields[standard_field] = max(filtered_values, key=len)

    for text in all_texts:
        final_fields.update(detect_unknown_fields(text))

    print("\n📋 Final extracted fields:")
    for field, value in final_fields.items():
        print(f"{field}: {value}")

    with open("data.txt", "w", encoding="utf-8") as f:
        for field, value in final_fields.items():
            f.write(f"{field}: {value}\n")
    print("✅ Saved to data.txt")

    # === Google Form Filling ===
    answer = input("\n📨 Do you want to fill a Google Form? (yes/no): ").strip().lower()
    if answer == "yes":
        form_url = input("🔗 Paste Google Form URL: ").strip()
        if form_url:
            try:
                from selenium import webdriver
                from selenium.webdriver.common.by import By
                import time
                from difflib import SequenceMatcher

                def similar(a, b):
                    return SequenceMatcher(None, a, b).ratio()

                def parse_data(path):
                    result = {}
                    with open(path, 'r', encoding='utf-8') as f:
                        for line in f:
                            if ':' in line:
                                key, value = line.split(':', 1)
                                result[key.strip().lower()] = value.strip()
                    return result

                # 🔁 Field alias mapping
                field_aliases = {
                    "passport no": ["passport number", "document number", "passport num"],
                    "date of birth": ["dob", "birthdate"],
                    "issue date": ["issued on", "date of issue"],
                    "expiry date": ["expires on", "expiration date"],
                    "personal no": ["national id", "id number"]
                }

                data = parse_data("data.txt")
                web = webdriver.Chrome()
                web.get(form_url)
                time.sleep(5)

                questions = web.find_elements(By.CSS_SELECTOR, 'div[role="listitem"]')
                for q in questions:
                    try:
                        q_text = q.find_element(By.CSS_SELECTOR, 'div[role="heading"]').text.lower().strip()
                        input_field = q.find_element(By.CSS_SELECTOR, 'input[type="text"]')
                        filled = False

                        for key, val in data.items():
                            key_lower = key.strip().lower()
                            aliases = field_aliases.get(key_lower, [])
                            all_variants = [key_lower] + aliases

                            for variant in all_variants:
                                if variant == q_text or variant in q_text.split() or (' ' in variant and variant in q_text):
                                    input_field.send_keys(val)
                                    print(f"📝 Filled: {key} → {val}")
                                    filled = True
                                    break
                            if filled:
                                break

                        if not filled:
                            for key, val in data.items():
                                if similar(key.lower(), q_text) > 0.8:
                                    input_field.send_keys(val)
                                    print(f"📝 Filled (Fuzzy): {key} → {val}")
                                    break
                    except Exception:
                        continue

                print("✅ Form filled. Please review and submit manually.")
                time.sleep(100)
            except Exception as e:
                print("❌ Selenium form fill error:", e)

# Sample Form: https://docs.google.com/forms/d/1rzOsfn_ZC6slV9hag_VA2QXZRdD9qGOu-SdJ9THaULQ/viewform?edit_requested=true
