import cv2
import easyocr
from ultralytics import YOLO
import re
from collections import defaultdict
from gtts import gTTS
import tkinter as tk
from tkinter import filedialog
import platform
import os

def get_image_from_user():
    print("\nHow would you like to provide the image?")
    print("1. Upload from device")
    print("2. Capture from webcam")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        root = tk.Tk()
        root.withdraw()
        image_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")]
        )
        if not image_path:
            print("No file selected.")
            return None
        print(f"Selected file: {image_path}")
        return image_path

    elif choice == "2":
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Could not open webcam.")
            return None

        print("Press SPACE to capture, ESC to cancel.")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture image.")
                break
            cv2.imshow("Capture Image", frame)
            key = cv2.waitKey(1)
            if key == 27:  # ESC
                print("Capture cancelled.")
                cap.release()
                cv2.destroyAllWindows()
                return None
            elif key == 32:  # SPACE
                image_path = "captured_image.jpg"
                cv2.imwrite(image_path, frame)
                print(f"Image captured and saved as {image_path}")
                cap.release()
                cv2.destroyAllWindows()
                return image_path
    else:
        print("Invalid choice.")
        return None

# === Get image path ===
image_input_path = get_image_from_user()
if not image_input_path:
    exit()

model_path = "Mymodel.pt"

yolo_model = YOLO(model_path)
ocr_reader = easyocr.Reader(['en'])

def play_audio(file_path="audio.mp3"):
    system = platform.system()
    try:
        if system == "Windows":
            os.system(f'start {file_path}')
        elif system == "Darwin":  # macOS
            os.system(f'open {file_path}')
        else:  # Linux
            os.system(f'xdg-open {file_path}')
    except Exception as e:
        print("Failed to play audio:", e)


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
    1: "Surname", 2: "Name", 3: "Nationality", 4: "Sex", 5: "Date of Birth",
    6: "Place of Birth", 7: "Issue Date", 8: "Expiry Date", 9: "Issuing Office",
    10: "Height", 11: "Type", 12: "Country", 13: "Passport No",
    14: "Personal No", 15: "Card No"
}

field_equivalents = {
    "Country": ["Country", "Code of State", "Code of Issuing State", "Codeof Issulng State", "ICode of State"],
    "Issuing Office": ["Issuing Office", "Issuing Authority", "Issuing office", "Iss. office", "Authority", "authoricy", "Iss office", "issuing authority"],
    "Passport No": ["Passport No", "Document No", "IPassport No","Passport Number"],
    "Personal No": ["Personal No", "National ID"],
    "Date of Birth": ["Date of Birth", "DOB", "Date ofbimn", "of birth", "ofbimn", "of pirth"],
    "Issue Date": ["Issue Date", "Date of Issue", "dale"],
    "Expiry Date": ["Expiry Date", "Date of Expiry", "of expiny"],
    "Name": ["Name", "Given Name", "Given", "nane", "Given name","IGiven"],
    "Surname": ["Surname", "Last Name", "Sumname"],
    "Place of Birth": ["Place of Birth", "Place of binth"],
    "Card No": ["Card No", "card no_"],
    "Nationality":["Nationalitv"],
    "Height":["IHeight"],
    "Sex":["ISex"]
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
    print(f"Failed to read {image_input_path}")
    exit()

results = yolo_model(img)[0]
boxes = results.boxes

# === Class 0/1/2 → TTS ===
raw_boxes = []
for box in boxes:
    class_id = int(box.cls[0])
    if class_id == 0 :
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
            text_to_speak = " ".join(audio_text)
            tts = gTTS(text=text_to_speak, lang='en')
            tts.save("audio.mp3")
            print("Saved audio as 'audio.mp3'")

            with open("t2s.txt", "w", encoding="utf-8") as f:
                f.write(text_to_speak)
            print("Saved spoken text to 't2s.txt'")

            play_audio("audio.mp3")
        except Exception as e:
            print("TTS error:", e)

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

    print("\nFinal extracted fields:")
    for field, value in final_fields.items():
        print(f"{field}: {value}")

    with open("data.txt", "w", encoding="utf-8") as f:
        for field, value in final_fields.items():
            f.write(f"{field}: {value}\n")
    print("Saved to data.txt")

    # === Google Form Filling ===
    answer = "yes"
    if answer == "yes":
        form_url = input("Paste Google Form URL: ").strip()
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

                def best_field_match(question_text, data_dict, aliases, threshold=0.7, used_fields=None):
                    best_match = None
                    best_score = 0
                    question_text = question_text.lower()

                    for field_key, field_val in data_dict.items():
                        field_key_lower = field_key.lower()
                        if used_fields and field_key_lower in used_fields:
                            continue
                        field_variants = [field_key_lower] + aliases.get(field_key_lower, [])
                        for variant in field_variants:
                            score = SequenceMatcher(None, variant, question_text).ratio()
                            if score > best_score:
                                best_score = score
                                best_match = (field_key, field_val, score)
                    if best_score >= threshold:
                        return best_match
                    return None

                # Field alias mapping
                field_aliases = {
                    "passport no": ["passport number", "document number", "passport num"],
                    "date of birth": ["dob", "birthdate"],
                    "issue date": ["issued on", "date of issue"],
                    "expiry date": ["expires on", "expiration date"],
                    "personal no": ["national id", "id number"]
                }

                data = parse_data("data.txt")
                used_fields = set()
                web = webdriver.Chrome()
                web.get(form_url)
                time.sleep(5)

                questions = web.find_elements(By.CSS_SELECTOR, 'div[role="listitem"]')
                for q in questions:
                    try:
                        q_text = q.find_element(By.CSS_SELECTOR, 'div[role="heading"]').text.lower().strip()
                        input_field = q.find_element(By.CSS_SELECTOR, 'input[type="text"]')
                        # === Manual Fix for Place of Birth and Date of Birth ===
                        if "place of birth" in q_text:
                            val = data.get("place of birth")
                            if val:
                                input_field.send_keys(val)
                                used_fields.add("place of birth")
                                print(f"Filled (Manual): Place of Birth → {val}")
                                continue
                        elif "date of birth" in q_text or "dob" in q_text:
                            if "date of birth" in data:
                                val = data["date of birth"]
                                input_field.send_keys(val)
                                used_fields.add("date of birth")
                                print(f"Filled (Manual): Date of Birth → {val}")
                            else:
                                input_field.send_keys("")  # Leave blank if not found
                                used_fields.add("date of birth")
                                print("Date of Birth not found in data.txt — left blank")
                            continue

                        match = best_field_match(q_text, data, field_aliases, threshold=0.7, used_fields=used_fields)
                        if match:
                            key, val, score = match
                            input_field.send_keys(val)
                            used_fields.add(key.lower())
                            print(f"Filled: {key} → {val} (Score: {score:.2f})")
                            continue

                        # Fallback: fuzzy match with all keys
                        for key, val in data.items():
                            if key.lower() in used_fields:
                                continue
                            if similar(key.lower(), q_text) > 0.8:
                                input_field.send_keys(val)
                                used_fields.add(key.lower())
                                print(f"Filled (Fuzzy): {key} → {val}")
                                break
                    except Exception:
                        continue

                print("Form filled. Please review and submit manually.")
                time.sleep(100)
            except Exception as e:
                print("Selenium form fill error:", e)

#sample form- https://docs.google.com/forms/d/e/1FAIpQLScvLDMeaMVU6V-6HVlM_EL6E7Ilwz1lrseiJXhb_DXaMSQcYA/viewform?usp=sharing&ouid=104507749809956660199
