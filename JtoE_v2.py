# ------------------------------------------------------------
#  JAPANESE OCR + AUTO-TRANSLATION (ONE SINGLE SCRIPT)
#  OCR (pyautogui + pytesseract) -> translate (M2M100)
# ------------------------------------------------------------

import pyautogui
import pytesseract
from PIL import Image
import time, datetime, os
import cv2
import numpy as np
import torch

from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

# ------------------------------------------------------------
# SETUP
# ------------------------------------------------------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

LANG = "jpn"
JPN_FILE = "japanese_text.txt"
ENG_FILE = "translated_english.txt"

os.makedirs("screenshots", exist_ok=True)

# ------------------------------------------------------------
# LOAD TRANSLATION MODEL
# ------------------------------------------------------------
print("Loading translation model... (this may take a moment)")

model_name = "facebook/m2m100_1.2B"
tokenizer = M2M100Tokenizer.from_pretrained(model_name)
model = M2M100ForConditionalGeneration.from_pretrained(model_name)

tokenizer.src_lang = "ja"
target_lang_id = tokenizer.get_lang_id("en")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(f"✅ Translation device: {device}")

# ------------------------------------------------------------
# REGION SELECTOR
# ------------------------------------------------------------
def select_region():
    print("🟡 Move mouse to TOP-LEFT corner and press Enter.")
    input()
    x1, y1 = pyautogui.position()

    print("🟡 Move mouse to BOTTOM-RIGHT corner and press Enter.")
    input()
    x2, y2 = pyautogui.position()

    if x2 <= x1 or y2 <= y1:
        print("❌ Invalid region selected — reselect.")
        return select_region()

    width = x2 - x1
    height = y2 - y1

    print(f"📌 Region selected: ({x1}, {y1}, {width}, {height})")
    return (x1, y1, width, height)

# ------------------------------------------------------------
# OCR PREPROCESSING (more stable)
# ------------------------------------------------------------
def preprocess(img):
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)

    img = cv2.GaussianBlur(img, (3, 3), 0)

    # Adaptive threshold (better for game text)
    img = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        31, 8
    )

    return Image.fromarray(img)

# ------------------------------------------------------------
# TRANSLATION
# ------------------------------------------------------------
def translate_text(text):
    """Translate Japanese → English"""
    if not text.strip():
        return ""

    batch = tokenizer([text], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        output = model.generate(**batch, forced_bos_token_id=target_lang_id)

    return tokenizer.batch_decode(output, skip_special_tokens=True)[0]

# ------------------------------------------------------------
# MAIN LOOP
# ------------------------------------------------------------
def main():
    region = select_region()
    last_jpn = ""

    print("\n🟢 OCR + Translation started (Ctrl + C to stop)\n")

    try:
        while True:
            shot = pyautogui.screenshot(region=region)
            processed = preprocess(shot)

            jpn = pytesseract.image_to_string(processed, lang=LANG).strip()

            # Skip blank or identical text
            if len(jpn) < 2 or jpn == last_jpn:
                time.sleep(1.5)
                continue

            last_jpn = jpn

            # Save Japanese
            with open(JPN_FILE, "w", encoding="utf-8") as f:
                f.write(jpn)

            print(f"📝 OCR updated at {datetime.datetime.now().strftime('%H:%M:%S')}")

            # Translate
            eng = translate_text(jpn)

            with open(ENG_FILE, "w", encoding="utf-8") as f:
                f.write(eng)

            print("🌐 Translation updated!\n")

            time.sleep(2)

    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")

# ------------------------------------------------------------
if __name__ == "__main__":
    main()
