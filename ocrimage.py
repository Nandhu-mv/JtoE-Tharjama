import pyautogui
import pytesseract
from PIL import Image
import time
import datetime
import cv2
import numpy as np

# ---- PATH TO TESSERACT ----
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

lang = "jpn"  # Japanese OCR
output_file = "japanese_text.txt"


def select_region():
    """Let user drag-select an area of the screen."""
    print("🟡 Move mouse to top-left corner and press Enter...")
    input()
    x1, y1 = pyautogui.position()
    print(f"Top-left: {x1}, {y1}")

    print("🟡 Move mouse to bottom-right corner and press Enter...")
    input()
    x2, y2 = pyautogui.position()
    print(f"Bottom-right: {x2}, {y2}")

    width, height = x2 - x1, y2 - y1
    print(f"✅ Selected region: ({x1}, {y1}, {width}, {height})")
    return (x1, y1, width, height)


def preprocess_image(image):
    """Optional: make text clearer before OCR."""
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    img_cv = cv2.threshold(img_cv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return Image.fromarray(img_cv)


def main():
    region = select_region()
    image_count = 0

    print("\n🟢 Starting capture every 5 seconds...")
    print("   Press Ctrl + C to stop.\n")

    try:
        while True:
            img = pyautogui.screenshot(region=region)
            img = preprocess_image(img)
            
            if image_count == 0:
                print("📸 First image captured for OCR processing.")
                img_name = f"screenshots/captured_area.png"
                img.save(img_name)
                image_count += 1

            text = pytesseract.image_to_string(img, lang=lang)
            timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")

            with open(output_file, "w", encoding="utf-8") as f:
                f.write(f"{timestamp}\n{text}\n{'-'*50}\n")

            print(f"🖼️ Captured and OCR'd at {timestamp}")
            time.sleep(10)

    except KeyboardInterrupt:
        print("\n🛑 Stopped. Text saved in:", output_file)


if __name__ == "__main__":
    main()
