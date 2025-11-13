# ------------------------------------------------------------
# JAPANESE → ENGLISH AUTO TRANSLATOR (OPTIMIZED REFRESH LOOP)
# using facebook/m2m100_1.2B
# ------------------------------------------------------------

from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from datetime import datetime
from tqdm import tqdm
import torch, time, os

# ------------------------------------------------------------
# MODEL SETUP
# ------------------------------------------------------------
model_name = "facebook/m2m100_1.2B"
print("Loading model... (this may take a minute)")

tokenizer = M2M100Tokenizer.from_pretrained(model_name)
model = M2M100ForConditionalGeneration.from_pretrained(model_name)

tokenizer.src_lang = "ja"
target_lang_id = tokenizer.get_lang_id("en")

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"✅ Using device: {device}\n")

# ------------------------------------------------------------
# TRANSLATION FUNCTIONS
# ------------------------------------------------------------

def log(msg):
    """Simple timestamped logging"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def translate_batch(texts):
    """Batch translation for efficiency"""
    encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        generated = model.generate(**encoded, forced_bos_token_id=target_lang_id)
    return tokenizer.batch_decode(generated, skip_special_tokens=True)

# ------------------------------------------------------------
# FILE SETUP
# ------------------------------------------------------------
input_file = "japanese_text.txt"
output_file = "translated_english.txt"

# Resume from existing output file (if any)
if os.path.exists(output_file):
    with open(output_file, "r", encoding="utf-8") as f:
        prevf = f
    log(f"About to translate")
else:
    prevf = ""
    log("Starting fresh translation session.")

# ------------------------------------------------------------
# MAIN LOOP
# ------------------------------------------------------------
log("🔁 Watching for new Japanese text... (Press Ctrl + C to stop)\n")

try:
    while True:
        if not os.path.exists(input_file):
            log(f"⚠️ File '{input_file}' not found. Waiting...")
            time.sleep(10)
            continue

        with open(input_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        new_lines = [line for line in lines[0:] if line.strip() or line == "\n"]

        if new_lines and (prevf != new_lines):
            log(f"🔎 Found {len(new_lines)} new line(s). Translating...\n")

            # Clean non-empty lines for batch processing
            text_batch = [line.strip() for line in new_lines if line.strip()]
            translated_batch = translate_batch(text_batch) if text_batch else []

            # Reinsert blank lines in the right positions
            translated_lines = []
            idx = 0
            for line in new_lines:
                if line.strip():
                    translated_lines.append(translated_batch[idx] + "\n")
                    idx += 1
                else:
                    translated_lines.append("\n")

            # Write output
            with open(output_file, "w", encoding="utf-8") as f:
                f.writelines(translated_lines)

            prevf = f
            log(f"✅ Added {len(new_lines)} translation(s) → {output_file}\n")

        time.sleep(2)

except KeyboardInterrupt:
    log("🛑 Stopped by user. Exiting safely...")
