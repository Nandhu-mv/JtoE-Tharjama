# ------------------------------------------------------------
# JAPANESE → ENGLISH TRANSLATOR (AUTO-REFRESH LOOP)
# using facebook/m2m100_1.2B
# ------------------------------------------------------------

from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from tqdm import tqdm
import torch, time, os

# Model setup
model_name = "facebook/m2m100_1.2B"
print("Loading model... (this may take a minute)")
tokenizer = M2M100Tokenizer.from_pretrained(model_name)
model = M2M100ForConditionalGeneration.from_pretrained(model_name)

tokenizer.src_lang = "ja"
target_lang_id = tokenizer.get_lang_id("en")

# Translation function
def translate_text(text):
    encoded = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        generated_tokens = model.generate(**encoded, forced_bos_token_id=target_lang_id)
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

# File paths
input_file = "japanese_text.txt"
output_file = "translated_english.txt"

# Track how many lines we've already translated
translated_count = 0

print("\n🔁 Watching for new text... (Press Ctrl + C to stop)\n")

try:
    while True:
        if not os.path.exists(input_file):
            print(f"⚠️ File '{input_file}' not found. Waiting...")
            time.sleep(10)
            continue

        with open(input_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Translate only new lines
        new_lines = lines[translated_count:]
        if new_lines:
            print(f"🔎 Found {len(new_lines)} new line(s). Translating...\n")
            translated_lines = []
            for line in tqdm(new_lines, desc="Translating"):
                if line.strip():
                    translated = translate_text(line.strip())
                    translated_lines.append(translated + "\n")
                else:
                    translated_lines.append("\n")

            # Append translations to file
            with open(output_file, "a", encoding="utf-8") as f:
                f.writelines(translated_lines)

            translated_count += len(new_lines)
            print(f"✅ Added {len(new_lines)} new translation(s) to {output_file}\n")

        # Wait before checking again
        time.sleep(5)

except KeyboardInterrupt:
    print("\n🛑 Stopped by user. Exiting safely...")
