# ------------------------------------------------------------
# JAPANESE → ENGLISH TRANSLATOR (UNLIMITED TEXT)
# using facebook/m2m100_1.2B
# ------------------------------------------------------------


from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from tqdm import tqdm
import torch

# Load model and tokenizer
model_name = "facebook/m2m100_1.2B"
print("Loading model... (this may take a minute)")
tokenizer = M2M100Tokenizer.from_pretrained(model_name)
model = M2M100ForConditionalGeneration.from_pretrained(model_name)

# Set source and target languages
tokenizer.src_lang = "ja"
target_lang_id = tokenizer.get_lang_id("en")

# Function to translate a chunk of Japanese text
def translate_text(text):
    encoded = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        generated_tokens = model.generate(**encoded, forced_bos_token_id=target_lang_id)
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

# Input / output file paths
input_file = "japanese_text.txt"         # <-- Replace with your filename
output_file = "translated_english.txt"

# Read Japanese text
with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Translate each line (in chunks)
translated_lines = []
for line in tqdm(lines, desc="Translating"):
    if line.strip():  # skip empty lines
        translated = translate_text(line.strip())
        translated_lines.append(translated + "\n")
    else:
        translated_lines.append("\n")

# Write output to file
with open(output_file, "w", encoding="utf-8") as f:
    f.writelines(translated_lines)

print(f"\n✅ Translation complete! Saved as: {output_file}")
