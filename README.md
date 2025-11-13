# 🈺 JtoE-Tharjama

**JtoE-Tharjama** is a Japanese → English translator designed primarily for Japanese RPG (JRPG) games, but it can be used for any general translation purpose.

---

## 🧠 Model

This project uses the **facebook/m2m100_1.2B** pretrained model, which provides relatively reliable translations for free.

> ⚠️ Requires at least **10 GB of free storage** for the model.

Performance improves if your system supports **CUDA** (GPU acceleration).

Check CUDA availability:
```bash
nvcc --version
```

Use the provided requirements.txt file to install the Dependencies :
``` bash
pip install -r requirements.txt
```

🪟 Step 1 — Run the OCR script

Select the screen region for automatic text capture:
```bash
python ocrimage.py
```

💬 Step 2 — Run the Translator

Run this in a separate terminal:
``` bash
python Translator.py
```


📂 Output
Translated text will be written to:

Translated_english.txt



