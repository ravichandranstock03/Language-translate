# 🚀 Custom LLM Translator Model - Training & Output

## Overview
This repository contains an **advanced LLM fine-tuning pipeline** that creates a custom translation model and exports it as a standalone `.safetensors` file.

## ✅ What This Code Does
1. **Downloads** a pre-trained base model (e.g., NLLB-200, T5, M2M100)
2. **Fine-tunes** it on your custom dataset using LoRA (efficient training)
3. **Merges** the adapter weights with the base model
4. **Exports** a complete, standalone model file in `.safetensors` format

## 📁 Generated Model Files
After running the training script, you will get a folder like `my_custom_translator_model/final_merged_model/` containing:

```
final_merged_model/
├── config.json           # Model architecture configuration
├── model.safetensors     # ⭐ THE MAIN MODEL WEIGHTS (safe, fast loading)
├── tokenizer_config.json # Tokenizer settings
├── special_tokens_map.json
├── spiece.model          # SentencePiece vocabulary (for T5/NLLB)
└── tokenizer.json        # Fast tokenizer mapping
```

### Key File: `model.safetensors`
- **Format**: Safe Tensors (`.safetensors`) - modern, secure alternative to PyTorch `.bin`
- **Size**: Varies by model (T5-Small ~240MB, NLLB-600M ~2.4GB)
- **Usage**: Load directly with Hugging Face Transformers

## 🛠️ How to Run

### 1. Install Dependencies
```bash
pip install transformers datasets peft accelerate bitsandbytes sentencepiece protobuf safetensors torch
```

### 2. Run Training (Quick Test)
```bash
python train_custom_llm.py \
    --model_name_or_path google-t5/t5-small \
    --source_lang en \
    --target_lang es \
    --max_samples 100 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --output_dir ./my_custom_translator_model
```

### 3. Run Training (Production with NLLB)
```bash
python train_custom_llm.py \
    --model_name_or_path facebook/nllb-200-distilled-600M \
    --source_lang eng_Latn \
    --target_lang spa_Latn \
    --max_samples 10000 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --fp16 True \
    --output_dir ./nllb_custom_model
```

## 📊 Expected Training Output

```
🚀 Starting Custom LLM Training Pipeline...
📦 Base Model: facebook/nllb-200-distilled-600M
🌍 Translating: eng_Latn -> spa_Latn
💾 Output Directory: ./nllb_custom_model
⚙️ Configuring LoRA adapters...
trainable params: 4,194,304 || all params: 608,234,752 || trainable%: 0.6896
📚 Loading dataset: opus100...
🔥 Starting Training Loop...

Step  | Loss   | Learning Rate
------|--------|---------------
10    | 2.341  | 0.0002
20    | 1.892  | 0.0002
50    | 1.234  | 0.0002
100   | 0.876  | 0.0002

✅ Saving final model artifacts...
🔗 Merging LoRA adapters with base model...
✅ SUCCESS! Custom LLM Model created at: ./nllb_custom_model/final_merged_model

📄 Files generated:
   - config.json
   - model.safetensors      <-- YOUR CUSTOM MODEL
   - tokenizer_config.json
   - spiece.model
   - tokenizer.json

🚀 You can now load this model directly using:
   model = AutoModelForSeq2SeqLM.from_pretrained('./nllb_custom_model/final_merged_model')
```

## 🔬 How to Use Your Custom Model

### Load and Translate
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load YOUR custom trained model
model_path = "./my_custom_translator_model/final_merged_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

# Move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Translate
text = "Hello, how are you?"
inputs = tokenizer(text, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_length=128)
translation = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"Input: {text}")
print(f"Translation: {translation}")
```

### Deploy with FastAPI (Real-Time API)
```python
from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

app = FastAPI()
model_path = "./my_custom_translator_model/final_merged_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

@app.post("/translate")
async def translate(text: str, target_lang: str = "es"):
    inputs = tokenizer(f"translate English to {target_lang}: {text}", return_tensors="pt")
    outputs = model.generate(**inputs, max_length=128)
    return {"translation": tokenizer.decode(outputs[0], skip_special_tokens=True)}
```

## 🎯 Performance Benchmarks

| Model | Size | VRAM Required | Inference Speed (GPU) | Languages |
|-------|------|---------------|----------------------|-----------|
| T5-Small | 240MB | 1GB | ~50ms/sentence | Custom pairs |
| NLLB-200 Distilled | 2.4GB | 4GB | ~80ms/sentence | 200+ |
| M2M100-418M | 1.6GB | 3GB | ~70ms/sentence | 100+ |

## 📝 Notes
- **Disk Space**: Ensure you have at least 5GB free for NLLB models
- **RAM**: Minimum 8GB recommended (16GB+ for large models)
- **GPU**: NVIDIA GPU with 4GB+ VRAM recommended for fast training
- **LoRA**: Uses parameter-efficient fine-tuning (only 0.5-1% of weights trained)

## 📂 Project Structure
```
/workspace/
├── train_custom_llm.py       # Main training script
├── my_custom_translator_model/
│   └── final_merged_model/   # OUTPUT: Your custom .safetensors model
│       ├── model.safetensors
│       ├── config.json
│       └── ...
└── README_MODEL.md           # This file
```

## 🎉 Success Criteria
You'll know it worked when you see:
1. ✅ `model.safetensors` file exists in output directory
2. ✅ File size > 100MB (indicates real weights, not empty)
3. ✅ Model loads without errors in Python
4. ✅ Translation produces coherent output

---

**Created with Advanced LLM Fine-Tuning Pipeline using PEFT + LoRA + Transformers**
