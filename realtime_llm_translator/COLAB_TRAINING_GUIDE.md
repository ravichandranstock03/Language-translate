# Google Colab Training Guide for Real-Time LLM Translator

This guide helps you train your custom translation LLM model on Google Colab with TPU/GPU acceleration.

## 🚀 Why Use Google Colab?

- **Free GPU/TPU Access**: Up to 12 hours per session (GPU) or specialized TPU pods
- **No Local Setup**: No need to install CUDA, drivers, or manage disk space
- **Pre-installed Libraries**: PyTorch, Transformers, Accelerate come pre-installed
- **Fast Training**: V100/A100 GPUs significantly speed up fine-tuning

## 📋 Prerequisites

1. A Google Account
2. Access to [Google Colab](https://colab.research.google.com/)
3. (Optional) Colab Pro for longer sessions and better GPUs (A100/V100)

## 🔗 Step-by-Step Instructions

### Step 1: Open a New Colab Notebook
1. Go to [colab.research.google.com](https://colab.research.google.com/)
2. Click **"New Notebook"**

### Step 2: Configure Runtime
1. Click **Runtime** → **Change runtime type**
2. **Hardware accelerator**: Select **GPU** (T4, P100, V100, or A100) or **TPU**
   - *Recommendation*: Use **GPU** for standard fine-tuning (better compatibility with `transformers`)
   - *TPU Note*: TPUs require specific XLA code modifications; GPU is easier for LoRA fine-tuning
3. **Runtime shape**: Select **High-RAM** if available (for larger models like NLLB-200 3.3B)
4. Click **Save**

### Step 3: Copy the Training Code
Copy the entire content of `train_custom_model.py` from this repository and paste it into the first cell of your Colab notebook.

### Step 4: Install Dependencies (If Needed)
Add this to the top of your notebook if packages are missing:

```python
!pip install transformers datasets accelerate peft bitsandbytes sentencepiece protobuf -q
```

### Step 5: Configure Parameters
Modify the `config` dictionary in the code cell to match your needs:

```python
config = {
    "model_name": "facebook/nllb-200-distilled-600M",  # Or "Helsinki-NLP/opus-mt-en-es"
    "output_dir": "./custom_translator_model",
    "dataset_name": "opus100",
    "source_lang": "eng_Latn",
    "target_lang": "spa_Latn",
    "num_train_epochs": 3,
    "per_device_train_batch_size": 16,
    "learning_rate": 2e-4,
    "use_lora": True,  # Recommended for Colab free tier
    "lora_r": 64,
    "max_samples": 10000,  # Limit dataset for faster training on free tier
}
```

### Step 6: Run the Training
Click the **Play** button (▶️) next to the cell.

**Expected Output:**
- Dataset downloading and preprocessing logs
- Model loading progress
- Training progress bar with loss metrics
- Final message: `✅ Model training complete! Saved to ./custom_translator_model`

### Step 7: Download Your Model
Once training finishes, run this cell to download your `.safetensors` model to your local machine:

```python
from google.colab import files
import os

output_dir = "./custom_translator_model"
print(f"Zipping model folder: {output_dir}...")

# Zip the directory
!zip -r custom_translator_model.zip {output_dir}

# Trigger download
files.download('custom_translator_model.zip')
print("✅ Download started! The zip contains config.json, model.safetensors, tokenizer files, etc.")
```

## ⚙️ Optimization Tips for Colab Free Tier

| Setting | Recommendation | Reason |
| :--- | :--- | :--- |
| **Model Size** | `distilled-600M` or `opus-mt` | Fits in 12GB VRAM; 3.3B might OOM |
| **Batch Size** | `8` or `16` | Prevents Out-Of-Memory errors |
| **Max Samples** | `5000` - `10000` | Keeps training under 2 hours |
| **LoRA** | `True` | Drastically reduces VRAM usage |
| **Precision** | `fp16` | Faster training, less memory |

## 🛑 Troubleshooting

### Error: "Runtime Disconnected" / "Out of Memory"
- **Cause**: Model too large or batch size too high.
- **Fix**: Reduce `per_device_train_batch_size` to 4 or 8. Switch to a smaller model like `Helsinki-NLP/opus-mt-en-es`.

### Error: "TPU Not Detected"
- **Note**: This script is optimized for GPU. If you specifically want TPU, you must enable XLA flags:
  ```python
  import os
  os.environ["TPU_USE_PRECOMPILED"] = "1"
  ```
  And use `accelerate launch` with TPU configuration. *Recommendation: Stick to GPU for simplicity.*

### Session Time Limit (12 Hours)
- Save checkpoints frequently by adding `save_steps=500` to `TrainingArguments`.
- If the session disconnects, you can resume from the last checkpoint saved in the output directory.

## 📂 What You Get
After downloading and unzipping `custom_translator_model.zip`, you will have:
- `adapter_model.safetensors` (if using LoRA) OR `model.safetensors` (if merged)
- `config.json`
- `tokenizer.json` & `special_tokens_map.json`
- `training_args.bin`

You can now load this folder in the `translator.py` inference script provided in the main repository!

```python
# Usage in your local inference script
translator = TextTranslator(model_path="./custom_translator_model", device="cuda")
```
