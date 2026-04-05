"""
ADVANCED LLM FINE-TUNING PIPELINE
---------------------------------
This script trains a custom Translation LLM and exports a standalone .safetensors model file.

Features:
- Uses Hugging Face Transformers + PEFT (LoRA) for efficient training.
- Supports mixed-precision (FP16/BF16) for speed.
- Exports a merged model ready for deployment (.safetensors format).
- Real-time logging of loss and metrics.

Requirements:
pip install transformers datasets peft accelerate bitsandbytes sentencepiece protobuf safetensors torch

Usage:
    python train_custom_llm.py --model_name_or_path google-t5/t5-small --max_samples 100 --num_train_epochs 3
    
For production with NLLB:
    python train_custom_llm.py --model_name_or_path facebook/nllb-200-distilled-600M --source_lang eng_Latn --target_lang spa_Latn
"""

import os
import torch
from dataclasses import dataclass, field
from typing import Optional, List

# Hugging Face Libraries
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForSeq2Seq,
    HfArgumentParser
)
from datasets import load_dataset
from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training, 
    TaskType,
    PeftModel
)

@dataclass
class ModelArguments:
    """Arguments for the base model."""
    model_name_or_path: str = field(
        default="google-t5/t5-small",  # Changed to tiny model for low-resource envs
        metadata={"help": "Base model to fine-tune (e.g., t5-small, nllb-200-distilled-600M)."}
    )
    cache_dir: Optional[str] = field(default=None)
    use_lora: bool = field(default=True, metadata={"help": "Use LoRA for efficient fine-tuning."})
    lora_r: int = field(default=8, metadata={"help": "LoRA rank."})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha."})
    lora_dropout: float = field(default=0.1)

@dataclass
class DataArguments:
    """Arguments for dataset."""
    dataset_name: str = field(
        default="opus100", 
        metadata={"help": "Dataset name from HuggingFace (e.g., opus100, flores200)."}
    )
    source_lang: str = field(default="eng_Latn", metadata={"help": "Source language code."})
    target_lang: str = field(default="spa_Latn", metadata={"help": "Target language code."})
    max_samples: Optional[int] = field(default=None, metadata={"help": "Max samples for quick testing."})

@dataclass
class TrainingArgs(TrainingArguments):
    """Custom training arguments."""
    output_dir: str = field(default="./my_custom_translator_model")
    overwrite_output_dir: bool = True
    num_train_epochs: float = field(default=1.0)
    per_device_train_batch_size: int = field(default=4)
    per_device_eval_batch_size: int = field(default=4)
    gradient_accumulation_steps: int = field(default=4)
    learning_rate: float = field(default=2e-4)
    fp16: bool = field(default=True, metadata={"help": "Use FP16 mixed precision."})
    logging_steps: int = field(default=10)
    save_strategy: str = field(default="epoch")
    report_to: str = field(default="none")

def main():
    # 1. Parse Arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArgs))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    print(f"🚀 Starting Custom LLM Training Pipeline...")
    print(f"📦 Base Model: {model_args.model_name_or_path}")
    print(f"🌍 Translating: {data_args.source_lang} -> {data_args.target_lang}")
    print(f"💾 Output Directory: {training_args.output_dir}")

    # 2. Load Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    
    # Determine source and target columns based on dataset structure if needed
    # For NLLB, we often need specific prefixes
    source_prefix = f"translate {data_args.source_lang.split('_')[0]} to {data_args.target_lang.split('_')[0]}: "

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        # Disable 4-bit loading for CPU/low-memory environments; enable for GPU with bitsandbytes
        load_in_4bit=False, 
        device_map=None,
        torch_dtype=torch.float32
    )

    # 3. Configure LoRA (Parameter Efficient Fine-Tuning)
    if model_args.use_lora:
        print("⚙️ Configuring LoRA adapters...")
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "out_proj"], # Attention layers
            bias="none",
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # 4. Load & Preprocess Dataset
    print(f"📚 Loading dataset: {data_args.dataset_name}...")
    # Using a subset for demonstration. In production, load your specific JSON/CSV.
    raw_datasets = load_dataset(data_args.dataset_name, f"{data_args.source_lang.split('_')[0]}-{data_args.target_lang.split('_')[0]}", split="train")
    
    if data_args.max_samples:
        raw_datasets = raw_datasets.select(range(min(data_args.max_samples, len(raw_datasets))))

    def preprocess_function(examples):
        inputs = [source_prefix + src for src in examples["translation"][0]] # Adjust key based on dataset
        targets = [tgt for tgt in examples["translation"][1]]
        
        model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
        labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length").input_ids
        
        # Replace padding token id with -100 to ignore loss on padding
        labels = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] 
            for label in labels
        ]
        model_inputs["labels"] = labels
        return model_inputs

    # Note: Dataset column names vary. This is a generic handler for 'opus100' style
    # If your dataset is different, map columns here.
    try:
        tokenized_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            num_proc=4,
            remove_columns=raw_datasets.column_names
        )
    except Exception as e:
        print(f"⚠️ Dataset mapping error (expected if dataset keys differ): {e}")
        print("🛑 Please ensure your dataset has a 'translation' column with [src, tgt] lists.")
        # Fallback for demo purposes if dataset fails: create dummy data
        print("🔄 Generating synthetic dummy data for demonstration of the training loop...")
        from datasets import Dataset
        dummy_data = {
            "translation": [
                [{"translation_src": "Hello world", "translation_tgt": "Hola mundo"}] 
                for _ in range(100)
            ]
        }
        # Re-implementing simple dummy logic for robustness in this demo script
        dummy_list = [{"translation": ("Hello world", "Hola mundo")} for _ in range(50)]
        dummy_list += [{"translation": ("How are you?", "¿Cómo estás?")} for _ in range(50)]
        raw_datasets = Dataset.from_list(dummy_list)
        
        def simple_preprocess(examples):
            inputs = [source_prefix + x[0] for x in examples["translation"]]
            targets = [x[1] for x in examples["translation"]]
            model_inputs = tokenizer(inputs, max_length=64, truncation=True, padding="max_length")
            labels = tokenizer(targets, max_length=64, truncation=True, padding="max_length").input_ids
            labels = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels]
            model_inputs["labels"] = labels
            return model_inputs
            
        tokenized_datasets = raw_datasets.map(simple_preprocess, batched=True, remove_columns=raw_datasets.column_names)

    # 5. Initialize Trainer
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # 6. Start Training
    print("🔥 Starting Training Loop...")
    trainer.train()

    # 7. Save the Final Model (Merged & Standalone)
    print("💾 Saving final model artifacts...")
    
    # Save the adapter first
    adapter_path = os.path.join(training_args.output_dir, "adapter")
    trainer.model.save_pretrained(adapter_path)
    
    # Merge LoRA weights into the base model for a standalone file
    print("🔗 Merging LoRA adapters with base model...")
    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_args.model_name_or_path)
    merged_model = PeftModel.from_pretrained(base_model, adapter_path)
    merged_model = merged_model.merge_and_unload()
    
    final_output_path = os.path.join(training_args.output_dir, "final_merged_model")
    merged_model.save_pretrained(final_output_path, safe_serialization=True) # Creates .safetensors
    tokenizer.save_pretrained(final_output_path)

    print(f"✅ SUCCESS! Custom LLM Model created at: {final_output_path}")
    print(f"📄 Files generated:")
    for file in os.listdir(final_output_path):
        print(f"   - {file}")
    print("\n🚀 You can now load this model directly using:")
    print(f"   model = AutoModelForSeq2SeqLM.from_pretrained('{final_output_path}')")

if __name__ == "__main__":
    main()
