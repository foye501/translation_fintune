import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Change this to your actual Qwen3 4B local path or HF repo ID when available
MODEL_NAME_OR_PATH = "Qwen/Qwen2.5-3B-Instruct"  
DATASET_PATH = "qwen_sft_dataset.jsonl"
OUTPUT_DIR = "./qwen_lora_output"

# LoRA Params
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Training Params
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
MAX_SEQ_LENGTH = 1024

def main():
    # 1. Determine Device Strategy (MPS for Mac, CUDA for Nvidia)
    if torch.cuda.is_available():
        device = "cuda"
        torch_dtype = torch.bfloat16 # modern GPUs support bf16
    elif torch.backends.mps.is_available():
        device = "mps"
        torch_dtype = torch.float16 # MPS optimized
    else:
        device = "cpu"
        torch_dtype = torch.float32

    print(f"[*] Detected Device: {device}")
    
    # 2. Load Tokenizer
    print("[*] Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH, trust_remote_code=True)
    
    # Qwen models usually don't have pad token defined explicitly, so we assign one if missing.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 3. Load Model
    print(f"[*] Loading Model from {MODEL_NAME_OR_PATH}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME_OR_PATH,
        torch_dtype=torch_dtype,
        device_map=device,
        trust_remote_code=True
    )
    
    # Optional: Prepare model for gradient checkpointing and stability
    model.gradient_checkpointing_enable()

    # 4. Setup LoRA
    print("[*] Configuring LoRA...")
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 5. Load Dataset
    print(f"[*] Loading Dataset from {DATASET_PATH}...")
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

    # The dataset is already formatted with "messages" (ChatML structure)
    # We define a formatting function for trl's SFTTrainer to turn it into conversational text
    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example['messages'])):
            messages = example['messages'][i]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            output_texts.append(text)
        return output_texts

    # 6. Training Arguments
    print("[*] Initializing Training Arguments...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        logging_steps=10,
        save_strategy="epoch",
        optim="adamw_torch",
        bf16=(device == "cuda" and torch.cuda.is_bf16_supported()),
        fp16=(device == "mps" or (device == "cuda" and not torch.cuda.is_bf16_supported())),
        report_to="none", # disable wandb etc. unless needed
        max_grad_norm=1.0,
    )

    # 7. SFT Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        max_seq_length=MAX_SEQ_LENGTH,
        tokenizer=tokenizer,
        args=training_args,
        formatting_func=formatting_prompts_func,
    )

    # 8. Start Training
    print("[*] Starting Fine-tuning...")
    trainer.train()

    # 9. Save final model
    print(f"[*] Saving final model to {OUTPUT_DIR}/final_model")
    trainer.model.save_pretrained(f"{OUTPUT_DIR}/final_model")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_model")
    print("[*] Done!")

if __name__ == "__main__":
    main()
