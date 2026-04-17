import json
import random
from transformers import AutoTokenizer

def main():
    # 1. Provide the model name we are targeting
    MODEL_NAME_OR_PATH = "Qwen/Qwen2.5-3B-Instruct" 
    DATASET_PATH = "qwen_sft_dataset.jsonl"
    
    print(f"[*] Loading Tokenizer '{MODEL_NAME_OR_PATH}'...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH, trust_remote_code=True)
    
    print(f"[*] Loading dataset from '{DATASET_PATH}'...")
    records = []
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
                
    print(f"[*] Total records loaded: {len(records)}")
    
    # Randomly pick 3 samples to evaluate
    samples = random.sample(records, min(3, len(records)))
    
    print("\n" + "="*80)
    print("DATASET VISUAL VALIDATION (What the model actually sees during training)")
    print("="*80)
    
    for idx, sample in enumerate(samples):
        # Apply the exact chat template used during training
        messages = sample.get("messages", [])
        
        # tokenize=False returns the exact formatted prompt string
        formatted_prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False
        )
        
        print(f"\n--- SAMPLE {idx + 1} ---")
        print(formatted_prompt)
        print("-" * 80)
        
    print("\n[*] Validation Check:")
    print("1. Do you see the special Qwen tokens like '<|im_start|>' and '<|im_end|>'?")
    print("2. Is the medical terminology clearly translated without cutoff?")
    print("3. Does it Alternate logically between 'user' and 'assistant' roles?")
    print("If YES, the dataset is PERFECTLY formatted for SFT!")

if __name__ == "__main__":
    main()
