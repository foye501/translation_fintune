import json
import random

def create_chatml_record_en2es(en_text, es_text):
    return {
        "messages": [
            {
                "role": "system",
                "content": "You are an expert bilingual medical translator."
            },
            {
                "role": "user",
                "content": f"Translate the following medical text from English to Spanish:\n\n{en_text}"
            },
            {
                "role": "assistant",
                "content": es_text
            }
        ]
    }

def create_chatml_record_es2en(en_text, es_text):
    return {
        "messages": [
            {
                "role": "system",
                "content": "You are an expert bilingual medical translator."
            },
            {
                "role": "user",
                "content": f"Translate the following medical text from Spanish to English:\n\n{es_text}"
            },
            {
                "role": "assistant",
                "content": en_text
            }
        ]
    }

def main():
    datasets = [
        "en-es_train.jsonl",
        "translated_terms_es.jsonl"
    ]
    
    combined_records = []
    
    for file_name in datasets:
        print(f"Reading {file_name}...")
        count = 0
        try:
            with open(file_name, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue
                    try:
                        record = json.loads(line)
                        if "en" in record and "es" in record:
                            combined_records.append(
                                create_chatml_record_en2es(record["en"], record["es"])
                            )
                            combined_records.append(
                                create_chatml_record_es2en(record["en"], record["es"])
                            )
                            count += 1
                    except json.JSONDecodeError:
                        pass
            print(f"Loaded {count} records from {file_name}.")
        except Exception as e:
            print(f"Warning: could not process {file_name} - {e}")
            
    # Shuffle the dataset to mix open-source and generated translations
    print("Shuffling combined dataset...")
    random.shuffle(combined_records)
    
    print(f"Total fine-tuning records: {len(combined_records)}")
    
    # Save as JSON configuration commonly used by LLaMA-Factory / Qwen finetuning
    # Usually saved as a JSON array or JSONL. We'll save both for convenience.
    
    output_json = "qwen_sft_dataset.json"
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(combined_records, f, ensure_ascii=False, indent=2)
    print(f"Saved JSON to {output_json}")

    output_jsonl = "qwen_sft_dataset.jsonl"
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for r in combined_records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f"Saved JSONL to {output_jsonl}")

if __name__ == "__main__":
    main()
