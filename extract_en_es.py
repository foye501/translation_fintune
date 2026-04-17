import sys
from datasets import load_dataset
import pandas as pd

def main():
    print("Loading qanastek/ELRC-Medical-V2 (en-es) configuration...")
    # Load dataset with trust_remote_code=True since the builder uses a python script 
    # and we downgraded datasets to a compatible version (2.18.0)
    ds = load_dataset('qanastek/ELRC-Medical-V2', 'en-es', trust_remote_code=True)
    
    # Check splits
    print(f"Dataset splits: {ds.keys()}")
    
    # Save train split as CSV and JSONL
    if 'train' in ds:
        df = pd.DataFrame(iter(ds['train']))
        
        # ELRC-Medical usually has 'translation' dict with 'en' and 'es' keys
        # We should parse it to give direct columns
        # translation: {'en': '...', 'es': '...'}
        
        if 'translation' in df.columns:
            print("Flattening translation column...")
            df['en'] = df['translation'].apply(lambda x: x.get('en', ''))
            df['es'] = df['translation'].apply(lambda x: x.get('es', ''))
            df = df.drop(columns=['translation'])
        
        df.to_csv('en-es_translation_data.csv', index=False)
        df.to_json('en-es_translation_data.jsonl', orient='records', lines=True, force_ascii=False)
        
        print(f"Saved {len(df)} records to en-es_translation_data.csv and en-es_translation_data.jsonl")
    else:
        print("No training data found.")

if __name__ == "__main__":
    main()
