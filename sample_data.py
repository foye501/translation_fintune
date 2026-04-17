import pandas as pd

def main():
    print("Loading data...")
    df = pd.read_csv('en-es_translation_data.csv')
    
    print(f"Total rows: {len(df)}")
    
    # Randomly sample 2000 rows for testing
    # random_state ensures reproducibility
    test_df = df.sample(n=2000, random_state=42)
    
    # The rest will be train data
    train_df = df.drop(test_df.index)
    
    print(f"Sampled {len(test_df)} for test set")
    print(f"Remaining {len(train_df)} for train set")
    
    # Save test set
    test_df.to_csv('en-es_test.csv', index=False)
    test_df.to_json('en-es_test.jsonl', orient='records', lines=True, force_ascii=False)
    
    # Save train set
    train_df.to_csv('en-es_train.csv', index=False)
    train_df.to_json('en-es_train.jsonl', orient='records', lines=True, force_ascii=False)
    
    print("Saved en-es_test.csv, en-es_test.jsonl, en-es_train.csv, and en-es_train.jsonl.")

if __name__ == "__main__":
    main()
