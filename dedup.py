import pandas as pd

def main():
    input_file = 'terms_text_filtered_freq_5_train.csv'
    output_file = 'terms_text_filtered_freq_5_train_dedup.csv'
    
    print(f"Loading {input_file}...")
    df = pd.read_csv(input_file)
    original_len = len(df)
    
    print("Deduplicating based on the 'text' column...")
    df_dedup = df.drop_duplicates(subset=['text'])
    dedup_len = len(df_dedup)
    
    print(f"Original rows: {original_len}")
    print(f"Rows after deduplication: {dedup_len}")
    print(f"Removed {original_len - dedup_len} duplicate texts.")
    
    df_dedup.to_csv(output_file, index=False)
    print(f"Saved deduplicated data to {output_file}.")

if __name__ == "__main__":
    main()
