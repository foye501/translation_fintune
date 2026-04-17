import pandas as pd

def main():
    print("Loading terms_text.csv...")
    df = pd.read_csv('terms_text.csv')
    
    print(f"Original total rows: {len(df)}")
    
    # Filter rows with count == 10 (since the maximum value in the dataset is 10)
    filtered_df = df[df['count'] >= 5]
    
    print(f"Filtered rows (count = 10): {len(filtered_df)}")
    
    # Save the filtered data to a new CSV file
    output_filename = 'terms_text_filtered_freq_10.csv'
    filtered_df.to_csv(output_filename, index=False)
    
    jsonl_filename = 'terms_text_filtered_freq_10.jsonl'
    filtered_df.to_json(jsonl_filename, orient='records', lines=True, force_ascii=False)
    
    print(f"Saved to {output_filename} and {jsonl_filename}.")

if __name__ == "__main__":
    main()
