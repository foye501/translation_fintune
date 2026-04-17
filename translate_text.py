import asyncio
import json
import os
import pandas as pd
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from dotenv import load_dotenv
import openai

# Load environment variables from .env file
load_dotenv()

# Check for API key
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL", None)

if not api_key or api_key == "your_api_key_here":
    print("Error: OPENAI_API_KEY not found or not set correctly in .env")
    exit(1)

client = AsyncOpenAI(api_key=api_key, base_url=base_url)

# Configuration
INPUT_FILE = 'terms_text_filtered_freq_5_train_dedup.csv'
OUTPUT_FILE = 'translated_terms_es.jsonl'
MODEL_NAME = 'gpt-4o'
MAX_CONCURRENT_REQUESTS = 10

# We only retry on RateLimitError, APIConnectionError, InternalServerError
def should_retry(exception):
    if isinstance(exception, (openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError)):
        return True
    return False

# Retry logic for API calls
@retry(
    wait=wait_exponential(multiplier=2, min=2, max=60),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(Exception),
    reraise=True
)
async def fetch_translation(text):
    # Quick fail for non-retryable errors
    try:
        prompt = f"Translate the following medical text from English to Spanish. Only output the translated text, do not add any additional comments.\n\nEnglish:\n{text}\n\nSpanish:"
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are an expert medical translator from English to Spanish."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=1000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        if not should_retry(e):
            print(f"\n[Fatal Error] {e.__class__.__name__}: {e}")
            raise  # Will break the retry loop immediately for non-retryable errors
        raise # Reraise to trigger retry

async def process_row(row, output_file_lock, output_file):
    text_en = row['text']
    try:
        text_es = await fetch_translation(text_en)
        # Construct output dictionary compatible with opensource dataset format
        result = {
            "en": text_en,
            "es": text_es
        }
        
        # Write to JSONL
        async with output_file_lock:
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        return True
    except Exception as e:
        print(f"Failed to translate text: {text_en[:50]}... Error: {e.__class__.__name__}: {e}")
        return False

async def worker(queue, output_file_lock, output_file, pbar, results_list):
    while True:
        try:
            row = queue.get_nowait()
        except asyncio.QueueEmpty:
            break
            
        success = await process_row(row, output_file_lock, output_file)
        results_list.append(success)
        pbar.update(1)
        queue.task_done()

async def main():
    print(f"Loading input file: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    
    # Check already completed to allow resuming
    processed_texts = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        if 'en' in record:
                            processed_texts.add(record['en'])
                    except json.JSONDecodeError:
                        pass
    
    print(f"Found {len(processed_texts)} already processed records.")
    
    # Filter out already processed texts
    tasks_to_run = []
    for _, row in df.iterrows():
        if row['text'] not in processed_texts:
            tasks_to_run.append(row)
    
    if not tasks_to_run:
        print("All records have been translated!")
        return

    print(f"Remaining records to translate: {len(tasks_to_run)}")
    
    # Instead of creating 64,000 tasks at once, we use an asyncio.Queue and fixed workers!
    queue = asyncio.Queue()
    for row in tasks_to_run:
        queue.put_nowait(row)
        
    output_file_lock = asyncio.Lock()
    results_list = []
    pbar = tqdm(total=queue.qsize(), desc="Translating")
    
    print(f"Starting translation using {MODEL_NAME} with concurrency {MAX_CONCURRENT_REQUESTS}...")
    
    # Create a fixed pool of workers equal to MAX_CONCURRENT_REQUESTS
    workers = [
        asyncio.create_task(worker(queue, output_file_lock, OUTPUT_FILE, pbar, results_list))
        for _ in range(MAX_CONCURRENT_REQUESTS)
    ]
    
    # Wait for all workers to finish
    await asyncio.gather(*workers)
    
    pbar.close()
    
    successes = sum(1 for r in results_list if r)
    print(f"\nProcessing complete: {successes} successful, {len(results_list) - successes} failed.")

if __name__ == "__main__":
    asyncio.run(main())
