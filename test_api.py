import os
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL", None)

print(f"API Key Starts with: {api_key[:10] if api_key else 'None'}...")
print(f"Base URL: {base_url}")

client = OpenAI(api_key=api_key, base_url=base_url, timeout=10.0) # 10 seconds generic timeout

try:
    print("\nAttempting to call gpt-4o-mini...")
    t0 = time.time()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello, this is a test. Answer 'OK'."}],
        max_tokens=10
    )
    t1 = time.time()
    print(f"Success! Response received in {t1 - t0:.2f} seconds.")
    print("Response Content:", response.choices[0].message.content)
except Exception as e:
    t1 = time.time()
    print(f"\nFAILED after {t1 - t0:.2f} seconds.")
    print(f"Error Type: {e.__class__.__name__}")
    print(f"Error Details: {e}")
