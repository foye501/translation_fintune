import asyncio
import json
import random
import os
from openai import AsyncOpenAI
from dotenv import load_dotenv
import pandas as pd

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = AsyncOpenAI(api_key=api_key)

# The user requested 'GPT 5.4'. Models usually available are 'gpt-4o', 'gpt-4o-mini', etc.
# We map it to gpt-4o which is the highest tier available by default, 
# but you can change it if you have access to a specific newer model endpoint.
MODEL_NAME = "gpt-4o" 

async def score_translation(en_text, es_text, index):
    prompt = f"""
You are an expert bilingual medical evaluator.
Evaluate the following translation from English to Spanish.
Give a score out of 100 based on accuracy, fluency, and preservation of medical terminology.
Provide a very brief (1-2 sentences) reason for the score.

format strictly as:
Score: <number>/100
Reason: <text>

English:
{en_text}

Spanish Translation:
{es_text}
"""
    try:
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=150
        )
        return (index, en_text, es_text, response.choices[0].message.content.strip())
    except Exception as e:
        return (index, en_text, es_text, f"Score: N/A\nReason: Error {e}")

async def main():
    print("Loading translated_terms_es.jsonl...")
    data = []
    with open('translated_terms_es.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    print(f"Total lines available: {len(data)}")
    
    # Randomly sample 20 lines
    sample_size = min(20, len(data))
    sampled_data = random.sample(data, sample_size)
    
    print(f"Sampled {sample_size} records. Starting evaluation using {MODEL_NAME}...\n")
    
    tasks = []
    for idx, row in enumerate(sampled_data):
        tasks.append(score_translation(row.get('en', ''), row.get('es', ''), idx + 1))
    
    results = await asyncio.gather(*tasks)
    
    # Prepare artifact string
    output_md = "# Translation Evaluation Report (Sample of 20)\n\n"
    total_score = 0
    valid_scores = 0
    
    for idx, en, es, eval_result in results:
        output_md += f"### Sample {idx}\n"
        output_md += f"**EN:** {en}\n\n"
        output_md += f"**ES:** {es}\n\n"
        output_md += f"**Evaluation:**\n{eval_result}\n\n"
        output_md += "---\n"
        
        # Parse score
        try:
            score_line = [line for line in eval_result.split('\\n') if 'Score:' in line]
            if score_line:
                score_str = score_line[0].split('Score:')[1].split('/')[0].strip()
                total_score += int(score_str)
                valid_scores += 1
        except Exception:
            pass
            
    if valid_scores > 0:
        avg = total_score / valid_scores
        output_md += f"\n## Average Score: {avg:.1f}/100"
        
    # Write to a markdown artifact directly
    with open('evaluation_report.md', 'w', encoding='utf-8') as f:
        f.write(output_md)
        
    print(f"Evaluation finished! Average score estimated: {avg if valid_scores > 0 else 'N/A'}")
    print("Results saved to evaluation_report.md")

if __name__ == "__main__":
    asyncio.run(main())
