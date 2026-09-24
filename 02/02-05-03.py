# Sends multiple questions concurrently to an OpenAI-compatible LLM server with a progress bar.
# OpenAI -> Chat Completions
# AsyncOpenAI -> Concurrent Chat Completion
# Asyncio -> Concurrency framework for I/O-bound tasks like DB queries or LLM inference
# Rich Progress -> Real-time progress tracking

import os
import time
import asyncio
from openai import AsyncOpenAI
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

vllm_server_fqdn = os.getenv("VLLM_SERVER_FQDN")
if not vllm_server_fqdn:
    raise ValueError("ERROR: VLLM_SERVER_FQDN environment variable is not set.")
vllm_url = f"http://{vllm_server_fqdn}:8000/v1"

MODEL_TEMPERATURE = 0.0
MODEL_NAME = "nvidia/Qwen3.6-35B-A3B-NVFP4"
SYSTEM_PROMPT = "You are a helpful assistant. Output plain text only. Do not use emojis or emoticons. "

TOTAL_QUESTIONS = 50          # Limit to 50 requests
CONCURRENT_QUESTIONS = 10     # Limit to 10 concurrent requests

# Initialize the semaphore and the async client
semaphore = asyncio.Semaphore(CONCURRENT_QUESTIONS)
client = AsyncOpenAI(base_url=vllm_url, api_key="EMPTY") 

# Coroutine (function defined with async def) for the multiple simultaneous questions
async def inference(prompt: str):
    async with semaphore:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
        response = await client.chat.completions.create(model=MODEL_NAME, messages=messages, temperature=MODEL_TEMPERATURE)  # Automatic retry with exponential backoff
        clean_response = response.choices[0].message.content.strip()  # Remove trailing \n in the LLM response

        return clean_response

# Wrap the execution block in a main function. This not needed in Jupyter Notebooks but required in .py for asyncio's "await" to work
async def main():

    # Create a list with all the questions
    start_time = time.time()
    prompts = [f"What is {i} + {i}?" for i in range(TOTAL_QUESTIONS)]
    console.print(f"\nAsking {TOTAL_QUESTIONS} questions concurrently with a concurrency limit of {CONCURRENT_QUESTIONS} at a time", style="gold1", highlight=False)

    # Execute all coroutines concurrently with a progress bar
    with Progress() as progress:
        progress_task = progress.add_task(f"[cyan]Processing {TOTAL_QUESTIONS} questions...", total=TOTAL_QUESTIONS)

        async def tracked_inference(prompt: str):
            result = await inference(prompt)
            progress.update(progress_task, advance=1)
            return result

        tasks = [tracked_inference(p) for p in prompts]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60

    # Show each question and the result
    for prompt, result in zip(prompts, results):
        console.print(f"Q: {prompt}", style="white", highlight=False)
        print(f"A: {result}\n")
    console.print(f"Time: {execution_time_minutes:.2f} minutes\n", style="cyan", highlight=False)
  

# Main  
asyncio.run(main())