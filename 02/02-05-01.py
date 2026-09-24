# Sends three questions concurrently to an OpenAI-compatible LLM server. It prints each question, its answer and the total execution time
# OpenAI -> Chat Completions
# AsyncOpenAI -> Concurrent Chat Completion
# Asyncio -> Concurrency framework for I/O-bound tasks like DB queries or LLM inference

import os
import time
import asyncio
from openai import AsyncOpenAI
from rich.console import Console

console = Console()

vllm_server_fqdn = os.getenv("VLLM_SERVER_FQDN")
if not vllm_server_fqdn:
    raise ValueError("ERROR: VLLM_SERVER_FQDN environment variable is not set.")
vllm_url = f"http://{vllm_server_fqdn}:8000/v1"

MODEL_TEMPERATURE = 0.0
MODEL_NAME = "nvidia/Qwen3.6-35B-A3B-NVFP4"
SYSTEM_PROMPT = "You are a helpful assistant. Output plain text only. Do not use emojis or emoticons. "

# Initialize the async client
client = AsyncOpenAI(base_url=vllm_url, api_key="EMPTY") 

# Coroutine (function defined with async def) for the three simultaneous questions
async def inference(prompt: str):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    response = await client.chat.completions.create(model=MODEL_NAME, messages=messages, temperature=MODEL_TEMPERATURE)
    clean_response = response.choices[0].message.content.strip()  # Remove trailing \n in the LLM response

    return clean_response


# Wrap the execution block in a main function. This not needed in Jupyter Notebooks but required in .py for asyncio's "await" to work
async def main():

    # Prepare the three questions
    start_time = time.time()
    prompts = ["What is 2 + 2?", "What is the capital of Japan?", "Who wrote Romeo and Juliet?"]
    console.print(f"\nAsking {len(prompts)} questions concurrently", style="gold1", highlight=False)

    # Create a list with the three coroutine calls, each one with a different question as parameter
    tasks = [inference(p) for p in prompts]

    # Execute the three coroutines concurrently and wait until all three are completed
    results = await asyncio.gather(*tasks)

    end_time = time.time()
    execution_time_seconds = (end_time - start_time)

    # Show each question and the result
    for prompt, result in zip(prompts, results):
        console.print(f"Q: {prompt}", style="white", highlight=False)
        print(f"A: {result}\n")
    console.print(f"Time: {execution_time_seconds:.2f} seconds\n", style="cyan", highlight=False)


# Main  
asyncio.run(main())