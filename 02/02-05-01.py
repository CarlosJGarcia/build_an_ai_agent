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

# Limit to 10 concurrent requests
"""
CONCURRENT = 10
semaphore = asyncio.Semaphore(CONCURRENT)
"""

# Initialize the async client (it will automatically look for OPENAI_API_KEY in your environment)
client = AsyncOpenAI(base_url=vllm_url, api_key="EMPTY") 

# Coroutine (function defined with async def) for the three simultaneous questions
async def get_response(prompt: str):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    response = await client.chat.completions.create(model=MODEL_NAME, messages=messages)
    clean_response = response.choices[0].message.content.strip()  # Remove trailing \n in the LLM response

    return clean_response

"""
# Coroutine (function defined with async def) for the ten simulatenous questions
async def call_llm(prompt: str):
    async with semaphore:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
        response = await client.chat.completions.create(model=MODEL_NAME, messages=messages)  # Automatic retry with exponential backoff
        clean_response = response.choices[0].message.content.strip()  # Remove trailing \n in the LLM response

        return clean_response
"""

# Wrap the execution block in a main function. This not needed in Jupyter Notebooks but required in .py for asyncio's "await" to work
async def main():

    # Execute three requests/questions concurrently
    start_time = time.time()
    prompts = ["What is 2 + 2?", "What is the capital of Japan?", "Who wrote Romeo and Juliet?"]
    console.print(f"\nAsking {len(prompts)} questions concurrently", style="gold1", highlight=False)
    tasks = [get_response(p) for p in prompts]

    # Waits for all the asynchronous operations to complete
    results = await asyncio.gather(*tasks)
    end_time = time.time()
    execution_time_seconds = (end_time - start_time)

    # Show each question and the result
    for prompt, result in zip(prompts, results):
        console.print(f"Q: {prompt}", style="white", highlight=False)
        print(f"A: {result}\n")
    console.print(f"Time: {execution_time_seconds:.2f} seconds\n", style="cyan", highlight=False)

    """
    # Now 100 concurrent tasks,with a concurrency limit of at a time
    start_time = time.time()
    prompts = [f"What is {i} + {i}?" for i in range(100)]
    console.print(f"Asking {len(prompts)} concurrent questions with a concurrency limit of {CONCURRENT} at a time", style="gold1", highlight=False)
    tasks = [call_llm(p) for p in prompts]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60

    # Show each question and the result
    for prompt, result in zip(prompts, results):
        console.print(f"Q: {prompt}", style="white", highlight=False)
        print(f"A: {result}\n")
    console.print(f"Time: {execution_time_minutes:.2f} minutes\n", style="cyan", highlight=False)
    """


# Main  
asyncio.run(main())