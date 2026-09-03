# OpenAI’s Chat Completions API 
# Asynchronous LLM calls

import asyncio
#from litellm import acompletion
#from openai import OpenAI
from openai import AsyncOpenAI
from rich.console import Console

console = Console()


vllm_server_fqdn = os.getenv("VLLM_SERVER_FQDN")
if not vllm_server_fqdn:
    raise ValueError("ERROR: VLLM_SERVER_FQDN environment variable is not set.")
vllm_url = f"http://{vllm_server_fqdn}:8000/v1"
MODEL_NAME = "nvidia/Qwen3.6-35B-A3B-NVFP4"
MODEL_TEMPERATURE = 0.0
SYSTEM_PROMPT = "You are a helpful assistant. Output plain text only. Do not use emojis or emoticons. "

# Initialize the async client (it will automatically look for OPENAI_API_KEY in your environment)
client = AsyncOpenAI(base_url=vllm_url, api_key="EMPTY") 



"""
async def get_response(prompt: str) -> str:
    response = await acompletion(model="gpt-5-mini", messages=[{"role": "user", "content": prompt}])
    return response.choices[0].message.content
"""
async def get_response(prompt: str) -> str:

    messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
    response = await client.chat.completions.create(model=MODEL_NAME, messages=messages)
    return response.choices[0].message.content


# Wrap the execution block in a main function
async def main():

    prompts = [
        "What is 2 + 2?",
        "What is the capital of Japan?",
        "Who wrote Romeo and Juliet?"
    ]

    

    """
    # Execute all requests concurrently
    tasks = [get_response(p) for p in prompts]
    results = await asyncio.gather(*tasks)

    for prompt, result in zip(prompts, results):
        print(f"Q: {prompt}")
        print(f"A: {result}\n")
    """
    # Execute all requests concurrently
    tasks = [get_response(p) for p in prompts]
    results = await asyncio.gather(*tasks)

    for prompt, result in zip(prompts, results):
        print(f"Q: {prompt}")
        print(f"A: {result}\n")


# Run the main function using asyncio.run
if __name__ == "__main__":
    asyncio.run(main())