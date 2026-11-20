# 4.14 Using the LLM communication layer

import os
import asyncio
from openai import AsyncOpenAI
from rich.console import Console
from scratch_agents import LlmClient, LlmRequest, Message

MODEL_NAME = "nvidia/Qwen3.6-35B-A3B-NVFP4"
SYSTEM_PROMPT = "You are a helpful assistant. Output plain text only. Do not use emojis or emoticons."

# Async main
async def main():

    # Read FQDN environment variable and build the VLLM URL for the model
    vllm_server_fqdn = os.getenv("VLLM_SERVER_FQDN")
    if not vllm_server_fqdn:
        raise ValueError("ERROR: VLLM_SERVER_FQDN environment variable is not set.")
    vllm_url = f"http://{vllm_server_fqdn}:8000/v1"

    console = Console()
    client = AsyncOpenAI(base_url=vllm_url, api_key="EMPTY")

    # List of dictionaries
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},              # How the conversational, instruction-tuned LLM should answer
        {"role": "user", "content": "What is 2 + 2?"}              # The specific question to be answered
    ]

    response = await client.chat.completions.create(model=MODEL_NAME, messages=messages)
    
    # Response contains the answer
    print(response.choices[0].message.content)                     # Response should be "4" (2+2)


# Main -> Run the async main
if __name__ == "__main__":
    asyncio.run(main())