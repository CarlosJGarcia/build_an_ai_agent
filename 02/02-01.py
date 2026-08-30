# OpenAI’s Chat Completions API 

import os
from openai import OpenAI
from rich.console import Console

console = Console()
vllm_server_fqdn = os.getenv("VLLM_SERVER_FQDN")
if not vllm_server_fqdn:
    raise ValueError("ERROR: VLLM_SERVER_FQDN environment variable is not set.")
vllm_url = f"http://{vllm_server_fqdn}:8000/v1"
MODEL_NAME = "nvidia/Qwen3.6-35B-A3B-NVFP4"

client = OpenAI(base_url=vllm_url, api_key="EMPTY") 

prompts = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ]
response = client.chat.completions.create(model=MODEL_NAME, messages=prompts)
clean_response = response.choices[0].message.content.strip()  # Remove trailing \n in the LLM response

print()
for item in prompts:
    console.print(f"{item}", style="white", highlight=False)
print(clean_response)
print()
