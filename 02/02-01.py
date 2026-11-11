# OpenAI’s Chat Completions API 
# This API endpoint generates the response of a model from a list of messages comprising a conversation

import os
from openai import OpenAI
from rich.console import Console

MODEL_NAME = "nvidia/Qwen3.6-35B-A3B-NVFP4"

# Read FQDN environment variable and build the VLLM URL for the model
vllm_server_fqdn = os.getenv("VLLM_SERVER_FQDN")
if not vllm_server_fqdn:
    raise ValueError("ERROR: VLLM_SERVER_FQDN environment variable is not set.")
vllm_url = f"http://{vllm_server_fqdn}:8000/v1"

console = Console()
client = OpenAI(base_url=vllm_url, api_key="EMPTY") 

# List of dictionaries. Should be named 'messages' for alignment with the examples in OpenAI's SDK specification 
prompts = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ]

question = prompts[1]["content"]
console.print(f"\nAsking the model '{question}'", style="gold1", highlight=False)

# Inference request
response = client.chat.completions.create(model=MODEL_NAME, messages=prompts)
clean_response = response.choices[0].message.content.strip()  # Remove trailing \n in the LLM response

print(f"Reply: {clean_response}\n")


# for item in prompts:
#     console.print(f"{item}", style="white", highlight=False)


