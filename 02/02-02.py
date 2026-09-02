# OpenAI’s Chat Completions API 
# Demostrating that LLMs are stateless
# Each call to client.chat.completions.create() API is independent and has no memory of a previous calls

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

# List of dictionaries
messages=[
        {"role": "system", "content": "You are a helpful assistant."},   # How the conversational, instruction-tuned LLM should answer
        {"role": "user", "content": "My name is Carlos"},                # Context provided by the user
        {"role": "user", "content": "What is my name?"}                  # The specific question to be answered
    ]
response = client.chat.completions.create(model=MODEL_NAME, messages=messages)
clean_response = response.choices[0].message.content.strip()  # Remove trailing \n in the LLM response

console.print("\nInteraction #1, context provided", style="gold1", highlight=False)
for item in messages:
    console.print(f"{item}", style="white", highlight=False)
print(clean_response)

# List of dictionaries
messages=[
        {"role": "system", "content": "You are a helpful assistant."},   # How the conversational, instruction-tuned LLM should answer
        {"role": "user", "content": "What is my name?"}                  # The specific question to be answered
    ]
response = client.chat.completions.create(model=MODEL_NAME, messages=messages)
clean_response = response.choices[0].message.content.strip()  # Remove trailing \n in the LLM response

console.print("\nInteraction #2, is there context (state) still there?", style="gold1", highlight=False)
for item in messages:
    console.print(f"{item}", style="white", highlight=False)
print(clean_response)
print()
