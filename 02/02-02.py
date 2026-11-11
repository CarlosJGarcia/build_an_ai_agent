# OpenAI’s Chat Completions API 
# Demostrating that LLMs are stateless
# Each call to client.chat.completions.create() API is independent and has no memory of previous calls

import os
from openai import OpenAI
from rich.console import Console

MODEL_NAME = "nvidia/Qwen3.6-35B-A3B-NVFP4"
SYSTEM_PROMPT = "You are a helpful assistant. Output plain text only. Do not use emojis or emoticons."

# Read FQDN environment variable and build the VLLM URL for the model
vllm_server_fqdn = os.getenv("VLLM_SERVER_FQDN")
if not vllm_server_fqdn:
    raise ValueError("ERROR: VLLM_SERVER_FQDN environment variable is not set.")
vllm_url = f"http://{vllm_server_fqdn}:8000/v1"


console = Console()
client = OpenAI(base_url=vllm_url, api_key="EMPTY") 

# List of dictionaries
messages=[
        {"role": "system", "content": SYSTEM_PROMPT},              # How the conversational, instruction-tuned LLM should answer
        {"role": "user", "content": "My name is Carlos"},          # Context provided by the user
        {"role": "user", "content": "What is my name?"}            # The specific question to be answered
    ]

response = client.chat.completions.create(model=MODEL_NAME, messages=messages)
clean_response = response.choices[0].message.content.strip()  # Remove trailing \n in the LLM response

console.print("\nInteraction #1. System prompt, context and question", style="gold1", highlight=False)
console.print("Question:", style="white", highlight=False)
for item in messages:
    console.print(f"{item}", style="white", highlight=False)
print(f"Inference: {clean_response}")

# List of dictionaries
messages=[
        {"role": "system", "content": SYSTEM_PROMPT},              # How the conversational, instruction-tuned LLM should answer
        {"role": "user", "content": "What is my name?"}            # The specific question to be answered
    ]
response = client.chat.completions.create(model=MODEL_NAME, messages=messages)
clean_response = response.choices[0].message.content.strip()  # Remove trailing \n in the LLM response

console.print("\nInteraction #2. System prompt and question. Is context (state) still there in the Model?", style="gold1", highlight=False)
console.print("Question:", style="white", highlight=False)
for item in messages:
    console.print(f"{item}", style="white", highlight=False)
print(f"Inference: {clean_response}")
print()
