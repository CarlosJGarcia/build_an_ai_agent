# OpenAI’s Chat Completions API 
# Managing conversation history
# Demonstrating how to maintain state by passing previous history
# By maintaining a 'messages' list and appending the LLM's replies to it after each interaction, we pass the entire conversation history back to the model with every new request, giving it the "memory" it needs.
# User messages are added with the user role (dictionary key) and model responses are added with the assistant role (dictionary key)

import os
from openai import OpenAI
from pydantic import BaseModel

class ExtractedInfo(BaseModel):      # Define data structure named ExtractedInfo using pydantic model
    name: str                        # Required string
    email: str                       # Required string
    phone: str | None = None         # Optonal string (it can be either a string or nothing, by default it's nothing) 

vllm_server_fqdn = os.getenv("VLLM_SERVER_FQDN")
if not vllm_server_fqdn:
    raise ValueError("ERROR: VLLM_SERVER_FQDN environment variable is not set.")
vllm_url = f"http://{vllm_server_fqdn}:8000/v1"
MODEL_NAME = "nvidia/Qwen3.6-35B-A3B-NVFP4"

client = OpenAI(base_url=vllm_url, api_key="EMPTY") 


# En lugar de client.chat.completions.create() ahora uso client.beta.chat.completions.parse()
# OpenAI's GPT-4 has been heavily fine-tuned to recognize the structured output request and completely suppress their conversational instincts.
# Qwen, open-source running locally often needs an explicit reminder to just output JSON
# I need to add a system message to explicitly instruct the model to act as a strict data parser and suppress all conversation
response = client.beta.chat.completions.parse(
    model=MODEL_NAME,
    messages=[
        {"role": "system", "content": "You are a data extraction assistant. You must output ONLY valid JSON that matches the requested schema. Do not include any greetings, conversational text, or markdown blocks."},
        {"role": "user", "content": "My name is John Smith, my email is john@example.com, and my phone is 555-1234."}],
    response_format=ExtractedInfo
)

# En lugar de message.content, recogemos el resultado de message.parsed
clean_response = response.choices[0].message.parsed

print()
print(clean_response)
print(f"Tokens: {response.usage.total_tokens} (Total) = {response.usage.prompt_tokens} (Prompt, including 'messages' list) + {response.usage.completion_tokens} (Completion, this reply including reasoning)")
print()

