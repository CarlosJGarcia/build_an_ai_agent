# OpenAI’s Chat Completions API 
# Structured data extraction
# Demonstrating how to 'convince' a LLM to reply using data structures (JSON)
# By swiching from "conversation mode" to "data structures" we bridge the gap between a "conversational brain" and the strict requirements of APIs and tools

import os
from openai import OpenAI
from pydantic import BaseModel

class ExtractedInfo(BaseModel):      # Define data structure named ExtractedInfo using pydantic 
    name: str                        # Required string
    email: str                       # Required string
    phone: str | None = None         # Optonal string (it can be either a string or nothing, by default it's nothing) 

vllm_server_fqdn = os.getenv("VLLM_SERVER_FQDN")
if not vllm_server_fqdn:
    raise ValueError("ERROR: VLLM_SERVER_FQDN environment variable is not set.")
vllm_url = f"http://{vllm_server_fqdn}:8000/v1"
MODEL_NAME = "nvidia/Qwen3.6-35B-A3B-NVFP4"

client = OpenAI(base_url=vllm_url, api_key="EMPTY") 


# Instead of client.chat.completions.create() now we invoke the LLM with client.beta.chat.completions.parse() and parameter response_format
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

# Instead of using message.content, we get the LLM response using message.parsed
clean_response = response.choices[0].message.parsed

print()
print(clean_response)
print(f"Tokens: {response.usage.total_tokens} (Total) = {response.usage.prompt_tokens} (Prompt, including 'messages' list) + {response.usage.completion_tokens} (Completion, this reply including reasoning)")
print()

