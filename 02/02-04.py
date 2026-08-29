# OpenAI’s Chat Completions API 
# Structured data extraction
# Demonstrating how to 'convince' a LLM to reply using data structures (JSON)
# By swiching from "conversation mode" to "data structures" we bridge the gap between a "conversational brain" and the strict requirements of APIs and tools
# This code forces the chaotic, natural-language output of an LLM into predictable, strongly-typed code objects before the rest of the software toches it

import os, re, json
from openai import OpenAI
from pydantic import BaseModel

class ExtractedInfo(BaseModel):      # Define data structure named ExtractedInfo using pydantic 
    name: str                        # Required string
    email: str                       # Required string
    phone: str | None = None         # Optonal string (it can be either a string or nothing, by default it's nothing) 

schema_template = {
    "name": "string",
    "email": "string",
    "phone": "string (optional)"
}
schema_string = json.dumps(schema_template)

vllm_server_fqdn = os.getenv("VLLM_SERVER_FQDN")
if not vllm_server_fqdn:
    raise ValueError("ERROR: VLLM_SERVER_FQDN environment variable is not set.")
vllm_url = f"http://{vllm_server_fqdn}:8000/v1"
MODEL_NAME = "nvidia/Qwen3.6-35B-A3B-NVFP4"

client = OpenAI(base_url=vllm_url, api_key="EMPTY") 

response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {"role": "system", "content": f"You are a data extraction assistant. Output ONLY a valid JSON object matching this schema: {schema_string}. Do not include markdown blocks or schema keywords like 'properties' in your final output."},
        {"role": "user", "content": "My name is John Smith, my email is john@example.com, and my phone is 555-1234."}
    ],
    temperature=0.0
)

# Get the raw text string
raw_content = response.choices[0].message.content.strip()

# Sanitizer. Regex that catches any variation of a stuttered opening brace ({{, {"{, etc.) and flattens it.
raw_content = re.sub(r'^\{\s*\"?\{', '{', raw_content)

# Unwrap Safeguard. Parse the raw string into a standard Python dictionary first
try:
    raw_dict = json.loads(raw_content)
except json.JSONDecodeError:
    raise ValueError(f"Model failed to output valid JSON. Raw output: {raw_content}")

# If the model stubbornly wrapped the output in a "properties" key, unwrap it
if "properties" in raw_dict:
    raw_dict = raw_dict["properties"]

# Manually parse the clean JSON string into the Pydantic object
clean_response = ExtractedInfo.model_validate(raw_dict)

print()
print(clean_response)
print(f"Tokens: {response.usage.total_tokens} (Total) = {response.usage.prompt_tokens} (Prompt, including 'messages' list) + {response.usage.completion_tokens} (Completion, this reply including reasoning)")
print()

