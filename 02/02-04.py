# OpenAI’s Chat Completions API 
# Structured data extraction
# Demonstrating how to 'convince' a LLM to reply using data structures (JSON)
# By swiching from "conversation mode" to "data structures" we bridge the gap between a "conversational brain" and the strict requirements of APIs and tools
# This code forces the chaotic, natural-language output of an LLM into predictable, strongly-typed code objects before the rest of the software toches it

import os, re, json
from openai import OpenAI
from pydantic import BaseModel
from rich.console import Console

# Define data structure (named ExtractedInfo) using pydantic 
class ExtractedInfo(BaseModel):      
    name: str                        # Required string
    email: str                       # Required string
    phone: str | None = None         # Optonal string (it can be either a string or nothing, by default it's nothing) 

# Define the same data, structure in JSON format, to be able to tell the LLM which format I expect to get
schema_template = {
    "name": "string",
    "email": "string",
    "phone": "string (optional)"
}
schema_string = json.dumps(schema_template)
console = Console()
console.print(f"\nJSON schema_string: {schema_string}", style="gold1", highlight=False)

vllm_server_fqdn = os.getenv("VLLM_SERVER_FQDN")
if not vllm_server_fqdn:
    raise ValueError("ERROR: VLLM_SERVER_FQDN environment variable is not set.")
vllm_url = f"http://{vllm_server_fqdn}:8000/v1"
MODEL_NAME = "nvidia/Qwen3.6-35B-A3B-NVFP4"
MODEL_TEMPERATURE = 0.0
SYSTEM_PROMPT = "You are a helpful assistant. Output plain text only. Do not use emojis or emoticons. "
SYSTEM_PROMPT += F"Output ONLY a valid JSON object matching this schema: {schema_string}. "
SYSTEM_PROMPT += "Do not include markdown blocks or schema keywords like 'properties' in your final output."

client = OpenAI(base_url=vllm_url, api_key="EMPTY") 

messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "My name is John Smith, my email is john@example.com, and my phone is 555-1234."}
    ]
console.print("Pregunta:", style="white", highlight=False)
for item in messages:
    console.print(f"{item}", style="white", highlight=False)

response = client.chat.completions.create(model=MODEL_NAME, messages=messages, temperature=MODEL_TEMPERATURE)

# Get the raw text string
clean_response = response.choices[0].message.content.strip()  

# Sanitizer. Regex that catches any variation of a stuttered opening brace ({{, {"{, etc.) and flattens it.
clean_response = re.sub(r'^\{\s*\"?\{', '{', clean_response)

# Unwrap Safeguard. Parse the raw string into a standard Python dictionary first
try:
    dict_response = json.loads(clean_response)
except json.JSONDecodeError:
    raise ValueError(f"Model failed to output valid JSON. Raw output: {clean_response}")

# If the model stubbornly wrapped the output in a "properties" key, unwrap it
if "properties" in dict_response:
    raw_dict = dict_response["properties"]

# Manually parse the clean JSON string into the Pydantic object
final_response = ExtractedInfo.model_validate(dict_response)

print(f"Response: {clean_response}")
print(f"Response, extrated from JSON using Pydantic: {final_response}")
print(f"Tokens: {response.usage.total_tokens} (Total) = {response.usage.prompt_tokens} (Prompt, including 'messages' list) + {response.usage.completion_tokens} (Completion, this reply including reasoning)")
print()

