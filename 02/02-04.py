# OpenAI’s Chat Completions API 
# Managing conversation history
# Demonstrating how to maintain state by passing previous history
# By maintaining a 'messages' list and appending the LLM's replies to it after each interaction, we pass the entire conversation history back to the model with every new request, giving it the "memory" it needs.
# User messages are added with the user role (dictionary key) and model responses are added with the assistant role (dictionary key)

import os
from openai import OpenAI
from pydantic import BaseModel


vllm_server_fqdn = os.getenv("VLLM_SERVER_FQDN")
if not vllm_server_fqdn:
    raise ValueError("ERROR: VLLM_SERVER_FQDN environment variable is not set.")
vllm_url = f"http://{vllm_server_fqdn}:8000/v1"
MODEL_NAME = "nvidia/Qwen3.6-35B-A3B-NVFP4"

client = OpenAI(base_url=vllm_url, api_key="EMPTY") 

# Initialize the message history with the system prompt. This indicates the conversational, instruction-tuned LLM, how it should answer
messages = [{"role": "system", "content": "You are a helpful assistant."}]

# Interaction #1
messages.append({"role": "user", "content": "My name is Carlos"})                 # Context provided by the user
response = client.chat.completions.create(model=MODEL_NAME, messages=messages)
clean_response = response.choices[0].message.content.strip()                      # Remove trailing \n in the LLM response
messages.append({"role": "assistant", "content": clean_response})                 # Add the reply to the list. Use the role (dictionary key) "assistant"

print()
print("Interaction #1, context provided")
print(clean_response)
print(f"Tokens: {response.usage.total_tokens} (Total) = {response.usage.prompt_tokens} (Prompt, including 'messages' list) + {response.usage.completion_tokens} (Completion, this reply including reasoning)")
print()


# Interaction #2
messages.append({"role": "user", "content": "What is my name?"})                  # The specific question to be answered
response = client.chat.completions.create(model=MODEL_NAME, messages=messages)
clean_response = response.choices[0].message.content.strip()                      # Remove trailing \n in the LLM response
messages.append({"role": "assistant", "content": clean_response})                 # Add the reply to the list. Use the role (dictionary key) "assistant"

print("Interaction #2, is there context (state) still there? Yes, because I pass all my previous user propmts together every time")
print(clean_response)
print(f"Tokens: {response.usage.total_tokens} (Total) = {response.usage.prompt_tokens} (Prompt, including 'messages' list) + {response.usage.completion_tokens} (Completion, this reply including reasoning)")
print()