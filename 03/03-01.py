# Calculator tool definition schema
# Bregenz (Austria) 10/Oct/2026

import os
from openai import OpenAI
from rich.console import Console


# Python dictionary with 5 levels of nesting
# Top level: two keys "type" and "function"
# The value of the fist key "type" is "function"
# The value of the second key "function" is another dictionary

# Level 0: type, function — top-level keys
# Level 1: name, description, parameters — inside function
# Level 2: type, properties, required — inside parameters
# Level 3: operator, first_number, second_number — inside properties
# Level 4: type, description, enum — inside operator
# Level 4: type, description — inside "first number"
# Level 4: type, description — inside "second number"

# Calculator tool definition schema
calculator_tool_definition = { 
    "type": "function",
    "function": {
        "name": "calculator", 
        "description": "Perform basic arithmetic operations.",
        "parameters": {
            "type": "object",
            "properties": {
                "operator": {
                    "type": "string",
                    "description": "Arithmetic operation to perform",
                    "enum": ["add", "subtract", "multiply", "divide"]
                },
                "first_number": {
                    "type": "number",
                    "description": "First number for the calculation"
                },
                "second_number": {
                    "type": "number",
                    "description": "Second number for the calculation"
                }
            },
            "required": ["operator", "first_number", "second_number"],
        }
    }
}


# Calculator function
def calculator(operator, first_number, second_number):
    if operator == 'add':
        return first_number + second_number
    elif operator == 'subtract':
        return first_number - second_number
    elif operator == 'multiply':
        return first_number * second_number
    elif operator == 'divide':
        if second_number == 0:
            raise ValueError("Cannot divide by zero")
        return first_number / second_number
    else:
        raise ValueError(f"Unsupported operator: {operator}")


MODEL_NAME = "nvidia/Qwen3.6-35B-A3B-NVFP4"

# Read FQDN environment variable and build the VLLM URL for the model
vllm_server_fqdn = os.getenv("VLLM_SERVER_FQDN")
if not vllm_server_fqdn:
    raise ValueError("ERROR: VLLM_SERVER_FQDN environment variable is not set.")
vllm_url = f"http://{vllm_server_fqdn}:8000/v1"


console = Console()
client = OpenAI(base_url=vllm_url, api_key="EMPTY") 

"""
# Print the top level keys
print(f"\nTop level keys: {calculator_tool_definition.keys()}")

# Print the values of the top level keys
console.print(f"\nkey: value (top level)", style="gold1", highlight=False)
for key, value in calculator_tool_definition.items():
    print(f"{key}: {value}")
"""

# List with one dictionary
messages=[{"role": "user", "content": "What is the capital of South Korea?"}]
    
# tools: lista con un solo item, el diccionario "calculator_tool_definition"
tools = [calculator_tool_definition]


#response_without_tool = completion(model=MODEL_NAME, messages=messages, tools=tools)
response_without_tool = client.chat.completions.create(model=MODEL_NAME, messages=messages, tools=tools)
print(response_without_tool.choices[0].message.content)
print(response_without_tool.choices[0].message.tool_calls)


# response_with_tool = completion(model='gpt-5-mini', messages=[{"role": "user", "content": "What is 1234 x 5678?"}], tools=tools)
messages=[{"role": "user", "content": "What is 1234 x 5678?"}]
response_with_tool = client.chat.completions.create(model=MODEL_NAME, messages=messages, tools=tools)
print(response_with_tool.choices[0].message.content)
print(response_with_tool.choices[0].message.tool_calls)
print()
