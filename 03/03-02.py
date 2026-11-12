# Calculator tool definition schema
# Bregenz (Austria) 10/Oct/2026
# Reinach (Schweiz) 12/Nov/2026

import os
import json
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

#1 ----------

# List with one dictionary
messages=[{"role": "user", "content": "What is the capital of South Korea?"}]
    
# tools: list with one item, the dictionary "calculator_tool_definition"
tools = [calculator_tool_definition]

question = messages[0]["content"]
console.print(f"\nInference request to the model: '{question}' + calculator tool definition", style="gold1", highlight=False)

# Inference request
#response_without_tool = completion(model=MODEL_NAME, messages=messages, tools=tools)
response_without_tool = client.chat.completions.create(model=MODEL_NAME, messages=messages, tools=tools)

clean_response = response_without_tool.choices[0].message.content.strip()  # Remove trailing \n in the LLM response
print(f"Reply in 'content': {clean_response}")
if clean_response != None:
    print("The model is replying from training data")

clean_response = response_without_tool.choices[0].message.content.strip()  # Remove trailing \n in the LLM response
console.print(clean_response, style="white")
console.print(response_without_tool.choices[0].message.tool_calls, style="white")


# 2 ----------
# List with one dictionary
messages=[{"role": "user", "content": "What is 1234 x 5678?"}]

question = messages[0]["content"]
console.print(f"\nInference request to the model: '{question}' + calculator tool definition", style="gold1", highlight=False)

# Inference request
# response_with_tool = completion(model='gpt-5-mini', messages=[{"role": "user", "content": "What is 1234 x 5678?"}], tools=tools)
response_with_tool = client.chat.completions.create(model=MODEL_NAME, messages=messages, tools=tools)

clean_response = str(response_with_tool.choices[0].message.tool_calls)[:120]  # Trim so it fits in one line
print(f"Reply in 'tool_calls': {clean_response}...")
if clean_response != None:
    print("The model is replying with a tool call")

console.print(response_with_tool.choices[0].message.content, style="white", highlight=False)
console.print(response_with_tool.choices[0].message.tool_calls, style="white", highlight=False)
print()

# Extract and execute the tool from the response
if response_with_tool.choices[0].message.tool_calls:
   for tool_call in response_with_tool.choices[0].message.tool_calls:
       function_name = tool_call.function.name
       function_args = json.loads(tool_call.function.arguments)

       if function_name == "calculator":
           result = calculator(**function_args)

console.print (f"Executing the calculator tool. Result = {result:,}", style="gold1", highlight=False)

# 3 ----------
# Feeding tool results back to the LLM, we use values 'user' (question), 'assistant' (tool call) and 'tool' (return from the tool)
ai_message = response_with_tool.choices[0].message
messages.append({  #1
   "role": "assistant",  #1
   "content": ai_message.content,  #1
   "tool_calls": ai_message.tool_calls  #1
})

if ai_message.tool_calls:
   for tool_call in ai_message.tool_calls:
       function_name = tool_call.function.name  #2
       function_args = json.loads(tool_call.function.arguments)  #2

       if function_name == "calculator":
           result = calculator(**function_args)  #2

           messages.append({  #3
               "role": "tool",  #3
               "tool_call_id": tool_call.id,  #3
               "content": str(result)  #3
           })
question = messages[0]["content"]
tool_answer = messages[2]["content"]
console.print(f"\nInference request to the model: '{question}' + calculator tool definition + '{tool_answer}'", style="gold1", highlight=False)


#final_response = completion(model='gpt-5-mini', messages=messages
final_response = client.chat.completions.create(model=MODEL_NAME, messages=messages) # We don't have to pass tools

clean_response = final_response.choices[0].message.content.strip()  # Remove trailing \n in the LLM response
print(f"Reply in 'content': {clean_response}")
                            
console.print(f"Messages: {messages}", style="white", highlight=False)
clean_response = final_response.choices[0].message.content.strip()  # Remove trailing \n in the LLM response
console.print(f"Final Answer: {clean_response}\n", style="white", highlight=False)