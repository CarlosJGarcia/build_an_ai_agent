# Calculator tool definition schema
# Bregenz (Austria) 10/Oct/2026

from rich.console import Console

# Python dictionary
# Top level: two keys "type" and "function"
# The value of the second key "function" is another dictionary containing:

# Representation of the structure with indentation showing nesting levels:
# type: function
# function: name: calculator
#           description: Perform basic arithmetic operations.
#           parameters: type: object
#                       properties: operator: type: string
#                                       description: Arithmetic operation to perform
#                                       enum: add, subtract, multiply, divide
#                               first_number: type: number
#                                             description: First number for the calculation
#                               second_number: type: number
#                                              description: Second number for the calculation
#                       required: operator, first_number, second_number

# Level 0: type, function — top-level keys
# Level 1: name, description, parameters — inside function
# Level 2: type, properties, required — inside parameters
# Level 3: operator, first_number, second_number — inside properties
# Level 4: type, description, enum — inside each property (e.g., operator)

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

console = Console()

# Print the top level keys
print(f"\nTop level keys: {calculator_tool_definition.keys()}")

# Print the values of the top level keys
console.print(f"\nkey: value (top level)", style="gold1", highlight=False)
for key, value in calculator_tool_definition.items():
    print(f"{key}: {value}")

# Tool calling calculator
tools = [calculator_tool_definition]

response_without_tool = completion(model='gpt-5-mini', messages=[{"role": "user", "content": "What is the capital of South Korea?"}], tools=tools)
print(response_without_tool.choices[0].message.content)
print(response_without_tool.choices[0].message.tool_calls)

response_with_tool = completion(model='gpt-5-mini', messages=[{"role": "user", "content": "What is 1234 x 5678?"}], tools=tools)
print(response_with_tool.choices[0].message.content)
print(response_with_tool.choices[0].message.tool_calls)

print()