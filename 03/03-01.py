# Calculator tool definition schema

# Python dictionary
# Top level: two keys "type" and "function"

# The value of the second key "function" is another dictionary containing:
# key : value
# "name": the function name ("calculator")
# "description": what it does
# "parameters": a nested dictionary defining the function's arguments (type, properties, required fields)

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

# Print the top level keys
print(calculator_tool_definition.keys())

# Print the values of the top level keys
print(f"key: value")
for key, value in calculator_tool_definition.items():
    print(f"{key}: {value}")

# Python dictionary
# Top level: two keys "type" and "function"