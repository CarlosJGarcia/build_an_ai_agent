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

console = Console()

# Print the top level keys
print(f"\nTop level keys: {calculator_tool_definition.keys()}")

# Print the values of the top level keys
console.print(f"\nkey: value (top level)", style="gold1", highlight=False)
for key, value in calculator_tool_definition.items():
    print(f"{key}: {value}")

print()