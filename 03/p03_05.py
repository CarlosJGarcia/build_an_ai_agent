# Extracting tool metadata using inspect
# Utility function to convert Python function definitions into OpenAI-style function tool definitions

import inspect
from rich.console import Console
from collections.abc import Callable

def example_tool(input_1, input_2=1):
    """docstring for example_tool"""
    return

# Generate an input schema from Python functions
def function_to_input_schema(func):
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
    }

    try:
        signature = inspect.signature(func)                                                            # Extract the function signature using the inspect module
    except ValueError as e:
        raise ValueError(f"Failed to get signature for function {func.__name__}: {e}") from e

    parameters = {}
    for param in signature.parameters.values():
        param_type = type_map.get(param.annotation, "string")                                          # Fallback to string if the type is not recognized
        parameters[param.name] = {"type": param_type}                                                  # Map Python types to JSON Schema types

    required = [p.name for p in signature.parameters.values() if p.default is inspect.Parameter.empty] # Parameters without default values are marked as required

    return {"type": "object", "properties": parameters, "required": required}


# Build an OpenAI-style function tool definition from schema (name, description and parameter)
def format_tool_definition(name: str, description: str, parameters: dict):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


# Returns a schema with the function name and description, parameter types and whether each parameter is required
def function_to_tool_definition(func: Callable):
    return format_tool_definition(
        func.__name__,
        inspect.cleandoc(func.__doc__ or ""),
        function_to_input_schema(func),
    )


console = Console()
console.print(f"\ndef example_tool(input_1, input_2=1):", style="gold1", highlight=False)
console.print(f"inspect:", style="gold1")
print(f"function name: {example_tool.__name__}")
print(f"function docstring: {example_tool.__doc__}")
print(f"function signature: {inspect.signature(example_tool)}")

console.print(f"\nfunction_to_tool:", style="gold1", highlight=False)
print(function_to_tool_definition(example_tool))
print()
      