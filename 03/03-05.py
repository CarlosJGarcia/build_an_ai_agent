# Extracting tool metadata using inspect

import inspect
from rich.console import Console

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
        signature = inspect.signature(func)
    except ValueError as e:
        raise ValueError(f"Failed to get signature for function {func.__name__}: {e}") from e

    parameters = {}
    for param in signature.parameters.values():
        param_type = type_map.get(param.annotation, "string")
        parameters[param.name] = {"type": param_type}

    required = [p.name for p in signature.parameters.values() if p.default is inspect.Parameter.empty]

    return {"type": "object", "properties": parameters, "required": required}


console = Console()
console.print(f"\ndef example_tool(input_1, input_2=1):", style="gold1", highlight=False)
print(f"function name: {example_tool.__name__}")
print(f"function docstring: {example_tool.__doc__}")
print(f"function signature: {inspect.signature(example_tool)}\n")