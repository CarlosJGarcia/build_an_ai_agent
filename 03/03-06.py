# The tool execution system
# Reinach 14/Nov/2026

import json

# Takes a toolbox (a dictionary that maps tool names to functions)
# Takes a tool call from the LLM response
# Extracts the tool name and arguments from the LLM response
# Looks up the corresponding function in the toolbox
# Executes the tool function with the provided arguments (from the LLM response)

# Utility function that executes a tool and returns its result
def tool_execution(tool_box, tool_call):
    function_name = tool_call.function.name
    function_args = json.loads(tool_call.function.arguments)

    tool_result = tool_box[function_name](**function_args)
    return tool_result

print("Todo bien")
