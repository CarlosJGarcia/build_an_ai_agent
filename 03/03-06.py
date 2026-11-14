# The tool execution system
# Reinach 14/Nov/2026

import json
import inspect
from collections.abc import Callable

from p03_04 import search_web
from p03_05 import function_to_tool_definition


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

# Agent loop for tool execution
def simple_agent_loop(system_prompt: str, question: str, max_iterations: int = 10) -> str:
    tools = [search_web]
    tool_box = {tool.__name__: tool for tool in tools}
    tool_definitions = [function_to_tool_definition(tool) for tool in tools]

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    for _ in range(max_iterations):
        response = completion(model="gpt-5-mini", messages=messages, tools=tool_definitions)

        assistant_message = response.choices[0].message

        if assistant_message.tool_calls:
            messages.append(assistant_message.model_dump(exclude_none=True))
            for tool_call in assistant_message.tool_calls:
                tool_result = tool_execution(tool_box, tool_call)
                messages.append({
                    "role": "tool",
                    "content": str(tool_result),
                    "tool_call_id": tool_call.id,
                })
        else:
            return assistant_message.content or ""

    raise RuntimeError(f"Agent exceeded max_iterations ({max_iterations}) without finishing.")


print("\nTodo bien\n")