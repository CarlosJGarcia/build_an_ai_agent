# Extending agents with callbacks for user approval

import os
import sys
import inspect
from pydantic import BaseModel
from typing import List, Optional, Type, Callable

from p05_06 import list_files, read_file

# Point Python to the folder one level up, named '04' so from p03_05 import format_tool_definition works
import_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../04'))
if import_dir not in sys.path:
    sys.path.insert(0, import_dir)

from scratch_agents import tool
from scratch_agents import BaseTool
from scratch_agents import ToolCall
from scratch_agents import LlmClient
from scratch_agents import ToolResult
from scratch_agents import ExecutionContext

# Instead of using "from scratch_agents import Agent", I define a new version with callback parameters
class Agent:
    """ReAct agent that reasons and acts using tools."""

    def __init__(
        self,
        model: LlmClient,
        tools: List[BaseTool] = None,
        instructions: str = "",
        max_steps: int = 10,
        name: str = "agent",
        output_type: Optional[Type[BaseModel]] = None,
        before_tool_callbacks: List[Callable] = None,
        after_tool_callbacks: List[Callable] = None,
    ):
        self.model = model
        self.instructions = instructions
        self.max_steps = max_steps
        self.name = name
        self.output_type = output_type
        self.output_tool_name = None
        self.tools = self._setup_tools(tools or [])

        # Initialize callback lists
        self.before_tool_callbacks = before_tool_callbacks or []
        self.after_tool_callbacks = after_tool_callbacks or []

    def _setup_tools(self, tools: List[BaseTool]):
        """Prepare tools and optionally create a dynamic final_answer tool."""
        if self.output_type is not None:

            # 1. Map the parameters EXACTLY to the Pydantic schema (no nesting)
            raw_schema = self.output_type.model_json_schema()
            raw_schema.pop("title", None)  # vLLM often crashes if it sees 'title'
            raw_schema.pop("description", None)

            schema_def = {
                'type': 'function',
                'function': {
                    'name': 'final_answer',
                    'description': 'Return the final structured answer matching the required schema.',
                    'parameters': raw_schema
                }
            }

            # 2. Use **kwargs so the function accepts the flat dictionary fields
            @tool(
                name='final_answer',
                description='Return the final structured answer matching the required schema.',
                tool_definition=schema_def
            )
            def final_answer(**kwargs):
                # 3. Pass the fields directly into the Pydantic model
                return self.output_type(**kwargs)

            tools = list(tools)
            tools.append(final_answer)
            self.output_tool_name = 'final_answer'

        return tools


async def act(self, context: ExecutionContext, tool_calls: List[ToolCall]):
    tools_dict = {tool.name: tool for tool in self.tools}
    results = []

    for tool_call in tool_calls:
        if tool_call.name not in tools_dict:
            results.append(ToolResult(tool_call_id=tool_call.tool_call_id, name=tool_call.name, status="error", content=[f"Tool '{tool_call.name}' not found"]))
            continue

        tool = tools_dict[tool_call.name]
        tool_response = None
        status = "success"

        # Stage 1: Execute before_tool_callbacks
        for callback in self.before_tool_callbacks:
            result = callback(context, tool_call)
            if inspect.isawaitable(result):
                result = await result
            if result is not None:
                tool_response = result
                break

        # Stage 2: Execute actual tool only if callback didn't provide a result
        if tool_response is None:
            try:
                tool_response = await tool(context, **tool_call.arguments)
            except Exception as e:
                tool_response = str(e)
                status = "error"

        tool_result = ToolResult(tool_call_id=tool_call.tool_call_id, name=tool_call.name, status=status, content=[tool_response])

        # Stage 3: Execute after_tool_callbacks
        for callback in self.after_tool_callbacks:
            result = callback(context, tool_result)
            if inspect.isawaitable(result):
                result = await result
            if result is not None:
                tool_result = result
                break

        results.append(tool_result)

    return results


# Listing 5.26 Implementing the delete_file tool
@tool
def delete_file(file_path: str) -> str:
    """Deletes a file. This action cannot be undone."""
    # Only returns message instead of actual deletion (for demo)
    return f"File {file_path} has been deleted."


# Listing 5.27 Implementing approval callback
def approval_callback(context: ExecutionContext, tool_call: ToolCall):
    """Requests user approval before executing dangerous tools."""
    # Execute immediately if not a dangerous tool
    if tool_call.name not in DANGEROUS_TOOLS:
        return None

    print(f"\n Dangerous tool execution requested")
    print(f"Tool: {tool_call.name}")
    print(f"Arguments: {tool_call.arguments}")

    response = input("Do you want to execute? (y/n): ").lower().strip()

    if response == 'y':
        print(" Approved. Executing...\n")
        return None  # Proceed with actual tool execution
    else:
        print("Denied. Skipping execution.\n")
        return f"User denied execution of {tool_call.name}"



MODEL_NAME = "nvidia/Qwen3.6-35B-A3B-NVFP4"
DANGEROUS_TOOLS = ["delete_file", "send_email", "execute_sql"]       # List of 'dangerous' tools requiring approval


# Listing 5.28 Creating an agent with approval callback
agent = Agent(
    model=LlmClient(model=MODEL_NAME),
    tools=[list_files, read_file, delete_file],
    instructions="You are a helpful assistant that can explore and manage files.",
    max_steps=20,
    before_tool_callbacks=[approval_callback],
)


print ("\nTodo bien\n")