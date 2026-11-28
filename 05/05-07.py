# Extending agents with callbacks for user approval

import os
import sys
import inspect
from pydantic import BaseModel
from typing import List, Optional, Type, Callable



# Point Python to the folder one level up, named '04' so from p03_05 import format_tool_definition works
import_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../04'))
if import_dir not in sys.path:
    sys.path.insert(0, import_dir)

#from scratch_agents import tool
from scratch_agents import BaseTool
from scratch_agents import ToolCall
from scratch_agents import LlmClient
from scratch_agents import ToolResult
from scratch_agents import ExecutionContext

# Instead of using "from scratch_agents import Agent", I define a new version with callback parameters
class Agent:
    def __init__(
        self,
        model: LlmClient,
        tools: List[BaseTool] = None,
        instructions: str = "",
        max_steps: int = 10,
        output_type: Optional[Type[BaseModel]] = None,
        before_tool_callbacks: List[Callable] = None,
        after_tool_callbacks: List[Callable] = None,
    ):
        self.model = model
        self.instructions = instructions
        self.max_steps = max_steps
        self.output_type = output_type
        self.output_tool_name = None
        self.tools = self._setup_tools(tools or [])

        # Initialize callback lists
        self.before_tool_callbacks = before_tool_callbacks or []
        self.after_tool_callbacks = after_tool_callbacks or []


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

print ("\nTodo bien\n")