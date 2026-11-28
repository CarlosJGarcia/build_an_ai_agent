# Extending agents with callbacks for user approval

import os
import sys
import inspect
import asyncio
import tempfile
from pathlib import Path
from pydantic import BaseModel
from rich.console import Console
from typing import List, Optional, Type, Callable


from p05_06 import list_files, read_file

# Point Python to the folder one level up, named '04' so from p03_05 import format_tool_definition works
import_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../04'))
if import_dir not in sys.path:
    sys.path.insert(0, import_dir)

from scratch_agents import tool
from scratch_agents import Event
from scratch_agents import Message
from scratch_agents import BaseTool
from scratch_agents import ToolCall
from scratch_agents import LlmClient
from scratch_agents import LlmRequest
from scratch_agents import ToolResult
from scratch_agents import AgentResult
from scratch_agents import LlmResponse
from scratch_agents import ExecutionContext


# Instead of using "from scratch_agents import Agent", I define a new version with callback parameters
class Agent:
    """ReAct agent that reasons and acts using tools."""

    def __init__(
        self,
        model: LlmClient,
        tools: List[BaseTool] = None,
        instructions: str = '',
        max_steps: int = 10,
        name: str = 'agent',
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

    async def run(self, user_input: str, context: ExecutionContext = None) -> AgentResult:
        """Entry point for agent execution."""
        if context is None:
            context = ExecutionContext()

        user_event = Event(
            execution_id=context.execution_id,
            author='user',
            content=[Message(role='user', content=user_input)]
        )
        context.add_event(user_event)

        while not context.final_result and context.current_step < self.max_steps:
            await self.step(context)
            last_event = context.events[-1]
            if self._is_final_response(last_event):
                context.final_result = self._extract_final_result(last_event)

        return AgentResult(output=context.final_result, context=context)

    def _is_final_response(self, event: Event) -> bool:
        """Check if this event contains a final response."""
        if self.output_tool_name:
            for item in event.content:
                if (isinstance(item, ToolResult)
                    and item.name == self.output_tool_name
                    and item.status == 'success'):
                    return True
            return False

        has_tool_calls = any(isinstance(c, ToolCall) for c in event.content)
        has_tool_results = any(isinstance(c, ToolResult) for c in event.content)
        return not has_tool_calls and not has_tool_results

    def _extract_final_result(self, event: Event) -> str | BaseModel:
        if self.output_tool_name:
            for item in event.content:
                if (isinstance(item, ToolResult)
                    and item.name == self.output_tool_name
                    and item.status == 'success'
                    and item.content):
                    return item.content[0]

        for item in event.content:
            if isinstance(item, Message) and item.role == 'assistant':
                return item.content
        return None

    async def step(self, context: ExecutionContext):
        """Perform one complete think-act cycle."""
        llm_request = self._prepare_llm_request(context)
        llm_response = await self.think(llm_request)

        response_event = Event(
            execution_id=context.execution_id,
            author=self.name,
            content=llm_response.content,
        )
        context.add_event(response_event)

        tool_calls = [c for c in llm_response.content if isinstance(c, ToolCall)]
        if tool_calls:
            tool_results = await self.act(context, tool_calls)
            tool_event = Event(
                execution_id=context.execution_id,
                author=self.name,
                content=tool_results,
            )
            context.add_event(tool_event)

        context.increment_step()

    def _prepare_llm_request(self, context: ExecutionContext) -> LlmRequest:
        """Flatten events into content items and package for the LLM."""
        flat_contents = []
        for event in context.events:
            flat_contents.extend(event.content)

        if self.tools:
            tool_choice = 'auto'
        else:
            tool_choice = None

        return LlmRequest(
            instructions=[self.instructions] if self.instructions else [],
            contents=flat_contents,
            tools=self.tools,
            tool_choice=tool_choice,
        )

    async def think(self, llm_request: LlmRequest) -> LlmResponse:
        """Call the LLM to analyze the situation and decide the next action."""
        return await self.model.generate(llm_request)

    async def act(
        self,
        context: ExecutionContext,
        tool_calls: List[ToolCall]
    ) -> List[ToolResult]:
        """Execute the tools that the LLM requested, running before/after callbacks."""
        tools_dict = {tool.name: tool for tool in self.tools}
        results = []

        for tool_call in tool_calls:
            if tool_call.name not in tools_dict:
                results.append(ToolResult(
                    tool_call_id=tool_call.tool_call_id,
                    name=tool_call.name,
                    status='error',
                    content=[f"Tool '{tool_call.name}' not found"],
                ))
                continue

            tool = tools_dict[tool_call.name]
            tool_response = None
            status = 'success'

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
                    status = 'error'

            tool_result = ToolResult(
                tool_call_id=tool_call.tool_call_id,
                name=tool_call.name,
                status=status,
                content=[tool_response],
            )

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


# Async main
async def main():
    console = Console()
    console.print ("\nAgent with callback for user approval", style="gold1", highlight=False) 

    # Listing 5.28 Creating an agent with approval callback
    agent = Agent(
        model=LlmClient(model=MODEL_NAME),
        tools=[list_files, read_file, delete_file],
        instructions="You are a helpful assistant that can explore and manage files.",
        max_steps=20,
        before_tool_callbacks=[approval_callback],
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        (Path(tmp_dir) / "keep.txt").write_text("Keep me")
        (Path(tmp_dir) / "remove.log").write_text("Delete me")

        prompt = (f"List the files in '{tmp_dir}' and then delete the '.log' file you find there.")
        print(f"Prompt: {prompt}\n")

        response = await agent.run(prompt)

        print("\nFinal Response:")
        print(response.output.strip())
        print()


# Main -> Run the async main
if __name__ == "__main__":
    asyncio.run(main())