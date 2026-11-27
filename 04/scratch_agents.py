# Code to build a ReAct (Reason + Act) agent
# It forces the LLM into a continuous loop
# This requires bundling a massive amount of logic together at once

import os
import sys
import json
import uuid
import asyncio
import inspect
from datetime import datetime
from openai import AsyncOpenAI
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from dataclasses import dataclass, field
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters
from typing import Any, Callable, Dict, List, Literal, Optional, Type, Union

# Point Python to the folder one level up, named '03' so from p03_05 import format_tool_definition works
import_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../03'))
if import_dir not in sys.path:
    sys.path.insert(0, import_dir)
from p03_05 import format_tool_definition, function_to_input_schema


# ---------------------------------------------------------------------------
# 4.2 Content types for agent communication
# ---------------------------------------------------------------------------

class Message(BaseModel):
    type: Literal["message"] = "message"
    role: Literal["system", "user", "assistant"]
    content: str

class ToolCall(BaseModel):
    type: Literal["tool_call"] = "tool_call"
    tool_call_id: str
    name: str
    arguments: dict

class ToolResult(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    tool_call_id: str
    name: str
    status: Literal["success", "error"]
    content: list

ContentItem = Union[Message, ToolCall, ToolResult]


# ---------------------------------------------------------------------------
# 4.3 Event class for execution tracking
# ---------------------------------------------------------------------------

class Event(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    execution_id: str
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())
    author: str
    content: List[ContentItem] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 4.4 Implementing ExecutionContext
# ---------------------------------------------------------------------------

@dataclass
class ExecutionContext:
    execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    events: List[Event] = field(default_factory=list)
    current_step: int = 0
    state: Dict[str, Any] = field(default_factory=dict)
    final_result: Optional[str | BaseModel] = None

    def add_event(self, event: Event):
        self.events.append(event)

    def increment_step(self):
        self.current_step += 1


# ---------------------------------------------------------------------------
# 4.5 BaseTool abstract class
# ---------------------------------------------------------------------------

class BaseTool(ABC):

    def __init__(
        self, 
        name: str = None, 
        description: str = None, 
        tool_definition: Dict[str, Any] = None,
    ):
        self.name = name or self.__class__.__name__
        self.description = description or self.__doc__ or ''
        self._tool_definition = tool_definition

    @property
    def tool_definition(self) -> Dict[str, Any] | None:
        return self._tool_definition

    @abstractmethod
    async def execute(self, context: ExecutionContext, **kwargs) -> Any:
        pass

    async def __call__(self, context: ExecutionContext, **kwargs) -> Any:
        return await self.execute(context, **kwargs)


# ---------------------------------------------------------------------------
# 4.6 FunctionTool implementation
# ---------------------------------------------------------------------------

class FunctionTool(BaseTool):

    def __init__(self, func: Callable, name: str = None, description: str = None, tool_definition: Dict[str, Any] = None):
        self.func = func
        self.needs_context = 'context' in inspect.signature(func).parameters

        name = name or func.__name__
        description = description or (func.__doc__ or '').strip()
        
        # 1. Initialize the BaseTool FIRST so self.name and self.description are created
        super().__init__(name=name, description=description, tool_definition=tool_definition)

        # 2. THEN generate the definition if it wasn't provided
        if not self._tool_definition:
            self._tool_definition = self._generate_definition()

        # super().__init__(name=name, description=description, tool_definition=tool_definition)


    async def execute(self, context: ExecutionContext, **kwargs) -> Any:
        if self.needs_context:
            result = self.func(context=context, **kwargs)
        else:
            result = self.func(**kwargs)

        if inspect.iscoroutine(result):
            return await result
        return result

    def _generate_definition(self) -> Dict[str, Any]:
        parameters = function_to_input_schema(self.func)
        return format_tool_definition(self.name, self.description, parameters)


# ---------------------------------------------------------------------------
# 4.6 The @tool decorator
# ---------------------------------------------------------------------------

def tool(
    func: Callable = None,
    name: str = None,
    description: str = None,
    tool_definition: Dict[str, Any] = None,
):
    """Decorator to create a FunctionTool from a function."""
    def decorator(f):
        return FunctionTool(func=f, name=name, description=description, tool_definition=tool_definition)

    if func is not None:
        return decorator(func)
    return decorator


# ---------------------------------------------------------------------------
# 4.7 Loading tools from MCP servers
# ---------------------------------------------------------------------------

async def load_mcp_tools(connection: dict) -> list[BaseTool]:
    """Load tools from an MCP server and convert to FunctionTools."""
    tools = []

    async with stdio_client(StdioServerParameters(**connection)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            mcp_tools = await session.list_tools()

            for mcp_tool in mcp_tools.tools:
                func_tool = _create_mcp_tool(mcp_tool, connection)
                tools.append(func_tool)

    return tools


# ---------------------------------------------------------------------------
# 4.8 Creating FunctionTool from the MCP tool
# ---------------------------------------------------------------------------

def _create_mcp_tool(mcp_tool, connection: dict) -> FunctionTool:
    """Create a FunctionTool that wraps an MCP tool."""

    async def call_mcp(**kwargs):
        async with stdio_client(StdioServerParameters(**connection)) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(mcp_tool.name, kwargs)
                return _extract_text_content(result)

    tool_definition = {
        'type': 'function',
        'function': {
            'name': mcp_tool.name,
            'description': mcp_tool.description,
            'parameters': mcp_tool.input_schema,        # Changed from mcp_tool.inputSchema in the book
        }
    }

    return FunctionTool(
        func=call_mcp,
        name=mcp_tool.name,
        description=mcp_tool.description,
        tool_definition=tool_definition
    )


def _extract_text_content(result) -> str:
    """Extract the text content from an MCP tool result."""
    for content in result.content:
        if hasattr(content, 'type') and content.type == 'text':
            return content.text
        if hasattr(content, 'text'):
            return content.text
        if isinstance(content, str):
            return content
    return ''


# ---------------------------------------------------------------------------
# 4.9 LlmRequest class
# ---------------------------------------------------------------------------

class LlmRequest(BaseModel):
    """Request object for LLM calls."""
    model_config = {"arbitrary_types_allowed": True}

    instructions: List[str] = Field(default_factory=list)
    contents: List[ContentItem] = Field(default_factory=list)
    tools: List[BaseTool] = Field(default_factory=list)
    tool_choice: Optional[Union[str, dict]] = None


# ---------------------------------------------------------------------------
# 4.10 LlmResponse class
# ---------------------------------------------------------------------------

class LlmResponse(BaseModel):
    """Response object from LLM calls."""
    content: List[ContentItem] = Field(default_factory=list)
    error_message: Optional[str] = None
    usage_metadata: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# 4.11 LlmClient class
# ---------------------------------------------------------------------------

class LlmClient:
    """Client for LLM API calls using OpenAI."""

    def __init__(self, model: str, **config):
        self.model = model
        self.config = config
        self._client: AsyncOpenAI | None = None

    @property
    def _openai(self) -> AsyncOpenAI:
        if self._client is None:
            vllm_server_fqdn = os.getenv("VLLM_SERVER_FQDN")
            if not vllm_server_fqdn:
                raise ValueError("VLLM_SERVER_FQDN environment variable is not set.")
            vllm_url = f"http://{vllm_server_fqdn}:8000/v1"
            self._client = AsyncOpenAI(base_url=vllm_url, api_key="EMPTY")
        return self._client

    async def generate(self, request: LlmRequest) -> LlmResponse:
        """Generate a response from the LLM."""
        try:
            messages = self._build_messages(request)
            tools = [t.tool_definition for t in request.tools] if request.tools else None

            response = await self._openai.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                **(
                    {'tool_choice': request.tool_choice}
                    if request.tool_choice
                    else {}
                ),
                **self.config
            )

            return self._parse_response(response)
        except Exception as e:
            print(f"\n[DEBUG] API Exception: {e}\n")
            return LlmResponse(error_message=str(e))


    # ------------------------------------------------------------------
    # 4.12 Building messages for the LLM API
    # ------------------------------------------------------------------
    def _build_messages(self, request: LlmRequest) -> List[dict]:
        """Convert LlmRequest to API message format."""
        messages = []

        for instruction in request.instructions:
            messages.append({'role': 'system', 'content': instruction})

        for item in request.contents:
            if isinstance(item, Message):
                messages.append({'role': item.role, 'content': item.content})

            elif isinstance(item, ToolCall):
                tool_call_dict = {
                    'id': item.tool_call_id,
                    'type': 'function',
                    'function': {
                        'name': item.name,
                        'arguments': json.dumps(item.arguments)
                    }
                }
                if messages and messages[-1]['role'] == 'assistant':
                    messages[-1].setdefault('tool_calls', []).append(tool_call_dict)
                else:
                    messages.append({
                        'role': 'assistant',
                        'content': None,
                        'tool_calls': [tool_call_dict]
                    })

            elif isinstance(item, ToolResult):
                messages.append({
                    'role': 'tool',
                    'tool_call_id': item.tool_call_id,
                    'content': str(item.content[0]) if item.content else ''
                })

        return messages

    # ------------------------------------------------------------------
    # 4.13 Parsing the LLM API's response
    # ------------------------------------------------------------------
    def _parse_response(self, response) -> LlmResponse:
        """Convert API response to LlmResponse."""
        choice = response.choices[0]
        content_items = []

        if choice.message.content:
            content_items.append(Message(
                role='assistant',
                content=choice.message.content
            ))

        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                content_items.append(ToolCall(
                    tool_call_id=tc.id,
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments)
                ))

        return LlmResponse(
            content=content_items,
            usage_metadata={
                'input_tokens': response.usage.prompt_tokens,
                'output_tokens': response.usage.completion_tokens,
            }
        )


# ---------------------------------------------------------------------------
# 4.17 AgentResult data class
# ---------------------------------------------------------------------------

@dataclass
class AgentResult:
    """Result of an agent execution."""
    output: str | BaseModel
    context: ExecutionContext


# -----------------------------------------------------------------------------------------------------------
# 4.6 Agent class - the ReAct agent loop
#  ReAct (Reason + Act) — The core loop: the agent thinks → calls a tool → observes the result → thinks again
# -----------------------------------------------------------------------------------------------------------

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
    ):
        self.model = model
        self.instructions = instructions
        self.max_steps = max_steps
        self.name = name
        self.output_type = output_type
        self.output_tool_name = None
        self.tools = self._setup_tools(tools or [])


    def _setup_tools(self, tools: List[BaseTool]):
        """Prepare tools and optionally create a dynamic final_answer tool."""
        if self.output_type is not None:
            
            # 1. Map the parameters EXACTLY to the Pydantic schema (no nesting)
            raw_schema = self.output_type.model_json_schema()
            raw_schema.pop("title", None) # vLLM often crashes if it sees 'title'
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
        """Execute the tools that the LLM requested."""
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

            try:
                output = await tool(context, **tool_call.arguments)
                results.append(ToolResult(
                    tool_call_id=tool_call.tool_call_id,
                    name=tool_call.name,
                    status='success',
                    content=[output],
                ))
            except Exception as e:
                results.append(ToolResult(
                    tool_call_id=tool_call.tool_call_id,
                    name=tool_call.name,
                    status='error',
                    content=[str(e)],
                ))

        return results



# ---------------------------------------------------------------------------
# 4.8 GAIA benchmark evaluation helpers
# ---------------------------------------------------------------------------

class GaiaOutput(BaseModel):
    """Structured output for GAIA benchmark problems."""
    is_solvable: bool
    unsolvable_reason: Optional[str] = None
    final_answer: Optional[str] = None


gaia_prompt = """You are a problem-solving agent. Analyze the following GAIA benchmark problem.

- If the problem can be solved with the tools available to you, use them to find the answer.
- If it cannot be solved (e.g., missing information), explain why.

Return your result as a structured GaiaOutput with:
  - is_solvable: True if you can solve it, False otherwise
  - unsolvable_reason: Only if is_solvable is False
  - final_answer: Your answer if solvable

"""

# ---------------------------------------------------------------------------
# Listing 4.32 Solving problems with an agent
# ---------------------------------------------------------------------------

SEMAPHORE = asyncio.Semaphore(3)

async def solve_problem(agent: Agent, question: str) -> AgentResult:
    """Solve a single GAIA problem with concurrency control."""
    async with SEMAPHORE:
        return await agent.run(question)


async def evaluate_single(
    problem: dict,
    model: str,
    tools: list[BaseTool]
) -> dict:
    """Evaluate a single GAIA problem and return result dict."""
    agent = Agent(
        model=LlmClient(model=model),
        tools=tools,
        instructions=gaia_prompt,
        output_type=GaiaOutput,
        max_steps=15,
    )
    result = await agent.run(problem['question'])
    answer = result.output.final_answer if result.output else None
    return {
        'question_id': problem.get('id', 'unknown'),
        'question': problem['question'][:100],
        'answer': answer,
        'is_solvable': result.output.is_solvable if isinstance(result.output, GaiaOutput) else None,
        'steps': result.context.current_step,
    }


async def run_experiment(
    problems: list[dict],
    models: list[str],
    tools: list[BaseTool] = None,
) -> list[dict]:
    """Run the full GAIA experiment across multiple models."""
    all_results = []
    for model in models:
        for problem in problems:
            r = await evaluate_single(problem, model, tools or [])
            r['model'] = model
            all_results.append(r)
    return all_results


# Listing 4.31 Creating the GAIA evaluation agent
def create_gaia_agent(model: str, tools: list) -> Agent:
    """Create an agent configured for GAIA evaluation."""
    return Agent(
        model=LlmClient(model=model),
        tools=tools,
        instructions=gaia_prompt,
        output_type=GaiaOutput,
        max_steps=15,
    )
