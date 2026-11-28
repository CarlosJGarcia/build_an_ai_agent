# Extending agents with callbacks for user approval

import os
import sys
from pydantic import BaseModel
from typing import List, Optional, Type, Callable

# Point Python to the folder one level up, named '04' so from p03_05 import format_tool_definition works
import_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../04'))
if import_dir not in sys.path:
    sys.path.insert(0, import_dir)

#from scratch_agents import tool
from scratch_agents import BaseTool
from scratch_agents import LlmClient

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

print ("\nTodo bien\n")