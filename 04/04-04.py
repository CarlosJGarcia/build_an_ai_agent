#  Basic ReAct agent harness - One sample question from the GAIA dataset

import os
import asyncio
from rich.console import Console
from scratch_agents import load_mcp_tools, run_experiment

MODEL_NAME = "nvidia/Qwen3.6-35B-A3B-NVFP4"

# Sample GAIA-formatted problems (replace with your dataset loader if reading from a file)
problems = [
    {
        "id": "test-1",
        "question": "Who won the Nobel Prize in Physics in 2024 and what was it awarded for?"
    },
]

# async def main() wrapper for await load_mcp_tools(tavily_connection)
async def main():

    # Read Tavily key environment variable
    tavily_key = os.getenv("TAVILY_API_KEY")
    if not tavily_key:
        raise ValueError("ERROR: TAVILY_API_KEY environment variable is not set.")

    # Dictionary - matches the exact types expected by StdioServerParameters(**connection) inside load_mcp_tools() function
    tavily_connection = {
        "command": "npx",                           # Value: string
        "args": ["-y", "tavily-mcp@latest"],        # Value: List of two strings
        "env": {"TAVILY_API_KEY": tavily_key}       # Value: Dictionary with one key-value  
    }

    print()
    # Loading MCP tools from the Tavily MCP server
    mcp_tools = await load_mcp_tools(tavily_connection)
    
    models = [MODEL_NAME]
    console = Console()
    console.print(f"\nGAIA test question (run_experiment -> Agent) to the agent with {len(problems)} problem and {len(mcp_tools)} MCP tools", style="gold1", highlight=False)
    results = await run_experiment(problems=list(problems), models=models, tools=mcp_tools)
    
    console.print("\nResults:", style="gold1")
    for r in results:
        print(f"ID: {r['question_id']}")
        print(f"Solvable: {r['is_solvable']}")
        print(f"Steps taken: {r['steps']}")
        print(f"Answer: {r['answer']}\n")
    
# Main -> Run the async main
if __name__ == "__main__":
    asyncio.run(main())