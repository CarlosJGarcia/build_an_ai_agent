# Connecting to an MCP server
# Uses the Tavily search service through the MCP interface, doesn't use the Tavily library
# This script is the MCP client and at the beginning it runs the MCP Server (03-08.py) using node.js

import os
import asyncio
from rich.console import Console
from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters

from p03_05 import format_tool_definition

# Convert MCP tools to OpenAI format
def mcp_tools_to_openai_format(mcp_tools):
    """Convert MCP tool definitions to OpenAI tool format."""
    return [format_tool_definition(name=tool.name, description=tool.description, parameters=tool.input_schema) for tool in mcp_tools.tools]


# Async main
async def main():
    # Read Tavily API key environment variable
    tavily_key = os.getenv("TAVILY_API_KEY")
    if not tavily_key:
        raise ValueError("ERROR: TAVILY_API_KEY environment variable is not set.")

    env = os.environ.copy()
    env["TAVILY_API_KEY"] = tavily_key

    # Start Tavily MCP Server
    console = Console()
    console.print("\nRun Tavily MCP server", style="gold1")
    # server_params = StdioServerParameters(command="npx", args=["-y", "tavily-mcp@latest"], env=env)
    # server_params = StdioServerParameters(command="uv", args=["run", "tavily_mcp_server.py"], env={"TAVILY_API_KEY": os.getenv("TAVILY_API_KEY")}
    server_params = StdioServerParameters(command="uv", args=["run", "03-09.py"], env=env)


    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # List the tools available in the MCP Server
            tools_result = await session.list_tools()
            console.print("\nAvailable tools:", style="gold1")
            for tool in tools_result.tools:
                print(f"  - {tool.name}: {tool.description[:60]}...")

            # List the tools in OpenAI format
            console.print("\nAvailable tools, in OpenAI format:", style="gold1")
            openai_format_tools = mcp_tools_to_openai_format(tools_result)
            for tool in openai_format_tools:
                print(tool)

            topic = "2026 Nobel Physics"
            result = await session.call_tool("search_web", arguments={"query": topic})
            console.print(f'\nTavily search result for topic "{topic}"', style="gold1", highlight=False)
            print(result.content)
            print()

# Main -> Run the async main
if __name__ == "__main__":
    asyncio.run(main())