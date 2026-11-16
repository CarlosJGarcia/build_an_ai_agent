# Connecting to an MCP server
# Uses the Tavily search service through the MCP interface, doesn't use the Tavily library
# This script is the MCP client and at the beginning it runs the MCP Server (Tavily) using node.js

'''
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
    env["NODE_TLS_REJECT_UNAUTHORIZED"] = "0"  # Bypass proxy certificate error (Allow self-signed certificates, disabling TLS certificate validation)

    # Start Tavily MCP Server
    console = Console()
    console.print("\nRun Tavily MCP server", style="gold1")
    server_params = StdioServerParameters(command="npx", args=["-y", "tavily-mcp@latest"], env=env)

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
            result = await session.call_tool("tavily_search", arguments={"query": topic})
            console.print(f'\nTavily search result for topic "{topic}"', style="gold1", highlight=False)
            print(result.content)
            print()

# Main -> Run the async main
if __name__ == "__main__":
    asyncio.run(main())
'''

import os
from tavily import TavilyClient
# from dotenv import load_dotenv
# from mcp.server.fastmcp import FastMCP
from mcp.server.mcpserver import MCPServer


'''
load_dotenv()

tavily_client = TavilyClient(os.getenv("TAVILY_API_KEY"))

# mcp = FastMCP("custom-tavily-search")
mcp = MCPServer("custom-tavily-search")

@mcp.tool()
def search_web(query: str, max_results: int = 5) -> str:
    """
    Search the web using Tavily API.

    Args:
        query: Search query string
        max_results: Maximum number of results to return (default: 5)

    Returns:
        Search results as formatted string
    """
    try:
        response = tavily_client.search(
            query,
            max_results=max_results,
        )
        results = response.get("results", [])
        return "\n\n".join(
            f"Title: {r['title']}\nURL: {r['url']}\nContent: {r['content']}"
            for r in results
        )
    except Exception as e:
        return f"Error searching web: {str(e)}"

if __name__ == "__main__":
    mcp.run(transport='stdio')
'''

print("\nTodo bien\n")