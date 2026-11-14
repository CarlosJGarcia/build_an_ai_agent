#Connecting to an MCP server

import os
import asyncio
from rich.console import Console
from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters

async def main():
    # Read Tavily key environment variable
    tavily_key = os.getenv("TAVILY_API_KEY")
    if not tavily_key:
        raise ValueError("ERROR: TAVILY_API_KEY environment variable is not set.")

    env = os.environ.copy()
    env["TAVILY_API_KEY"] = tavily_key
    env["NODE_TLS_REJECT_UNAUTHORIZED"] = "0"  # Bypass proxy certificate error

    console = Console()
    console.print("\nRun Tavily MCP server", style="gold1")
    server_params = StdioServerParameters(command="npx", args=["-y", "tavily-mcp@latest"], env=env)


    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # List available tools
            tools_result = await session.list_tools()
            console.print("\nAvailable tools:", style="gold1")
            for tool in tools_result.tools:
                print(f"  - {tool.name}: {tool.description[:60]}...")

            topic = "2026 Nobel Physics"
            result = await session.call_tool("tavily_search", arguments={"query": topic})
            console.print(f'\nTavily search result for topic "{topic}"', style="gold1", highlight=False)
            print(result.content)
            print()

# Run the async main function
if __name__ == "__main__":
    asyncio.run(main())