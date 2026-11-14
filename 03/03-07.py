#Connecting to an MCP server

import os
import asyncio
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

    server_params = StdioServerParameters(command="npx", args=["-y", "tavily-mcp@latest"], env=env)

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # List available tools
            tools_result = await session.list_tools()
            print("Available tools:")
            for tool in tools_result.tools:
                print(f"  - {tool.name}: {tool.description[:60]}...")

            result = await session.call_tool("tavily_search", arguments={"query": "2025 Nobel Physics"})
            print("Search Result:")
            print(result.content)

# Run the async main function
if __name__ == "__main__":
    asyncio.run(main())