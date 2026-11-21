import os
import asyncio
from scratch_agents import load_mcp_tools

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
    
    


    print("Todo bien.\n")
    
# Main -> Run the async main
if __name__ == "__main__":
    asyncio.run(main())