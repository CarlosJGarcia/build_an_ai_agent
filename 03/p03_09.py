# MCP server that exposes the Tavily search service
# This version does not print anything on screen. Just communicates with the MCP client over stdio

import os
import sys
from tavily import TavilyClient
from mcp.server.mcpserver import MCPServer


# Read Tavily API key environment variable
tavily_key = os.getenv("TAVILY_API_KEY")
if not tavily_key:
    raise ValueError("ERROR: TAVILY_API_KEY environment variable is not set.")

tavily_client = TavilyClient(os.getenv("TAVILY_API_KEY"))

# Define Tavily MCP Server
mcp = MCPServer("custom-tavily-search")

# Because of the mcp decorator, search_web() needs to be defined once mcp has been instantiated
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


# Main
if __name__ == "__main__":
    try:
        # Run the server
        mcp.run(transport='stdio')
    except KeyboardInterrupt:
        # Exit cleanly without throwing an error
        sys.exit(0)
