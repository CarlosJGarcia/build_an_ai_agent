# MCP server that exposes the Tavily search service

import os
import sys
from tavily import TavilyClient
from rich.console import Console
from mcp.server.mcpserver import MCPServer


# Read Tavily API key environment variable
tavily_key = os.getenv("TAVILY_API_KEY")
if not tavily_key:
    raise ValueError("ERROR: TAVILY_API_KEY environment variable is not set.")

tavily_client = TavilyClient(os.getenv("TAVILY_API_KEY"))

# Define Tavily MCP Server
console = Console()
console.print("\nDefine Tavily MCP server", style="gold1")
mcp = MCPServer("custom-tavily-search")

# The definition of search_web() must be done once mcp has been instantiated
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
    console.print("Server running. Ctrl+C to exit.\n", style="gold1")
    
    try:
        # Run the server
        mcp.run(transport='stdio')
    except KeyboardInterrupt:
        # Exit cleanly without throwing an error
        console.print("MCP Server stopped\n", style="gold1")
        sys.exit(0)
