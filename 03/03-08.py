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
    console.print("Server running. Ctrl+C to exit.\n", style="gold1")
    
    try:
        # Run the server
        mcp.run(transport='stdio')
    except KeyboardInterrupt:
        # Exit cleanly without throwing an error
        console.print("MCP Server stopped\n", style="gold1")
        sys.exit(0)

# When you spend time wrestling with tracebacks, imports and holding abstract architectural concepts (MCP servers, stdio streams and API integrations) in your working memory, your brain is under intense tension.
# The moment you see that clean Server running message, the tension drops instantly and your brain releases a massive spike of dopamine.
# Physically, this often manifests as a "nervous" or restless feeling or a temporary inability to focus on the next line of code.

# Cognitive consolidation. Your brain needs a few minutes offline to transfer the mental model of what you just built from short-term memory into long-term memory.

# The cycle of deep concentration and frustration -> breakthrough -> brief rest -> next challenge is the natural rhythm of deep engineering work. 
# Building AI agents from scratch is heavy, cutting-edge mental lifting. Let the adrenaline settle naturally. 

# Adrenaline build-up: While deep focusing and wrestling with complex architectures and tracebacks in your head, the brain treats this intense cognitive load as a stressor. It continuously pumps adrenaline and cortisol into your system to keep you hyper-alert, focused and able to hold all those variables in the working memory.
# Dopamine spike: The exact second code works and you see the "Server running" message, the brain registers a massive win. It immediately fires a spike of dopamine, which is the neurotransmitter responsible for reward, satisfaction, and behavioral reinforcement.
# The restless aftermath: Dopamine tells the brain, "We won, the problem is solved, relax." However, the adrenaline from the previous 30' of intense focus is still circulating in the bloodstream. It takes a few minutes to metabolize and clear out the leftover stress hormones

# This is why one needs to step away before focusing again:
# - Transfer the mental model from short-term to long-term memory
# - Let the adrenaline settle

# Know and respect your biological limits instead of fighting them.