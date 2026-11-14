# Web search function using Tavily, now with error handling

import os
from tavily import TavilyClient
from rich.console import Console

# Read Tavily key environment variable
tavily_key = os.getenv("TAVILY_API_KEY")
if not tavily_key:
    raise ValueError("ERROR: TAVILY_API_KEY environment variable is not set.")

# Initialize the tavily_client so tavily_client.search() works when other module calls search_web()
tavily_client = TavilyClient(tavily_key)

# Web search function with options and error handling
def search_web(query, max_results=5, topic="general", time_range=None):
    try:
        response = tavily_client.search(query, max_results=max_results, topic=topic, time_range=time_range)
        return response.get("results")
    except Exception as e:
        return f"Error: Search failed - {e}"

# Main
if __name__ == "__main__":    
    # Read Tavily key environment variable
    tavily_key = os.getenv("TAVILY_API_KEY")
    if not tavily_key:
        raise ValueError("ERROR: TAVILY_API_KEY environment variable is not set.")

    console = Console()
    console.print (f"\nInitializing Tavily client", style="gold1", highlight=False)
    tavily_client = TavilyClient(tavily_key)

    string = "Kipchoge's marathon world record"
    console.print (f"\nSearch: {string}", style="gold1", highlight=False)
    print(f"{search_web(string, max_results=2)}\n")
