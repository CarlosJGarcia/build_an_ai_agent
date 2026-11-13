# Basic web search function using Tavily

import os
from tavily import TavilyClient
from rich.console import Console

# Web search function with options
def search_web(query: str, max_results = 2):
    response = tavily_client.search(query, max_results=max_results)
    return response.get("results")

# Read Tavily key environment variable
tavily_key = os.getenv("TAVILY_API_KEY")
if not tavily_key:
    raise ValueError("ERROR: TAVILY_API_KEY environment variable is not set.")

console = Console()
console.print (f"\nInitializing Tavily client", style="gold1", highlight=False)
tavily_client = TavilyClient(tavily_key)

string = "Kipchoge's marathon world record"
console.print (f"\nSearch: {string}", style="gold1", highlight=False)
print(f"{search_web(string)}\n")
