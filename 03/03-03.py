# Basic web search function using Tavily

import os
from tavily import TavilyClient

def search_web(query: str, max_results: int = 2) -> list:
    response = tavily_client.search(query, max_results=max_results)
    return response.get("results")


# Read Tavily key environment variable
tavily_key = os.getenv("TAVILY_API_KEY")
if not tavily_key:
    raise ValueError("ERROR: TAVILY_API_KEY environment variable is not set.")

tavily_client = TavilyClient(tavily_key)
print(search_web("Kipchoge's marathon world record"))
