# 4.29 Using structured output in practice

import asyncio
from pydantic import BaseModel
from rich.console import Console
from typing import Literal, List
from scratch_agents import Agent, LlmClient

# Define the output structure
class SentimentAnalysis(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float
    key_phrases: List[str]

MODEL_NAME = "nvidia/Qwen3.6-35B-A3B-NVFP4"
# SYSTEM_PROMPT = "Analyze the sentiment of the provided text. Do not use emojis or emoticons."   # The system prompt can't include "Output plain text only." as it will prevent the model to generate JSON output
SYSTEM_PROMPT = "Analyze the sentiment of the provided text. You must use the final_answer tool to output your result. Do not use emojis or emoticons."


# Async main
async def main():

    console = Console()

    # Initialize the agent with the output_type
    agent = Agent(model=LlmClient(model=MODEL_NAME), tools=[], instructions=SYSTEM_PROMPT, output_type=SentimentAnalysis)

    # Sentiment analysis request
    text = "This product exceeded my expectations! Highly recommend."
    console.print(f"\nSentiment analysis (Agent) to the agent for text: '{text}'", style="gold1", highlight=False)
    result = await agent.run(text)

    # Access the structured data safely
    console.print("Results:", style="gold1")
    print(f"- Sentiment: {result.output.sentiment}")      
    print(f"- Confidence: {result.output.confidence}")    
    print(f"- Key phrases: {result.output.key_phrases}\n") 
   

# Main -> Run the async main
if __name__ == "__main__":
    asyncio.run(main())