# 4.14 Using the LLM communication layer

import asyncio
from rich.console import Console
from scratch_agents import LlmClient, LlmRequest, Message

MODEL_NAME = "nvidia/Qwen3.6-35B-A3B-NVFP4"
SYSTEM_PROMPT = "You are a helpful assistant. Output plain text only. Do not use emojis or emoticons."

# Async main
async def main():

    console = Console()
     
    # client = AsyncOpenAI(base_url=vllm_url, api_key="EMPTY")
    client = LlmClient(model=MODEL_NAME)

    question = "What is 2 + 2?"
    console.print(f"\nInference request (LlmClient) to the model: '{question}'", style="gold1", highlight=False)

    # Inference request
    request = LlmRequest(instructions=[SYSTEM_PROMPT], contents=[Message(role="user", content=question)])
    response = await client.generate(request)
    
    
    # Response contains the answer
    for item in response.content:
        if isinstance(item, Message) and item.role == "assistant":
            clean_response = item.content.strip()
            print(f"Reply: {clean_response}\n")                     # Response should be "4" (2+2)


# Main -> Run the async main
if __name__ == "__main__":
    asyncio.run(main())