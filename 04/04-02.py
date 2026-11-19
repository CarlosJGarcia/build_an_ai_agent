# 4.14 Using the LLM communication layer

import asyncio
from scratch_agents import LlmClient, LlmRequest, Message

# Async main
async def main():

    # Create client
    client = LlmClient(model="gpt-5-mini")

    # Build request
    request = LlmRequest(instructions=["You are a helpful assistant."], contents=[Message(role="user", content="What is 2 + 2?")])

    # Generate response
    response = await client.generate(request)

    # Response contains the answer
    for item in response.content:
        if isinstance(item, Message):
            print(item.content)            # "4"


# Main -> Run the async main
if __name__ == "__main__":
    asyncio.run(main())