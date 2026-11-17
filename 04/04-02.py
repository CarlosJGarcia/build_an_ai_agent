# 4.14 Using the LLM communication layer

from scratch_agents import LlmClient, LlmRequest, Message

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