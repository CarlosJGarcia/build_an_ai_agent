# Agent example

from scratch_agents import Agent, LlmClient
from scratch_agents.tools import calculator, search_web

# The agent uses a model (with a system prompt) and two tools (calculator and web search) 
agent = Agent(model=LlmClient(model="gpt-5-mini"), tools=[calculator, search_web], instructions="You are a helpful assistant")

# The agent receives a user question
# It decides whether to use tools, executes them if necessary and returns an answer
result = await agent.run("What is 1234 * 5678?")