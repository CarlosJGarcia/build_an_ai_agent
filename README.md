**Environment installation** 

```bash
conda create -n ai_agent_scratch python=3.13 -y 
conda activate ai_agent_scratch 
conda install -c conda-forge rich -y
pip install openai
pip install datasets
pip install tavily-python
```

To check versions run `python appendix_a/version.py`

# Environment variables

| Name | Description |
|---------|------------------|
| VLLM_SERVER_FQDN | FQDN of the workstation running Qwen 3.5 32B on vLLM |
| TAVILY_API_KEY | Tavily API key |


# My approach to the book

The book’s verbatim approach:
- Jupyter Notebooks
- LiteLLM
- Paid OpenAI/Anthropic subscriptions

I am adapting the code to my preferred local stack, which is:
- VS Code
- OpenAI Python SDK
- Local LLMs: Qwen 3.5 32B on vLLM and Qwen 3 14B on Ollama


VS Code instead of Jupyter Notebooks: I understand Jupyter is great for prototyping and is heavily used by professionals, but I am writing everything in standard Python scripts within VS Code. It just feels like a more structured development environment and I prefer the editor.

OpenAI Python SDK instead of LiteLLM: I appreciate LiteLLM’s multi-vendor routing, but both vLLM and Ollama provide OpenAI-compatible endpoints. I prefer to use the official OpenAI Python library directly rather than adding a wrapper.

Local LLMs instead of Paid APIs: To avoid worrying about API billing, I am running local Qwen 3.5 32B on vLLM and Qwen 3 14B on Ollama instead of using OpenAI and Anthropic.

# Models

| Name | Description |
|---------|------------------|
| nvidia/Qwen3.6-35B-A3B-NVFP4 | |
| | |

# Documentation references

| Name | Description |
|---------|---------|
| [OpenAI Chat Completions API](https://developers.openai.com/api/reference/chat-completions/overview) | |
| [NousResearch Hermes-function-calling-v1](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1) | Dataset with tool definitions and corresponding tool calls, used to train models for function calling |