import os
from openai import OpenAI
from rich.console import Console


console = Console()
vllm_server_fqdn = os.getenv("VLLM_SERVER_FQDN")
if not vllm_server_fqdn:
    raise ValueError("ERROR: VLLM_SERVER_FQDN environment variable is not set.")
vllm_url = f"http://{vllm_server_fqdn}:8000/v1"
MODEL_NAME = "nvidia/Qwen3.6-35B-A3B-NVFP4"

client = OpenAI(base_url=vllm_url, api_key="EMPTY")

# Initialize conversation 'message' history with system prompt. This indicates the conversational, instruction-tuned LLM, how it should answer 
messages = [
    {"role": "system", "content": "You are a helpful assistant."}
]

console.print("\nAI Chat Terminal Started", style="gold1")
console.print("Type your message and press Enter. Enter 'quit' to exit.\n", style="gold1", highlight=False)

active = True

while active:
    # Get user input
    user_input = input("You: ")

    # Check for quit command
    if user_input.strip().lower() == 'quit':
        console.print("\nEnd of conversation. Goodbye!\n", style="gold1"))
        active = False
        continue

    # 1. Add user message to history
    messages.append({"role": "user", "content": user_input})

    try:
        # 2. Call local vLLM server passing full conversation history
        response = client.chat.completions.create(
            model=MODEL_NAME, 
            messages=messages
        )

        clean_response = response.choices[0].message.content.strip()

        # 3. Add assistant response back to history to preserve state
        messages.append({"role": "assistant", "content": clean_response})

        # 4. Display output and token usage
        print(f"\nAI: {clean_response}\n")
        
        usage = response.usage
        console.print(
            f"[dim]Tokens: {usage.total_tokens} Total = {usage.prompt_tokens} (Prompt) + {usage.completion_tokens} (Completion)[/dim]\n",
            style="white"
        )

    except Exception as e:
        console.print(f"[bold red]Error communicating with vLLM server:[/bold red] {e}\n")
        # Remove the unfulfilled user message so history isn't corrupted
        messages.pop()