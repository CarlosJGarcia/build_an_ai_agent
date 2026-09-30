# Load GAIA dataset Level 1 questions and evaluate two OpenAI-compatible LLMs 
# Report each model’s accuracy and token-processing speed
# Concurrent inference requests
# Reinach 26/Sep/2026

# 02-07-01
# Step 01 - Async "Plumbing" Test
#    - Move the inference to a function
#    - Switch from OpenAI to AsyncOpenAI
#    - Wrap the function in async def and use asyncio.run() for just one single question.

# Libraries
# 1. Datasets (Hugging Face) to load GAIA (General AI Assistants) dataset created by Meta and Hugging Face
# 2. Chat Completions API (OpenAI) - Inference
# 3. JSON (JavaScript Object Notation) - Prompt engineering 'convinces' the LLM to reply using data structures (JSON). Serialize the pydantic data for interaction with the model
# 4. Pydantic - Validate the structure of the response from the LLM
# 5. Async - Concurrent inference

import os
import re
import json
import time
import asyncio
from openai import AsyncOpenAI    
from pydantic import BaseModel
from rich.console import Console
from datasets import load_dataset

# Using pydantic, define a data structure (class GaiaOutput) for the LLM reply
# Class GaiaOutput(CamelCase), variables = name of each data field, type of each variable = type of each data field
class GaiaOutput(BaseModel):
    is_solvable: bool
    unsolvable_reason: str = ""
    final_answer: str = ""

# Using JSON format, define the same data structure, to be able to tell the LLM in which format I expect to get the response
# Dictionary gaia_output_json_schema (snake_case), keys = names of each data field, values = type of each data field
# In JSON terminology, the python data type bool is called a "boolean"
gaia_output_json_schema = {
    "is_solvable": "boolean",
    "unsolvable_reason": "string",
    "final_answer": "string"
}

# Function that sends a question to the model and returns the reply using the GaiaOutput format and the number of tokens processed
async def inference(question):

    # List of dictionaries. Named 'messages' for alignment with OpenAI's SDK specification 
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question}
    ]
    response = await client.chat.completions.create(model=MODEL_NAME, messages=messages, temperature=MODEL_TEMPERATURE)

    # Response parsing ()
    # response -> clean_response -> dict_response -> final_response
    clean_response = response.choices[0].message.content.strip()      # Remove trailing \n in the LLM response
    clean_response = re.sub(r'^\{\s*\"?\{', '{', clean_response)      # Sanitizer

    # Unwrap Safeguard.Parse the raw string into a standard Python dictionary first
    try:
        dict_response = json.loads(clean_response)
    except json.JSONDecodeError:
        raise ValueError(f"Model failed to output valid JSON. Raw output: {clean_response}")

    # If the model stubbornly wrapped the output in a "properties" key, unwrap it
    if "properties" in dict_response:
        dict_response = dict_response["properties"]

    # Parse into Pydantic object
    final_response = GaiaOutput.model_validate(dict_response)
        
    return final_response, response.usage.total_tokens


# Answer validation
def is_correct(prediction: str | None, answer: str) -> bool:
    """Check exact match between prediction and answer (case-insensitive)."""
    if prediction is None:
        return False
    return prediction.strip().lower() == answer.strip().lower()


# ==========================================================================================================
# Inference using one GAIA dataset question, with JSON reply. Evaluate answer as right or wrong
# ==========================================================================================================
async def main():

    # Inferencia using the GAIA dataset with JSON reply
    console.print(f"Test simple inference from GAIA sample with JSON response:", style="gold1")

    # Dataset. Extract the question and the expected ground-truth answer from the first dataset item
    sample = gaia_level1_problems[0] 
    question = sample["Question"]
    expected_answer = sample["Final answer"]

    console.print(f"Question: {question}", style="white", highlight=False)
    start_time = time.time()
    response, tokens = await inference(question)
    end_time = time.time()

    execution_time_seconds = (end_time - start_time)
    speed = tokens / execution_time_seconds

    print(f"Answer: {response.final_answer}")

    # Validate if the LLM got it right using the is_correct function
    is_match = is_correct(response.final_answer, expected_answer)
    console.print(f"Match: {is_match}", style="bright_green" if is_match else "red", highlight=False)

    console.print(f"Time: {execution_time_seconds:.2f} seconds, speed: {speed:.2f} tokens/second\n", style="cyan", highlight=False)


# =====================
# Main - Initialization
# =====================
console = Console()

# Dataset. Load GAIA Dataset, subset Level 1, validation split
SUBSET = "2023_level1"
DATASET_ID = "gaia-benchmark/GAIA"
console.print(f"\nLoading GAIA dataset", style="gold1", highlight=False)
gaia_level1_problems = load_dataset(DATASET_ID, SUBSET, split="validation")
console.print(f"Dataset loaded successfully. Number of problems: {len(gaia_level1_problems)}", style="gold1", highlight = False)

# Async. Initialize the semaphore and the async client
TOTAL_QUESTIONS = 50          # Limit to 50 requests
CONCURRENT_QUESTIONS = 10     # Limit to 10 concurrent requests
semaphore = asyncio.Semaphore(CONCURRENT_QUESTIONS)

# Model. First model
vllm_server_fqdn = os.getenv("VLLM_SERVER_FQDN")
if not vllm_server_fqdn:
    raise ValueError("ERROR: VLLM_SERVER_FQDN environment variable is not set.")
vllm_url = f"http://{vllm_server_fqdn}:8000/v1"
MODEL_NAME = "nvidia/Qwen3.6-35B-A3B-NVFP4"
MODEL_TEMPERATURE = 0.0

# Prompt engineering. GAIA’s standard evaluation prompt, instructs the model to provide answers in a consistent format
gaia_output_string = json.dumps(gaia_output_json_schema)
# console.print(f"\nJSON schema_string: {gaia_output_string}", style="gold1", highlight=False)
SYSTEM_PROMPT = "You are a general AI assistant. I will ask you a question. First, determine if you can solve this problem with your current capabilities "
SYSTEM_PROMPT += "and set “is_solvable” accordingly. If you can solve it, set “is_solvable” to true and provide your answer in “final_answer”. "
SYSTEM_PROMPT += "If you cannot solve it, set “is_solvable” to false and explain why in “unsolvable_reason”. Your final answer should be a number OR "
SYSTEM_PROMPT += "as few words as possible OR a comma-separated list of numbers and/or strings. If you are asked for a number, don’t use a comma to write "
SYSTEM_PROMPT += "your number; also don’t use units such as $ or a percent sign unless specified otherwise. If you are asked for a string, don’t use articles, "
SYSTEM_PROMPT += "neither abbreviations (e.g., for cities), and write the digits in plain text unless specified otherwise. If you are asked for a comma-separated "
SYSTEM_PROMPT += "list, apply the above rules depending on whether the element is a number or a string."
SYSTEM_PROMPT += "Output plain text only. Do not use emojis or emoticons. "
SYSTEM_PROMPT += f"Output ONLY a valid JSON object matching this schema: {gaia_output_string}. "
SYSTEM_PROMPT += "Do not include markdown blocks or schema keywords like 'properties' in your final output."

# Model. Open connection
client = AsyncOpenAI(base_url=vllm_url, api_key="EMPTY") 
print()

# Run the main block that contains the async calls
asyncio.run(main())

