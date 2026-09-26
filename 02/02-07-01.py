# Load GAIA dataset Level 1 questions and evaluate two OpenAI-compatible LLMs 
# Report each model’s accuracy and token-processing speed
# Concurrent inference requests
# Reinach 26/Sep/2026

# 02-07-01
# Step 01 - The Async "Plumbing" Test
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
from openai import OpenAI
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

"""
# Coroutine (function defined with async def) that sends a GAIA problem to the model and receives the structured reply
async def solve_problem(model: str, question: str) -> GaiaOutput:
    
    provider = get_provider(model)

    async with PROVIDER_SEMAPHORES[provider]:
        response = await acompletion(
            model=model,
            messages=[
                {"role": "system", "content": gaia_prompt},
                {"role": "user", "content": question},
            ],
            response_format=GaiaOutput,
            num_retries=2,
        )
        finish_reason = response.choices[0].finish_reason
        content = response.choices[0].message.content

        if finish_reason == "refusal" or content is None:
            return GaiaOutput(
                is_solvable=False,
                unsolvable_reason=f"Model refused to answer (finish_reason: {finish_reason})",
                final_answer=""
            )
        return GaiaOutput.model_validate_json(content)
"""


# Answer validation
def is_correct(prediction: str | None, answer: str) -> bool:
    """Check exact match between prediction and answer (case-insensitive)."""
    if prediction is None:
        return False
    return prediction.strip().lower() == answer.strip().lower()


"""
# Evaluate a single problem-model pair and return result.
async def evaluate_gaia_single(problem: dict, model: str) -> dict:
    
    try:
        output = await solve_problem(model, problem["Question"])
        return {
            "task_id": problem["task_id"],
            "model": model,
            "correct": is_correct(output.final_answer, problem["Final answer"]),
            "is_solvable": output.is_solvable,
            "prediction": output.final_answer,
            "answer": problem["Final answer"],
            "unsolvable_reason": output.unsolvable_reason,
        }
    except Exception as e:
        return {
            "task_id": problem["task_id"],
            "model": model,
            "correct": False,
            "is_solvable": None,
            "prediction": None,
            "answer": problem["Final answer"],
            "error": str(e),
        }


# Evaluate all models on all problems.
async def run_experiment(
    problems: list[dict],
    models: list[str],
) -> dict[str, list]:
    
    tasks = [
        evaluate_gaia_single(problem, model)
        for problem in problems
        for model in models
    ]

    all_results = await tqdm_asyncio.gather(*tasks)

    # Group results by model
    results = {model: [] for model in models}
    for result in all_results:
        results[result["model"]].append(result)

    return results
"""
    

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

client = OpenAI(base_url=vllm_url, api_key="EMPTY") 
#client = AsyncOpenAI(base_url=vllm_url, api_key="EMPTY") 


"""
# Inspect the first item in the dataset
console.print(f"\nDataset sample item:", style="gold1")
sample = level1_problems[0] 
for key, value in sample.items():
    content_preview = str(value)[:200].replace('\n', ' ')
    print(f"{key}: {content_preview}...")
"""

print()

# ================================================
# Test step 1: Simple inference with JSON response
# ================================================
"""
console.print(f"Test simple inference with JSON response", style="gold1")



# Model. Inference with a harcoded question
question = "What is the capital of France?"
# List of dictionaries. Named 'messages' for alignment with OpenAI's SDK specification 
messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question}
    ]
"""
"""
console.print("Question:", style="white", highlight=False)
for item in messages:
    console.print(f"{item}", style="white", highlight=False)
"""
"""
console.print(f"Question: {question}", style="white", highlight=False)    

start_time = time.time()
response = client.chat.completions.create(model=MODEL_NAME, messages=messages, temperature=MODEL_TEMPERATURE)

# Response parsing ()
# response -> clean_response -> dict_response -> final_response
clean_response = response.choices[0].message.content.strip()  # Remove trailing \n in the LLM response

# Unwrap Safeguard. Parse the raw string into a standard Python dictionary first
try:
    dict_response = json.loads(clean_response)
except json.JSONDecodeError:
    raise ValueError(f"Model failed to output valid JSON. Raw output: {clean_response}")

# If the model stubbornly wrapped the output in a "properties" key, unwrap it
if "properties" in dict_response:
    raw_dict = dict_response["properties"]

# Parse the clean JSON string into the Pydantic object
final_response = GaiaOutput.model_validate(dict_response)
end_time = time.time()
execution_time_seconds = (end_time - start_time)
speed = response.usage.total_tokens / execution_time_seconds

# print(f"Response: {clean_response}")
# print(f"Response, extrated from JSON using Pydantic: {final_response}")
# print(f"Answer, extrated from JSON using Pydantic: {final_response.final_answer}")
# print(f"Tokens: {response.usage.total_tokens} (Total) = {response.usage.prompt_tokens} (Prompt, including 'messages' list) + {response.usage.completion_tokens} (Completion, this reply including reasoning)")
# console.print(f"Time: {execution_time_seconds:.2f} seconds\n", style="cyan", highlight=False)
print(f"Answer: {final_response.final_answer}")
console.print(f"Time: {execution_time_seconds:.2f} seconds, speed: {speed:.2f} tokens/second\n", style="cyan", highlight=False)
"""

# ==========================================================================================================
# Test step 2: Inference using one GAIA dataset question, with JSON reply. Evaluate answer as right or wrong
# ==========================================================================================================

# Inferencia using the GAIA dataset with JSON reply
console.print(f"Test simple inference from GAIA sample with JSON response:", style="gold1")

# Dataset. Extract the question and the expected ground-truth answer from the first dataset item
sample = gaia_level1_problems[0] 
question = sample["Question"]
expected_answer = sample["Final answer"]

# List of dictionaries. Named 'messages' for alignment with OpenAI's SDK specification 
messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question}
    ]
"""
console.print("Question:", style="white", highlight=False)
for item in messages_gaia:
    console.print(f"{item}", style="white", highlight=False)
"""
console.print(f"Question: {question}", style="white", highlight=False)    

start_time = time.time()
response = client.chat.completions.create(model=MODEL_NAME, messages=messages, temperature=MODEL_TEMPERATURE)
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
    dict_respons = dict_response["properties"]

# Parse into Pydantic object
final_response = GaiaOutput.model_validate(dict_response)
end_time = time.time()
execution_time_seconds = (end_time - start_time)
speed = response.usage.total_tokens / execution_time_seconds

# print(f"\nResponse: {clean_response_gaia}")
# print(f"Pydantic Object: {final_response_gaia}")
# print(f"Answer (LLM): {final_response_gaia.final_answer}")
# print(f"Answer (dataset): {expected_answer}")
print(f"Answer: {final_response.final_answer}")

# Validate if the LLM got it right using the is_correct function
is_match = is_correct(final_response.final_answer, expected_answer)
console.print(f"Match: {is_match}", style="bright_green" if is_match else "red", highlight=False)

console.print(f"Time: {execution_time_seconds:.2f} seconds, speed: {speed:.2f} tokens/second\n", style="cyan", highlight=False)



"""
print(f"Tokens: {response_gaia.usage.total_tokens} (Total) = {response_gaia.usage.prompt_tokens} (Prompt, including 'messages' list) + {response_gaia.usage.completion_tokens} (Completion, this reply including reasoning)")
speed = response_gaia.usage.total_tokens / execution_time_seconds
console.print(f"Time: {execution_time_seconds:.2f} seconds, speed: {speed:.2f} tokens/second\n", style="cyan", highlight=False)
print()
"""

"""
# ==========================================
# Full GAIA Level 1 Validation Loop
# ==========================================
correct_answers = 0
total_problems = len(level1_problems)
console.print(f"\nStarting evaluation of all {total_problems} GAIA level 1 problems", style="gold1", highlight=False)


for i, problem in enumerate(level1_problems, 1):
    gaia_question = problem["Question"]
    expected_answer = problem["Final answer"]

    messages_gaia = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": gaia_question}
    ]
    
    console.print(f"Problem {i}/{total_problems}:", style="white", highlight=False)
    
    start_time = time.time()
    try:
        # Make the API call
        response_gaia = client.chat.completions.create(
            model=MODEL_NAME, 
            messages=messages_gaia,
            temperature=MODEL_TEMPERATURE
        )
        
        clean_response_gaia = response_gaia.choices[0].message.content.strip()
        
        # Sanitizer
        clean_response_gaia = re.sub(r'^\{\s*\"?\{', '{', clean_response_gaia)
        
        # Unwrap Safeguard
        dict_response_gaia = json.loads(clean_response_gaia)
        if "properties" in dict_response_gaia:
            dict_response_gaia = dict_response_gaia["properties"]
            
        # Parse into Pydantic object
        final_response_gaia = GaiaOutput.model_validate(dict_response_gaia)
        predicted_answer = final_response_gaia.final_answer
        
    except Exception as e:
        console.print(f"Error processing problem {i}: {e}", style="red")
        predicted_answer = ""
        
    end_time = time.time()
    execution_time_seconds = (end_time - start_time)
    
    # Validate if the LLM got it right using the is_correct function
    is_match = is_correct(predicted_answer, expected_answer)
    if is_match:
        correct_answers += 1
        
    print(f"Answer (LLM): {predicted_answer}")
    print(f"Answer (dataset): {expected_answer}")
    
    console.print(f"Match: {is_match}", style="cyan" if is_match else "red", highlight=False)
    speed = response_gaia.usage.total_tokens / execution_time_seconds
    console.print(f"Time: {execution_time_seconds:.2f} seconds, speed: {speed:.2f} tokens/second", style="cyan", highlight=False)
    print("-" * 48)

# Display final score
console.print(f"\nEvaluation results:", style="gold1", highlight=False)
print(f"Correct answers = Accuracy {correct_answers} / {total_problems} ({(correct_answers / total_problems * 100):.0f}%)")
print()
"""

