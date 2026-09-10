# Goal: Build a research agent that get information from multiple sources, analyzes findings and produce comprehensive answers
# Use the GAIA benchmark to determine if the agent is doing that or not and measure how well

# Loads the GAIA (General AI Assistants) dataset from Meta and Hugging Face
# 'Convinces' the LLM to reply using data structures (JSON)
# OpenAI’s Chat Completions API 
# Reinach 04/Sep/2026

import time
import os, re, json
from openai import OpenAI
from pydantic import BaseModel
from rich.console import Console
from datasets import load_dataset

# Define data structure (named GaiaOutput) for the LLM reply, using pydantic 
class GaiaOutput(BaseModel):
    is_solvable: bool
    unsolvable_reason: str = ""
    final_answer: str = ""

# Define the same data, structured in JSON format, to be able to tell the LLM which format I expect to get
# In JSON terminology, the python data type bool is called a "boolean"
schema_template = {
    "is_solvable": "boolean",
    "unsolvable_reason": "string",
    "final_answer": "string"
}

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


# Answer validation
def is_correct(prediction: str | None, answer: str) -> bool:
    """Check exact match between prediction and answer (case-insensitive)."""
    if prediction is None:
        return False
    return prediction.strip().lower() == answer.strip().lower()



async def evaluate_gaia_single(problem: dict, model: str) -> dict:
    """Evaluate a single problem-model pair and return result."""
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


async def run_experiment(
    problems: list[dict],
    models: list[str],
) -> dict[str, list]:
    """Evaluate all models on all problems."""
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


# Main
console = Console()
schema_string = json.dumps(schema_template)
console.print(f"\nJSON schema_string: {schema_string}", style="gold1", highlight=False)

vllm_server_fqdn = os.getenv("VLLM_SERVER_FQDN")
if not vllm_server_fqdn:
    raise ValueError("ERROR: VLLM_SERVER_FQDN environment variable is not set.")
vllm_url = f"http://{vllm_server_fqdn}:8000/v1"
MODEL_NAME = "nvidia/Qwen3.6-35B-A3B-NVFP4"
MODEL_TEMPERATURE = 0.0

ollama_server_fqdn = os.getenv("OLLAMA_SERVER_FQDN")
if not ollama_server_fqdn:
    raise ValueError("ERROR: OLLAMA_SERVER_FQDN environment variable is not set.")
ollama_url = f"http://{ollama_server_fqdn}:11434/v1"
BIS_MODEL_NAME = "qwen3:14b"



# GAIA’s standard evaluation prompt, instructs the model to provide answers in a consistent format
SYSTEM_PROMPT = "You are a general AI assistant. I will ask you a question. First, determine if you can solve this problem with your current capabilities "
SYSTEM_PROMPT += "and set “is_solvable” accordingly. If you can solve it, set “is_solvable” to true and provide your answer in “final_answer”. "
SYSTEM_PROMPT += "If you cannot solve it, set “is_solvable” to false and explain why in “unsolvable_reason”. Your final answer should be a number OR "
SYSTEM_PROMPT += "as few words as possible OR a comma-separated list of numbers and/or strings. If you are asked for a number, don’t use a comma to write "
SYSTEM_PROMPT += "your number; also don’t use units such as $ or a percent sign unless specified otherwise. If you are asked for a string, don’t use articles, "
SYSTEM_PROMPT += "neither abbreviations (e.g., for cities), and write the digits in plain text unless specified otherwise. If you are asked for a comma-separated "
SYSTEM_PROMPT += "list, apply the above rules depending on whether the element is a number or a string."
SYSTEM_PROMPT += "Output plain text only. Do not use emojis or emoticons. "
SYSTEM_PROMPT += f"Output ONLY a valid JSON object matching this schema: {schema_string}. "
SYSTEM_PROMPT += "Do not include markdown blocks or schema keywords like 'properties' in your final output."

DATASET_ID = "gaia-benchmark/GAIA"
SUBSET = "2023_level1"

console.print(f"\nLoading GAIA dataset, Level 1, validation split", style="gold1", highlight=False)
level1_problems = load_dataset(DATASET_ID, SUBSET, split="validation")
console.print(f"Dataset loaded successfully", style="gold1")
print(f"Number of problems: {len(level1_problems)}")
print(f"Dataset structure: {level1_problems}")

# Inspecting the first item in Level 1, 'validation' split
console.print(f"\nDataset sample item:", style="gold1")
sample = level1_problems[0] 
for key, value in sample.items():
    content_preview = str(value)[:200].replace('\n', ' ')
    print(f"{key}: {content_preview}...")

print()

# Inferencia simple
client = OpenAI(base_url=vllm_url, api_key="EMPTY") 

console.print(f"Test simple inference:", style="gold1")

# List of dictionaries. Should be named 'messages' for alignment with the examples in OpenAI's SDK specification 
messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": 'Based strictly on your underlying architecture, are you a standard Dense model or a Mixture-of-Experts (MoE) model? Set "is_solvable" to true, and output strictly the word "Dense" or "MoE" in the final_answer.'}
    ]

console.print("Question:", style="white", highlight=False)
for item in messages:
    console.print(f"{item}", style="white", highlight=False)

start_time = time.time()
response = client.chat.completions.create(model=MODEL_NAME, messages=messages)
clean_response = response.choices[0].message.content.strip()  # Remove trailing \n in the LLM response

# Sanitizer. Regex that catches any variation of a stuttered opening brace ({{, {"{, etc.) and flattens it.
clean_response = re.sub(r'^\{\s*\"?\{', '{', clean_response)

# Unwrap Safeguard. Parse the raw string into a standard Python dictionary first
try:
    dict_response = json.loads(clean_response)
except json.JSONDecodeError:
    raise ValueError(f"Model failed to output valid JSON. Raw output: {clean_response}")

# If the model stubbornly wrapped the output in a "properties" key, unwrap it
if "properties" in dict_response:
    raw_dict = dict_response["properties"]

# Manually parse the clean JSON string into the Pydantic object
final_response = GaiaOutput.model_validate(dict_response)
end_time = time.time()
execution_time_seconds = (end_time - start_time)

print(f"Response: {clean_response}")
print(f"Response, extrated from JSON using Pydantic: {final_response}")
print(f"Answer, extrated from JSON using Pydantic: {final_response.final_answer}")
print(f"Tokens: {response.usage.total_tokens} (Total) = {response.usage.prompt_tokens} (Prompt, including 'messages' list) + {response.usage.completion_tokens} (Completion, this reply including reasoning)")
console.print(f"Time: {execution_time_seconds:.2f} seconds\n", style="cyan", highlight=False)
print()


# ---------------------------
# Segundo modelo
# ---------------------------


# Inferencia simple
client_bis = OpenAI(base_url=ollama_url, api_key="EMPTY") 

console.print(f"Test simple inference with the second model:", style="gold1")

# List of dictionaries. Should be named 'messages' for alignment with the examples in OpenAI's SDK specification 
messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": 'Based strictly on your underlying architecture, are you a standard Dense model or a Mixture-of-Experts (MoE) model? Set "is_solvable" to true, and output strictly the word "Dense" or "MoE" in the final_answer.'}
    ]

console.print("Question:", style="white", highlight=False)
for item in messages:
    console.print(f"{item}", style="white", highlight=False)

start_time = time.time()
response = client_bis.chat.completions.create(model=BIS_MODEL_NAME, messages=messages)
clean_response = response.choices[0].message.content.strip()  # Remove trailing \n in the LLM response

# Sanitizer. Regex that catches any variation of a stuttered opening brace ({{, {"{, etc.) and flattens it.
clean_response = re.sub(r'^\{\s*\"?\{', '{', clean_response)

# Unwrap Safeguard. Parse the raw string into a standard Python dictionary first
try:
    dict_response = json.loads(clean_response)
except json.JSONDecodeError:
    raise ValueError(f"Model failed to output valid JSON. Raw output: {clean_response}")

# If the model stubbornly wrapped the output in a "properties" key, unwrap it
if "properties" in dict_response:
    raw_dict = dict_response["properties"]

# Manually parse the clean JSON string into the Pydantic object
final_response = GaiaOutput.model_validate(dict_response)
end_time = time.time()
execution_time_seconds = (end_time - start_time)

print(f"Response: {clean_response}")
print(f"Response, extrated from JSON using Pydantic: {final_response}")
print(f"Answer, extrated from JSON using Pydantic: {final_response.final_answer}")
print(f"Tokens: {response.usage.total_tokens} (Total) = {response.usage.prompt_tokens} (Prompt, including 'messages' list) + {response.usage.completion_tokens} (Completion, this reply including reasoning)")
console.print(f"Time: {execution_time_seconds:.2f} seconds\n", style="cyan", highlight=False)
print()


"""
# Test inference using the GAIA dataset sample
console.print(f"Inference with GAIA sample:", style="gold1")

# Extract the question and the expected ground-truth answer from the sample
gaia_question = sample["Question"]
expected_answer = sample["Final answer"]

messages_gaia = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": gaia_question}
]

console.print("Question:", style="white", highlight=False)
for item in messages_gaia:
    console.print(f"{item}", style="white", highlight=False)

# Make the API call (including the MODEL_TEMPERATURE variable defined earlier)
start_time = time.time()
response_gaia = client.chat.completions.create(
    model=MODEL_NAME, 
    messages=messages_gaia,
    temperature=MODEL_TEMPERATURE
)

clean_response_gaia = response_gaia.choices[0].message.content.strip()

# Sanitizer
clean_response_gaia = re.sub(r'^\{\s*\"?\{', '{', clean_response_gaia)

# Unwrap Safeguard
try:
    dict_response_gaia = json.loads(clean_response_gaia)
except json.JSONDecodeError:
    raise ValueError(f"Model failed to output valid JSON. Raw output: {clean_response_gaia}")

if "properties" in dict_response_gaia:
    dict_response_gaia = dict_response_gaia["properties"]

# Parse into Pydantic object
final_response_gaia = GaiaOutput.model_validate(dict_response_gaia)
end_time = time.time()
execution_time_seconds = (end_time - start_time)

print(f"\nResponse: {clean_response_gaia}")
print(f"Pydantic Object: {final_response_gaia}")
print(f"Answer (LLM): {final_response_gaia.final_answer}")
print(f"Answer (dataset): {expected_answer}")


# Validate if the LLM got it right using the is_correct function
is_match = is_correct(final_response_gaia.final_answer, expected_answer)
console.print(f"Match: {is_match}", style="cyan" if is_match else "red", highlight=False)

print(f"Tokens: {response_gaia.usage.total_tokens} (Total) = {response_gaia.usage.prompt_tokens} (Prompt, including 'messages' list) + {response_gaia.usage.completion_tokens} (Completion, this reply including reasoning)")
speed = response_gaia.usage.total_tokens / execution_time_seconds
console.print(f"Time: {execution_time_seconds:.2f} seconds, speed: {speed:.2f} tokens/second\n", style="cyan", highlight=False)
print()


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
