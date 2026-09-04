# Loads the GAIA (General AI Assistants) dataset from HuggingFace
# Reinach 04/Sep/2026

from pydantic import BaseModel
from rich.console import Console
from datasets import load_dataset

# Define data structure (named GaiaOutput) using pydantic 
class GaiaOutput(BaseModel):
    is_solvable: bool
    unsolvable_reason: str = ""
    final_answer: str = ""

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
SYSTEM_PROMPT = "You are a general AI assistant. I will ask you a question. First, determine if you can solve this problem with your current capabilities "
SYSTEM_PROMPT += "and set “is_solvable” accordingly. If you can solve it, set “is_solvable” to true and provide your answer in “final_answer”. "
SYSTEM_PROMPT += "If you cannot solve it, set “is_solvable” to false and explain why in “unsolvable_reason”. Your final answer should be a number OR "
SYSTEM_PROMPT += "as few words as possible OR a comma-separated list of numbers and/or strings. If you are asked for a number, don’t use a comma to write "
SYSTEM_PROMPT += "your number; also don’t use units such as $ or a percent sign unless specified otherwise. If you are asked for a string, don’t use articles, "
SYSTEM_PROMPT += "neither abbreviations (e.g., for cities), and write the digits in plain text unless specified otherwise. If you are asked for a comma-separated "
SYSTEM_PROMPT += "list, apply the above rules depending on whether the element is a number or a string."

DATASET_ID = "gaia-benchmark/GAIA"
SUBSET = "2023_level1"

console = Console()
console.print(f"\nLoading GAIA dataset", style="gold1", highlight=False)

level1_problems = load_dataset(DATASET_ID, SUBSET, split="validation")

console.print(f"Dataset loaded successfully!\n", style="gold1")
print(f"Number of Level 1 problems: {len(level1_problems)}")
print(f"Dataset structure: {level1_problems}")

# Inspecting the first item in the 'validation' split
console.print(f"\nSample data:", style="gold1")
sample = level1_problems[0] 
for key, value in sample.items():
    content_preview = str(value)[:200].replace('\n', ' ')
    print(f"{key}: {content_preview}...")

print()




