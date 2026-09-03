# Load the GAIA (General AI Assistants) dataset from HuggingFace

from pydantic import BaseModel
from rich.console import Console
from datasets import load_dataset

# Define data structure (named GaiaOutput) using pydantic 
class GaiaOutput(BaseModel):
    is_solvable: bool
    unsolvable_reason: str = ""
    final_answer: str = ""

SYSTEM_PROMPT = "You are a general AI assistant. I will ask you a question. First, determine if you can solve this problem with your current capabilities "
SYSTEM_PROMPT += "and set “is_solvable” accordingly. If you can solve it, set “is_solvable” to true and provide your answer in “final_answer”. "
SYSTEM_PROMPT += "If you cannot solve it, set “is_solvable” to false and explain why in “unsolvable_reason”. Your final answer should be a number OR "
SYSTEM_PROMPT += "as few words as possible OR a comma-separated list of numbers and/or strings. If you are asked for a number, don’t use a comma to write "
SYSTEM_PROMPT += "your number; also don’t use units such as $ or a percent sign unless specified otherwise. If you are asked for a string, don’t use articles, "
SYSTEM_PROMPT += "neither abbreviations (e.g., for cities), and write the digits in plain text unless specified otherwise. If you are asked for a comma-separated "
SYSTEM_PROMPT += "list, apply the above rules depending on whether the element is a number or a string."

console = Console()
console.print(f"\nLoading GAIA dataset", style="gold1", highlight=False)

level1_problems = load_dataset("gaia-benchmark/GAIA", "2023_level1", split="validation")
print(f"Number of Level 1 problems: {len(level1_problems)}")
print()

print(SYSTEM_PROMPT)
print()



