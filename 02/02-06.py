# Load the GAIA (General AI Assistants) dataset from HuggingFace

from pydantic import BaseModel
from rich.console import Console
from datasets import load_dataset

# Define data structure (named GaiaOutput) using pydantic 
class GaiaOutput(BaseModel):
    is_solvable: bool
    unsolvable_reason: str = ""
    final_answer: str = ""

console = Console()
console.print(f"\nLoading GAIA dataset", style="gold1", highlight=False)

level1_problems = load_dataset("gaia-benchmark/GAIA", "2023_level1", split="validation")
print(f"Number of Level 1 problems: {len(level1_problems)}")
print()



