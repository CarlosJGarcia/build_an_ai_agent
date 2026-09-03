# Load the GAIA (General AI Assistants) dataset from HuggingFace

from rich.console import Console
from datasets import load_dataset

console = Console()
console.print(f"\nLoading GAIA dataset", style="gold1", highlight=False)

level1_problems = load_dataset("gaia-benchmark/GAIA", "2023_level1", split="validation")
print(f"Number of Level 1 problems: {len(level1_problems)}")
print()