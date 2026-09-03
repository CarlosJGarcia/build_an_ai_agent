from datasets import load_dataset

level1_problems = load_dataset("gaia-benchmark/GAIA", "2023_level1", split="validation")
print(f"Number of Level 1 problems: {len(level1_problems)}")