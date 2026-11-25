

import shutil
from tqdm import tqdm
from pathlib import Path
from rich.console import Console
from datasets import load_dataset
from huggingface_hub import snapshot_download

def reset_workspace():
    """Restore the workspace to its initial state."""
    shutil.rmtree(WORK_DIR, ignore_errors=True)
    shutil.copytree(CACHE_DIR / "2023/validation", WORK_DIR)


NOTEBOOK_DIR = Path.cwd()
PROJECT_ROOT = NOTEBOOK_DIR.parent
REPOSITORY = "gaia-benchmark/GAIA"
CACHE_DIR = PROJECT_ROOT /"data" / "raw" / "gaia_cache"
WORK_DIR = PROJECT_ROOT / "data" / "interim" / "gaia_workspace"

# 5.12 Load the GAIA dataset with attachments (as in Chapter 2)
dataset = load_dataset(REPOSITORY, "2023_all", split="validation")
console = Console()
console.print(f"\nDownloading {REPOSITORY} to {CACHE_DIR}", style="gold1", highlight=False)
snapshot_download(repo_id=REPOSITORY, repo_type="dataset", allow_patterns="2023/validation/*", local_dir=CACHE_DIR)

# Moves past the active tqdm line
for bar in list(tqdm._instances):
    bar.close()
print()

# Listing 5.13 Resetting the workspace function
reset_workspace()
console.print(f"Workspace reset: {WORK_DIR}\n", style="gold1", highlight=False)

# 5.14 Finding problems with file attachments
console.print(f"Finding problems with file attachments", style="gold1")
problems_with_files = [p for p in dataset if p.get('file_name')]
problem_with_zip = [p for p in problems_with_files if p['file_name'].endswith('.zip')]

print(f"Total problems: {len(dataset)}")
print(f"Problems with attachments: {len(problems_with_files)}")
print(f"Total problems with zip files: {len(problem_with_zip)}")

problem = problem_with_zip[0]
print(f"Question: {problem['Question'][:100]}...")
print(f"File name: {problem['file_name']}")

file_path = WORK_DIR / problem['file_name']
print(f"File exists: {file_path.exists()}\n")