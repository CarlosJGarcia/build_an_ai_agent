import shutil
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
console.print(f"\nDownloading {REPOSITORY} to {CACHE_DIR}\n", style="gold1", highlight=False)
snapshot_download(repo_id=REPOSITORY, repo_type="dataset", allow_patterns="2023/validation/*", local_dir=CACHE_DIR)

# Listing 5.13 Resetting the workspace function
console.print(f"Workspace reset: {WORK_DIR}\n")
reset_workspace()