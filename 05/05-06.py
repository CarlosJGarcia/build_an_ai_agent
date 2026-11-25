from pathlib import Path
from rich.console import Console
from datasets import load_dataset
from huggingface_hub import snapshot_download


# Load dataset (same as Chapter 2)
dataset = load_dataset("gaia-benchmark/GAIA", "2023_all", split="validation")

# Download attached files
NOTEBOOK_DIR = Path.cwd()
PROJECT_ROOT = NOTEBOOK_DIR.parent
REPOSITORY = "gaia-benchmark/GAIA"
CACHE_DIR = PROJECT_ROOT /"data" / "raw" / "gaia_cache"

console = Console()
console.print(f"\nDownloading {REPOSITORY} to {CACHE_DIR}\n", style="gold1", highlight=False)
snapshot_download(repo_id=REPOSITORY, repo_type="dataset", allow_patterns="2023/validation/*", local_dir=CACHE_DIR)