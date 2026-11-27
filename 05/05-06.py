import os
import sys
import base64
import shutil
import asyncio
import logging
import pymupdf
import zipfile
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from openai import OpenAI
from rich.console import Console
from datasets import load_dataset
from huggingface_hub import snapshot_download

# Point Python to the folder one level up, named '03' so from p03_05 import format_tool_definition works
import_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../03'))
if import_dir not in sys.path:
    sys.path.insert(0, import_dir)
from p03_09 import search_web


# Point Python to the folder one level up, named '04' so from p03_05 import format_tool_definition works
import_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../04'))
if import_dir not in sys.path:
    sys.path.insert(0, import_dir)
from scratch_agents import tool
from scratch_agents import Agent
from scratch_agents import LlmClient


def reset_workspace():
    """Restore the workspace to its initial state."""
    shutil.rmtree(WORK_DIR, ignore_errors=True)
    shutil.copytree(CACHE_DIR / "2023/validation", WORK_DIR)


# Unzip file tool unzip_file() 
@tool
def unzip_file(zip_path: str, extract_to: str = None) -> str:
    """Extract a zip file to the specified directory."""
    zip_path = Path(zip_path)

    if not zip_path.exists():
        return f"File not found: {zip_path}"

    # Default extraction path: create folder with zip filename
    if extract_to is None:
        extract_to = zip_path.parent / zip_path.stem
    else:
        extract_to = Path(extract_to)

    extract_to.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        file_list = zip_ref.namelist()
        zip_ref.extractall(extract_to)

    # Format results
    result = f"Extracted {len(file_list)} files to {extract_to}/\n\n"
    result += "Contents:\n"
    for f in file_list[:20]:
        result += f"  - {f}\n"
    if len(file_list) > 20:
        result += f"  ... and {len(file_list) - 20} more files\n"

    return result


@tool
def list_files(path: str = ".") -> str:
    """List files and directories in the given path."""
    path = Path(path)

    if not path.exists():
        return f"Path not found: {path}"

    if not path.is_dir():
        return f"Not a directory: {path}"

    items = []
    for item in sorted(path.iterdir()):
        if item.name.startswith('.'):
            continue

        if item.is_dir():
            items.append(f"{item.name}/")
        else:
            items.append(f"{item.name}")

    # Sort directories first
    dirs = [i for i in items if i.endswith('/')]
    files = [i for i in items if not i.endswith('/')]

    result = f"Directory: {path}\n"
    for item in dirs + files:
        result += f"  {item}\n"

    return result


def _read_text_file(file_path: str, start_line: int, end_line: int) -> str:
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Adjust line numbers (1-indexed to 0-indexed)
    start_idx = max(0, start_line - 1)
    end_idx = len(lines) if end_line == -1 else min(end_line, len(lines))

    selected_lines = lines[start_idx:end_idx]

    result = []
    for i, line in enumerate(selected_lines, start=start_line):
        result.append(f"{i:4d} | {line.rstrip()}")
    return '\n'.join(result)


def _read_csv(file_path: str) -> str:
    df = pd.read_csv(file_path)
    return df.to_markdown(index=False)


def _read_excel(file_path: str) -> str:
    df = pd.read_excel(file_path)
    return df.to_markdown(index=False)


@tool
def read_file(file_path: str, start_line: int = 1, end_line: int = -1) -> str:
    """Read file content. Supports txt, py, json, md, csv, xlsx."""
    path = Path(file_path)

    if not path.exists():
        return f"File not found: {file_path}"

    ext = path.suffix.lower()

    if ext in TEXT_EXTENSIONS:
        return _read_text_file(file_path, start_line, end_line)
    elif ext == '.csv':
        return _read_csv(file_path)
    elif ext in SPREADSHEET_EXTENSIONS:
        return _read_excel(file_path)
    else:
        return _read_text_file(file_path, start_line, end_line)


@tool
def read_media_file(file_path: str, query: str) -> str:
    """Analyze an image, audio, or PDF file using LLM."""
    ext = Path(file_path).suffix.lower()

    if ext in IMAGE_EXTENSIONS:
        return _analyze_image(file_path, query)
    elif ext in AUDIO_EXTENSIONS:
        return _analyze_audio(file_path, query)
    elif ext in PDF_EXTENSIONS:
        return _analyze_pdf(file_path, query)
    else:
        return f"Unsupported media format: {ext}"


def _analyze_image(file_path: str, query: str) -> str:
    with open(file_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    ext = Path(file_path).suffix.lower().lstrip('.')
    media_type = "image/jpeg" if ext == "jpg" else f"image/{ext}"

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": query},
                {"type": "image_url", "image_url": {
                    "url": f"data:{media_type};base64,{image_data}"
                }}
            ]
        }]
    )
    return response.choices[0].message.content


def _analyze_audio(file_path: str, query: str) -> str:
    with open(file_path, "rb") as f:
        audio_data = base64.b64encode(f.read()).decode("utf-8")

    ext = Path(file_path).suffix.lower().lstrip('.')

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-audio-preview",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": query},
                {"type": "input_audio", "input_audio": {
                    "data": audio_data,
                    "format": ext
                }}
            ]
        }]
    )
    return response.choices[0].message.content


def _analyze_pdf(file_path: str, query: str) -> str:
    doc = pymupdf.open(file_path)

    # Extract text for context
    text_content = ""
    for page in doc:
        text_content += page.get_text()

    # Convert pages to images
    images = []
    for page in doc[:5]:  # First 5 pages
        pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2))
        img_bytes = pix.tobytes("png")
        images.append(base64.b64encode(img_bytes).decode('utf-8'))

    # Build content with text and images
    content = [{
        "type": "text", 
        "text": f"{query}\n\nExtracted text:\n{text_content[:3000]}"
    }]

    for img_b64 in images:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{img_b64}"}
        })

    vllm_server_fqdn = os.getenv("VLLM_SERVER_FQDN")
    if not vllm_server_fqdn:
        raise ValueError("VLLM_SERVER_FQDN environment variable is not set.")
    vllm_url = f"http://{vllm_server_fqdn}:8000/v1"
    client = OpenAI(base_url=vllm_url, api_key="EMPTY")

    #client = OpenAI()
    response = client.chat.completions.create(model=MODEL_NAME,messages=[{"role": "user", "content": content}])
    return response.choices[0].message.content


MODEL_NAME = "nvidia/Qwen3.6-35B-A3B-NVFP4"
NOTEBOOK_DIR = Path.cwd()
PROJECT_ROOT = NOTEBOOK_DIR.parent
REPOSITORY = "gaia-benchmark/GAIA"
CACHE_DIR = PROJECT_ROOT /"data" / "raw" / "gaia_cache"
WORK_DIR = PROJECT_ROOT / "data" / "interim" / "gaia_workspace"
SPREADSHEET_EXTENSIONS = ['.xlsx', '.xls', '.csv']
IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp']
AUDIO_EXTENSIONS = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.webm']
PDF_EXTENSIONS = ['.pdf']
TEXT_EXTENSIONS = ['.txt', '.py', '.js', '.json', '.md', '.html', 
                   '.css', '.xml', '.yaml', '.yml', '.log', '.sh']


# Async main
async def main():

    # Configure the tools
    tools = [tool(search_web), unzip_file, list_files, read_file, read_media_file]    # search_web() is not defined with @tool decorator so I do it here
    agent = Agent(model=LlmClient(model=MODEL_NAME), tools=tools, instructions="You are a helpful assistant that can search the web and explore files to answer questions.", max_steps=20)
    
    # Silence noisy HTTP request logs from httpx / huggingface_hub / datasets / openai
    logging.getLogger("httpx").setLevel(logging.WARNING)              
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)               
    logging.getLogger("openai._base_client").setLevel(logging.WARNING)  
    
    # 5.12 Load the GAIA dataset with attachments (as in Chapter 2)
    dataset = load_dataset(REPOSITORY, "2023_all", split="validation")
    console = Console()
    console.print(f"\nDownloading {REPOSITORY} to {CACHE_DIR}", style="gold1", highlight=False)
    snapshot_download(repo_id=REPOSITORY, repo_type="dataset", allow_patterns="2023/validation/*", local_dir=CACHE_DIR)

    # Moves past the active tqdm line
    for bar in list(tqdm._instances):
        bar.close()
    print()

    # Listing 5.13 Reset the workspace
    reset_workspace()
    console.print(f"Workspace reset: {WORK_DIR}\n", style="gold1", highlight=False)

    # 5.14 Find problems with file attachments
    console.print(f"Count total problems, how many with attachments and how many are .zip.", style="gold1")
    problems_with_files = [p for p in dataset if p.get('file_name')]
    problem_with_zip = [p for p in problems_with_files if p['file_name'].endswith('.zip')]

    print(f"Total problems: {len(dataset)}")
    print(f"Problems with attachments: {len(problems_with_files)}")
    print(f"Problems with zip files: {len(problem_with_zip)}\n")

    console.print(f"First problem with .zip attachment", style="gold1")
    problem = problem_with_zip[0]
    print(f"Question: {problem['Question'][:100]}...")
    print(f"File name: {problem['file_name']}")

    file_path = WORK_DIR / problem['file_name']
    print(f"File exists: {file_path.exists()}\n")


    # Paragraph 5.4.4 Solving GAIA Zip file problems
    console.print(f"Solving the {len(problem_with_zip)} GAIA problems with Zip files attached", style="gold1", highlight=False)
    # Listing   5.22  Preparing the GAIA zip file problem

    # Select a problem with zip file attachment
    zip_problems = [p for p in problems_with_files if 
    p['file_name'].endswith('.zip')]
    problem = zip_problems[1]

    file_path = WORK_DIR / problem['file_name']

    # Construct prompt including file location
    question = problem['Question'].strip()
    prompt = f"{question} The attached file is located at: {file_path}"
    print(f"Prompt: {prompt}\n")

    # Listing 5.23 Running the agent on the zip file problem
    response = await agent.run(prompt)
    print(response)


# Main -> Run the async main
if __name__ == "__main__":
    asyncio.run(main())