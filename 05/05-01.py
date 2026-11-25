# RAG Retrieval Augmented Generation
# Define the embed() function, using Qwen 3 Embeddings
# Converts any text (max size 32K tokens) into a list of 2.560 floating-point numbers
# Internally, for tokenization, Quen 3 Embedding uses the Hugging Face tokenizers library, instead of OpenAI tiktoken

import os
import numpy as np
from openai import OpenAI
from rich.console import Console

# MODEL_NAME = "qwen3-embedding:4b"
MODEL_NAME = "qwen3-embedding:4b-f16-32k" 

# Read FQDN environment variable and build the Ollama URL
ollama_server_fqdn = os.getenv("OLLAMA_SERVER_FQDN")
if not ollama_server_fqdn:
    raise ValueError("ERROR: OLLAMA_SERVER_FQDN environment variable is not set.")
ollama_url = f"http://{ollama_server_fqdn}:11434/v1"

# Create ollama handle
client = OpenAI(base_url=ollama_url, api_key="ollama")

# Convert text into a numpy enbedding vector, using an embedding model
def embed(text, model=MODEL_NAME):
    """Convert a single string to a numpy embedding vector."""
    response = client.embeddings.create(input=text, model=model)
    return np.array(response.data[0].embedding)

# Comparing embedding similarity
sentence = "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG"
console=Console()
console.print(f"\nRequest to the embedding model {MODEL_NAME}", style="gold1", highlight=False)
embedding = embed(sentence)
print(f"Sentence: {sentence}")
print(f"Embedding: {embedding}")
print(f"Embedding dimensions: {embedding.ndim}")
print(f"Embedding shape: {embedding.shape}")
print(f"Embedding size: {embedding.size}")
print(f"Embedding dtype: {embedding.dtype}\n")

# Embedding is the mathematical format used to capture the semantic meaning of test
# Embedding = Convert a text (single word, phrase, paragraph, book) into a list of floating-point numbers of specific lenght
# When using Qwen 3 and numpy, an embedding is a 1-dimension vector 2560 items long, each item is a float64 number
# Embeddings are too computationally heavy to search at scale without a vector database
# A vector database stores embeddings. It is the specialized engine built to store, index and search that specific format
# Vector databases: FAISS, Chroma, pgvector

# embedding_2560 = [0.0123, -0.0456, 0.0891, ...]  2,560 numbers, single bracket level
# full_vec = np.array(embedding_2560, dtype=np.float64)
# Dimensions: 1D (full_vec.ndim == 1)
# Shape / Items: 2,560 items (full_vec.shape == (2560,))
# Data type: float64 (full_vec.dtype == dtype('float64'))
