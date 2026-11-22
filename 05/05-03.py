import os
import numpy as np
from openai import OpenAI
from rich.console import Console

MODEL_NAME = "qwen3-embedding:4b"

# Read FQDN environment variable and build the Ollama URL for the model
ollama_server_fqdn = os.getenv("OLLAMA_SERVER_FQDN")
if not ollama_server_fqdn:
    raise ValueError("ERROR: OLLAMA_SERVER_FQDN environment variable is not set.")
ollama_url = f"http://{ollama_server_fqdn}:11434/v1"


client = OpenAI(base_url=ollama_url, api_key="ollama")

# get_embeddings() Converts text into embedding vectors using an embedding model
# The book uses OpenAI "text-embedding-3-small" (1,536 dimensions). Requires an OpenAI account with credits
# I use "qwen3-embedding:4b" (2,560 dimensions natively, but can be configured for 1,536 if needed). Open-weights (Apache 2.0 license) on ollama
def get_embeddings(texts, model=MODEL_NAME):
    """Convert text to embedding vectors."""
    if isinstance(texts, str):
        texts = [texts]                                              # Wraps a single string in list for uniform processing

    response = client.embeddings.create(input=texts, model=model)    # Calls the OpenAI embedding API
    return np.array([item.embedding for item in response.data])      # Extracts embedding vectors as a numpy array


# Calculate cosine similarity matrix between 2D arrays a and b (L2-normalized vectors) using NumPy instead of Scikit-Learn
def cosine_similarity(a, b):
    """Compute cosine similarity matrix between 2D arrays a and b (assumes L2-normalized vectors)."""
    return a @ b.T


def vector_search(query, chunks, chunk_embeddings, top_k=3):
    """Find the most similar chunks to the query."""
    console.print("\nEmbedding request to the model", style="gold1", highlight=False)
    query_embedding = get_embeddings(query)
    similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
    top_indices = similarities.argsort()[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            'chunk': chunks[idx],
            'similarity': similarities[idx]
        })
    return results


# Test vector search
documents = [
    "Python is a programming language",
    "Machine learning uses Python extensively", 
    "Cats are popular pets",
    "Deep learning is a subset of machine learning"
]

console = Console()
doc_embeddings = get_embeddings(documents)

query = "Artificial Intelligence"
results = vector_search(query, documents, doc_embeddings, top_k=4)
console.print(f'Similarity between query "{query}" and documents:', style="gold1", highlight=False)

for r in results:
    print(f"{r['similarity']:.3f}: {r['chunk']}")
print()