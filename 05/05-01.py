# RAG Retrieval Augmented Generation
# Implementing the get_embeddings() function

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

# get_embeddings() Converts text into enbedding vectors using an embedding model
# The book uses OpenAI "text-embedding-3-small" (1,536 dimensions). Requires an OpenAI account with credits
# I use "qwen3-embedding:4b" (2,560 dimensions natively, but can be configured for 1,536 if needed). Open-weights (Apache 2.0 license) on ollama
def get_embeddings(texts, model=MODEL_NAME):
    """Convert text to embedding vectors."""
    if isinstance(texts, str):
        texts = [texts]                                              # Wraps a single string in list for uniform processing

    response = client.embeddings.create(input=texts, model=model)    # Calls the OpenAI embedding API
    return np.array([item.embedding for item in response.data])      # Extracts embedding vectors as a numpy array

# Comparing embedding similarity
sentences = ["The cat is sleeping on the couch", "A kitten is playing with a toy", "The dog is running in the park"]
console=Console()
console.print("\nEmbedding request to the model", style="gold1", highlight=False)
embeddings = get_embeddings(sentences)

# Since OpenAI and Qwen embeddings are already normalized to unit length, the dot product (@) is the cosine similarity
cat_kitten = embeddings[0] @ embeddings[1] # Calculates the cosine similarity between embedding vectors
cat_dog = embeddings[0] @ embeddings[2]    # Calculates the cosine similarity between embedding vectors

print(f"Cat vs Kitten: {cat_kitten:.3f}")  # Expected 0.3 - 0.5
print(f"Cat vs Dog: {cat_dog:.3f}")        # Expected 0.2

# Check ordering (True = ranked in the expected order)
ranked_ok = cat_kitten > cat_dog  # 0.546 > 0.297 -> True
print(f"Are ranked correctly: {ranked_ok}")

# Measure the relative gap or ratio
print(f"Diff: {(cat_kitten - cat_dog):.3f}")  # Expected+0.249
print(f"Ratio: {(cat_kitten / cat_dog):.3f}\n") # Expected 1.84x more similar
