# RAG Retrieval Augmented Generation
# Implementing the get_embeddings() function

import numpy as np
from openai import OpenAI


client = OpenAI()

def get_embeddings(texts, model="text-embedding-3-small"):
    """Convert text to embedding vectors."""
    if isinstance(texts, str):
        texts = [texts]                                              # Wraps a single string in list for uniform processing

    response = client.embeddings.create(input=texts, model=model)    # Calls the OpenAI embedding API
    return np.array([item.embedding for item in response.data])      # Extracts embedding vectors as a numpy array

print("\nTodo bien.\n")