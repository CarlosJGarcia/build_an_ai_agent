# This code ilustrates why vector databases are necessary
# Shows that:
# - Dumping 70K raw tokens into an LLM prompt is expensive (in number of tokens) and noisy
# - Vector search filters it down to the ~3 most relevant chunks)


import os
import tiktoken
import numpy as np
from openai import OpenAI
from tavily import TavilyClient
from rich.console import Console


# fixed_length_chunking() Splits text into 'chunks'
# From 05-03
def fixed_length_chunking(text, chunk_size=500, overlap=50):
    """Split text into fixed-length chunks with overlap"""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap >= chunk_size:
        raise ValueError("overlap must be less than chunk_size to avoid infinite loop")

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap if end < len(text) else end  #1

    return chunks


# get_embeddings() Converts texts into embedding vectors using an embedding model
def get_embeddings(texts, model_name):
    """Convert text to embedding vectors."""
    
    

    # Read FQDN environment variable and build the Ollama URL for the model
    ollama_server_fqdn = os.getenv("OLLAMA_SERVER_FQDN")
    if not ollama_server_fqdn:
        raise ValueError("ERROR: OLLAMA_SERVER_FQDN environment variable is not set.")
    ollama_url = f"http://{ollama_server_fqdn}:11434/v1"

    client = OpenAI(base_url=ollama_url, api_key="ollama")
    
    if isinstance(texts, str):
        texts = [texts]                                                  # Wraps a single string in list for uniform processing

    response = client.embeddings.create(input=texts, model=model_name)   # Calls the OpenAI embedding API
    return np.array([item.embedding for item in response.data])          # Extracts embedding vectors as a numpy array


# Calculate cosine similarity matrix between 2D arrays a and b (L2-normalized vectors) using NumPy instead of Scikit-Learn
# From 05-04
def cosine_similarity(a, b):
    """Compute cosine similarity matrix between 2D arrays a and b (assumes L2-normalized vectors)."""
    return a @ b.T


# vector_search() Using embedding and cosine similarity sorts texts for similarity to another one
def vector_search(query, model_name, chunks, chunk_embeddings, top_k=3):
    """Find the most similar chunks to the query."""
    query_embedding = get_embeddings(query, model_name)
    similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
    top_indices = similarities.argsort()[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            'chunk': chunks[idx],
            'similarity': similarities[idx]
        })
    return results


MODEL_NAME = "qwen3-embedding:4b-f16-32k"
QUERY = "Quantum computing" 


# Main
console = Console()

# Step 1: Raw web search. The response will have a large number of tokens (+70.000) and is noisy as it includes HTML and encoded images.
tavily = TavilyClient()
response = tavily.search(QUERY, max_results=10, include_raw_content=True)

search_results = []
for result in response['results']:
    if result.get('raw_content'):
        search_results.append({'title': result['title'],'content': result['raw_content'], 'url': result['url']})

console.print("\nStep 1: Raw web search (Tavily)", style="gold1", highlight=False)
print(f"Query: '{QUERY}'\n")
print(f"Search_results: {search_results}\n")


# Step 2: Check token count
enc = tiktoken.encoding_for_model("gpt-5")
full_text = "\n\n".join([f"Title: {r['title']}\n{r['content']}" for r in search_results])
total_tokens = len(enc.encode(full_text))

console.print("Step 2: Token count (tiktoken)", style="gold1", highlight=False)
print(f"Total characters: {len(full_text):,}")
print(f"Total tokens: {total_tokens:,}\n")


# Step 3: Perform chunking and embedding
all_chunks = []
for result in search_results:
    text = f"Title: {result['title']}\n{result['content']}"
    chunks = fixed_length_chunking(text, chunk_size=500, overlap=50)
    for chunk in chunks:
        all_chunks.append({
            'text': chunk,
            'title': result['title'],
            'url': result['url']
        })

console.print(f"Step 3: Chunking and embedding ({MODEL_NAME})", style="gold1", highlight=False)
print(f"Total chunks: {len(all_chunks)}\n")

chunk_texts = [c['text'] for c in all_chunks]
chunk_embeddings = get_embeddings(chunk_texts, MODEL_NAME)


# Step 4: Execute vector search
results = vector_search(QUERY, MODEL_NAME, chunk_texts, chunk_embeddings, top_k=3)

console.print(F"Step 4: Vector search ({MODEL_NAME})", style="gold1", highlight=False)
print(f"Query: '{QUERY}'\n")
#print("=" * 60)
print("Search results:")
for i, r in enumerate(results, 1):
    print(f"[{i}] Similarity: {r['similarity']:.3f}")
    print(f"{r['chunk'][:300]}...\n")


# Step 5: Measure token savings
top_chunks = [r['chunk'] for r in results]
selected_text = "\n\n".join(top_chunks)
selected_tokens = len(enc.encode(selected_text))

console.print("Step 5: Measure savings", style="gold1", highlight=False)
print(f"Total tokens: {total_tokens}")
print(f"Selected tokens: {selected_tokens}")
print(f"Savings rate: {(1 - selected_tokens/total_tokens)*100:.1f}%\n")