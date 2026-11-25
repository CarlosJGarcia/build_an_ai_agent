# This code ilustrates why vector databases are necessary


import os
import tiktoken
import numpy as np
from openai import OpenAI
from tavily import TavilyClient


# fixed_length_chunking() Splits text into 'chunks'
def fixed_length_chunking(text, chunk_size, overlap):
    return

# get_embeddings() Converts texts into embedding vectors using an embedding model
def get_embeddings(texts):
    MODEL_NAME = "qwen3-embedding:4b-f16-32k" 
    return np.array([item.embedding for item in response.data])      # Extracts embedding vectors as a numpy array

# vector_search() Using embedding and cosine similarity sorts texts for similarity to another one
def vector_search(query, chunk_texts, chunk_embeddings, top_k):
    return


# Step 1: Prepare web search results
tavily = TavilyClient()
response = tavily.search("2025 Nobel Prize winners", max_results=10, include_raw_content=True)

search_results = []
for result in response['results']:
    if result.get('raw_content'):
        search_results.append({'title': result['title'],'content': result['raw_content'], 'url': result['url']})


# Step 2: Check token count
enc = tiktoken.encoding_for_model("gpt-5")
full_text = "\n\n".join([f"Title: {r['title']}\n{r['content']}" for r in search_results])
total_tokens = len(enc.encode(full_text))
print(f"\nTotal characters: {len(full_text):,}")
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

print(f"Total chunks: {len(all_chunks)}")

chunk_texts = [c['text'] for c in all_chunks]
chunk_embeddings = get_embeddings(chunk_texts)


# Step 4: Execute vector search
query = "quantum computing"
results = vector_search(query, chunk_texts, chunk_embeddings, top_k=3)

print(f"Query: '{query}'\n")
print("=" * 60)
for i, r in enumerate(results, 1):
    print(f"\n[{i}] Similarity: {r['similarity']:.3f}")
    print(f"{r['chunk'][:300]}...")


# Step 5: Measure token savings
top_chunks = [r['chunk'] for r in results]
selected_text = "\n\n".join(top_chunks)
selected_tokens = len(enc.encode(selected_text))

print(f"Total tokens: {total_tokens}")
print(f"Selected tokens: {selected_tokens}")
print(f"Savings rate: {(1 - selected_tokens/total_tokens)*100:.1f}%")