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

CHUNK_SIZE = 100
OVERLAP = 20
sample = "A" * 283

print(f"\nTotal number of characters in the text: {len(sample)}")
print(f"Chunk size: {CHUNK_SIZE}")
print(f"Overlap: {OVERLAP}")

chunks = fixed_length_chunking(sample, chunk_size=CHUNK_SIZE, overlap=OVERLAP)
print(f"Number of chunks: {len(chunks)}\n")


# LLMs have a maximum context window (maximum number of tokens they can process at once)
# In RAG (Retrieval-Augmented Generation):
#  — You chunk your knowledge base
#  - Embed each chunk
#  - Store them in a vector database.
#  The most common use is that, at query time, you retrieve only the most relevant chunks (not the whole database) and pass them to the model
