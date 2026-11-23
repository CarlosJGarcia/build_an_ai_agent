import tiktoken
from tavily import TavilyClient

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
