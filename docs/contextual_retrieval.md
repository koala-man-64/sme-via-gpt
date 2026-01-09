# Contextual Retrieval Implementation Guide

This document outlines the design and plan for implementing **Contextual Retrieval** in the Local RAG Chatbot. This technique improves retrieval accuracy by preserving context that is often lost during standard chunking.

## Overview
Standard RAG splits documents into chunks (e.g., 800 tokens). If a chunk says "The company's revenue grew by 20%", it loses the context of *which* company or *which* quarter if that information was in a previous chunk.

**Contextual Retrieval** solves this by using an LLM to generate a specific context summary for each chunk (using the full document as reference) and prepending it to the chunk before embedding.

## Implementation Steps

### 1. New Helper Function
Add a function to `app.py` to generate context using a fast/cheap model (e.g., `gpt-4o-mini`).

```python
def generate_chunk_context(document_text: str, chunk_text: str) -> str:
    """
    Generates a concise context summary for a specific chunk using the full document.
    """
    prompt = f"""
    <document>
    {document_text}
    </document>

    <chunk>
    {chunk_text}
    </chunk>

    Please give a short concise context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else.
    """
    
    # Call OpenAI (pseudo-code)
    try:
        client = openai_client()
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Failed to generate context: {e}")
        return ""
```

### 2. Update Indexing Logic
Modify `build_or_update_from_adls` in `app.py`.

*   **Current Flow**: Download -> Chunk -> Embed(Chunk).
*   **New Flow**: Download -> Chunk -> **Generate Context(Chunk)** -> Embed(Context + Chunk).

```python
# Pseudo-code for the loop inside build_or_update_from_adls

for j, chunk in enumerate(chunks):
    # ... existing setup ...
    
    # 1. Generate Context
    context = generate_chunk_context(full_doc_text, chunk)
    
    # 2. Prepare Text for Embedding
    # We embed the COMBINATION, but we might only return the raw chunk to the user (or both).
    text_to_embed = f"{context}\n\n{chunk}"
    
    # 3. Embed
    vector = embed_text(text_to_embed)
    
    # 4. Save to Metadata
    chunk_meta.append({
        "text": chunk,       # Original text for display
        "context": context,  # Saved for debugging/display
        # ... validation fields ...
    })
```

### 3. Caching Updates
The cache validation logic (`_cache_is_compatible`) must be updated.
-   **Why**: If we switch to contextual retrieval, old cached embeddings (which lacked context) are now invalid/inferior.
-   **Fix**: Add a `"contextual": True` flag to the metadata. If the flag is missing or matches the wrong setting, invalidate the cache and re-index.

### 4. Trade-offs
> [!WARNING]
> **Cost**: Indexing becomes significantly more expensive. Every chunk triggers an LLM call.
> **Time**: Indexing will take much longer.

## Future Enhancements
-   **Hybrid Search**: Combine vector search (Contextual) with keyword search (BM25) for best results.
-   **Re-ranking**: Add a re-ranking step (e.g., Cohere) after retrieval to further filter the top K results.
