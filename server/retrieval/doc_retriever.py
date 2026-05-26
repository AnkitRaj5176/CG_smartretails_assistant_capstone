"""
doc_retriever.py
────────────────
RAG retrieval layer — uses the VectorStore for semantic search over
retail policy documents.

Retrieval pipeline:
  Query → VectorStore.search() → Top-K chunks → context string
"""

import logging

from server.retrieval.vector_store import get_vector_store

logger = logging.getLogger(__name__)

GROUNDED_SYSTEM_PROMPT: str = (
    "Answer ONLY using the provided context. "
    "Do not use any outside knowledge or make assumptions. "
    "If the answer is not in the context, say: "
    "'This information is not available in the policy documents.'"
)


def retrieve_policy_answer(search_query: str, top_k: int = 3) -> dict:
    """
    Retrieve the most relevant policy chunks for the query.

    Uses the VectorStore (TF-IDF embeddings + cosine similarity).
    Returns a dict with query, answer (top chunk text), sources, and all chunks.
    """
    store = get_vector_store()

    if not store.is_ready or store.chunk_count == 0:
        return {
            "query": search_query,
            "answer": "No policy documents are available.",
            "sources": [],
            "chunks": [],
        }

    results = store.search(search_query, top_k=top_k)

    if not results:
        return {
            "query": search_query,
            "answer": "No relevant content found.",
            "sources": [],
            "chunks": [],
        }

    # Top chunk as primary answer
    answer_text = results[0]["text"]

    # Unique sources
    source_list = list({r["source"] for r in results})

    # Full context (all top-K chunks concatenated)
    full_context = store.get_context(search_query, top_k=top_k)

    logger.info(
        "RAG retrieval: query='%s' → %d chunks, top_score=%.3f",
        search_query[:60], len(results), results[0]["score"],
    )

    return {
        "query": search_query,
        "answer": answer_text,
        "full_context": full_context,
        "sources": source_list,
        "chunks": [
            {"text": r["text"], "source": r["source"], "score": r["score"]}
            for r in results
        ],
    }
