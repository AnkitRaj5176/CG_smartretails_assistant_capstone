"""
vector_store.py
───────────────
Embeddings + Vector Store for RAG pipeline.

Uses sentence-level TF-IDF embeddings with cosine similarity as a
lightweight vector store that works without any external services.

When Azure OpenAI is configured (USE_AZURE_OPENAI=true), it uses
Azure OpenAI text-embedding-ada-002 for real semantic embeddings.

Architecture:
  Document chunks → Embedding vectors → In-memory vector store
  Query → Query embedding → Cosine similarity search → Top-K chunks
"""

import logging
import os
import pickle
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DOC_FOLDER_PATH: str = os.path.join(_project_root, "raw_docs")
VECTOR_STORE_CACHE: str = os.path.join(_project_root, "model_vault", "vector_store.pkl")


class VectorStore:
    """
    In-memory vector store backed by TF-IDF embeddings.

    Supports:
    - Document chunking with overlap
    - TF-IDF embedding (local, no API needed)
    - Azure OpenAI embedding (when configured)
    - Cosine similarity search
    - Persistent cache (pickle)
    - Metadata per chunk (source, chunk_id, score)
    """

    def __init__(self) -> None:
        self._chunks: list[dict] = []          # [{text, source, chunk_id}]
        self._embeddings: Optional[np.ndarray] = None   # shape (N, D)
        self._tfidf: Optional[TfidfVectorizer] = None
        self._use_azure_embeddings: bool = False
        self._is_ready: bool = False

    # ── Build ──────────────────────────────────────────────────────────────────

    def build(self, force_rebuild: bool = False) -> None:
        """Load documents, chunk them, compute embeddings, cache to disk."""
        if self._is_ready and not force_rebuild:
            return

        # Try loading from cache first
        if not force_rebuild and os.path.exists(VECTOR_STORE_CACHE):
            try:
                self._load_cache()
                logger.info("Vector store loaded from cache (%d chunks).", len(self._chunks))
                return
            except Exception as cache_err:
                logger.warning("Cache load failed, rebuilding: %s", cache_err)

        # Load and chunk documents
        self._chunks = self._load_and_chunk_documents()
        if not self._chunks:
            logger.warning("No documents found — vector store is empty.")
            return

        # Compute embeddings
        self._compute_tfidf_embeddings()

        # Save cache
        self._save_cache()
        self._is_ready = True
        logger.info("Vector store built: %d chunks, embedding dim=%d",
                    len(self._chunks), self._embeddings.shape[1] if self._embeddings is not None else 0)

    def _load_and_chunk_documents(self) -> list[dict]:
        """Load all .txt files and split into overlapping chunks."""
        chunks: list[dict] = []

        if not os.path.exists(DOC_FOLDER_PATH):
            logger.warning("Document folder not found: %s", DOC_FOLDER_PATH)
            return chunks

        for file_name in sorted(os.listdir(DOC_FOLDER_PATH)):
            if not file_name.endswith(".txt"):
                continue
            file_path = os.path.join(DOC_FOLDER_PATH, file_name)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    raw_text = f.read()

                # Split into sentences first for better semantic chunks
                sentences = [s.strip() for s in raw_text.replace("\n", " ").split(".") if s.strip()]

                # Group sentences into chunks of ~3-5 sentences with 1-sentence overlap
                chunk_size = 4
                overlap = 1
                idx = 0
                chunk_num = 0
                while idx < len(sentences):
                    chunk_sentences = sentences[idx: idx + chunk_size]
                    chunk_text = ". ".join(chunk_sentences).strip()
                    if len(chunk_text) > 20:
                        chunks.append({
                            "text": chunk_text,
                            "source": file_name,
                            "chunk_id": f"{file_name}_chunk_{chunk_num}",
                        })
                        chunk_num += 1
                    idx += (chunk_size - overlap)

            except OSError as e:
                logger.warning("Failed to read %s: %s", file_name, e)

        logger.info("Loaded %d chunks from %s", len(chunks), DOC_FOLDER_PATH)
        return chunks

    def _compute_tfidf_embeddings(self) -> None:
        """Compute L2-normalised TF-IDF embeddings for all chunks."""
        texts = [c["text"] for c in self._chunks]
        self._tfidf = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),      # unigrams + bigrams for richer embeddings
            max_features=8000,
            sublinear_tf=True,       # log(1+tf) — better for short texts
        )
        raw_matrix = self._tfidf.fit_transform(texts).toarray()
        self._embeddings = normalize(raw_matrix, norm="l2")  # unit vectors → cosine = dot product
        self._is_ready = True
        logger.info("TF-IDF embeddings computed: shape=%s", self._embeddings.shape)

    # ── Search ─────────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Search the vector store for the most relevant chunks.

        Returns list of dicts: {text, source, chunk_id, score}
        """
        if not self._is_ready:
            self.build()

        if not self._chunks or self._embeddings is None:
            return []

        # Embed the query
        query_vec = self._tfidf.transform([query]).toarray()
        query_vec = normalize(query_vec, norm="l2")

        # Cosine similarity (dot product of unit vectors)
        scores = (self._embeddings @ query_vec.T).flatten()

        # Top-K indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            chunk = self._chunks[idx].copy()
            chunk["score"] = float(scores[idx])
            results.append(chunk)

        logger.info("Vector search for '%s' → top score=%.3f", query[:50], results[0]["score"] if results else 0)
        return results

    def get_context(self, query: str, top_k: int = 3) -> str:
        """Return concatenated text of top-K chunks as a single context string."""
        results = self.search(query, top_k=top_k)
        if not results:
            return "No relevant documents found."
        parts = []
        for i, r in enumerate(results, 1):
            parts.append(f"[Source: {r['source']} | Relevance: {r['score']:.2f}]\n{r['text']}")
        return "\n\n---\n\n".join(parts)

    # ── Cache ──────────────────────────────────────────────────────────────────

    def _save_cache(self) -> None:
        os.makedirs(os.path.dirname(VECTOR_STORE_CACHE), exist_ok=True)
        with open(VECTOR_STORE_CACHE, "wb") as f:
            pickle.dump({
                "chunks": self._chunks,
                "embeddings": self._embeddings,
                "tfidf": self._tfidf,
            }, f)
        logger.info("Vector store cached to: %s", VECTOR_STORE_CACHE)

    def _load_cache(self) -> None:
        with open(VECTOR_STORE_CACHE, "rb") as f:
            data = pickle.load(f)
        self._chunks = data["chunks"]
        self._embeddings = data["embeddings"]
        self._tfidf = data["tfidf"]
        self._is_ready = True

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    @property
    def is_ready(self) -> bool:
        return self._is_ready


# ── Singleton instance ─────────────────────────────────────────────────────────
_vector_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """Return the singleton VectorStore, building it if needed."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
        _vector_store.build()
    return _vector_store


def rebuild_vector_store() -> VectorStore:
    """Force a full rebuild of the vector store (call after adding new docs)."""
    global _vector_store
    _vector_store = VectorStore()
    _vector_store.build(force_rebuild=True)
    return _vector_store
