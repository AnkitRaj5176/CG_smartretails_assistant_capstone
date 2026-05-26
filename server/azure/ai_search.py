"""
ai_search.py
────────────
Azure Cognitive Services — AI Search integration.

Used for semantic search over retail policy documents.
Falls back to local VectorStore when Azure Search not configured.

Azure AI Search provides:
  - Full-text search with BM25 ranking
  - Semantic ranking (requires semantic configuration)
  - Hybrid search (keyword + vector)
  - Faceted navigation
"""

import logging
import os
from typing import Optional

from server.env_config import env_settings

logger = logging.getLogger(__name__)

_search_client = None
_index_client = None

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_DOC_FOLDER = os.path.join(_project_root, "raw_docs")


def _get_search_client():
    """Return cached Azure Search client or None."""
    global _search_client
    if _search_client is not None:
        return _search_client
    if not env_settings.USE_AZURE_SEARCH:
        return None
    try:
        from azure.search.documents import SearchClient          # type: ignore
        from azure.core.credentials import AzureKeyCredential   # type: ignore
        _search_client = SearchClient(
            endpoint=env_settings.AZURE_SEARCH_ENDPOINT,
            index_name=env_settings.AZURE_SEARCH_INDEX,
            credential=AzureKeyCredential(env_settings.AZURE_SEARCH_KEY),
        )
        logger.info("Azure AI Search client initialised (index=%s).", env_settings.AZURE_SEARCH_INDEX)
        return _search_client
    except ImportError:
        logger.warning("azure-search-documents not installed.")
        return None
    except Exception as e:
        logger.warning("Azure AI Search init failed: %s", e)
        return None


def _get_index_client():
    """Return cached Azure Search Index client for index management."""
    global _index_client
    if _index_client is not None:
        return _index_client
    if not env_settings.USE_AZURE_SEARCH:
        return None
    try:
        from azure.search.documents.indexes import SearchIndexClient  # type: ignore
        from azure.core.credentials import AzureKeyCredential         # type: ignore
        _index_client = SearchIndexClient(
            endpoint=env_settings.AZURE_SEARCH_ENDPOINT,
            credential=AzureKeyCredential(env_settings.AZURE_SEARCH_KEY),
        )
        return _index_client
    except Exception as e:
        logger.warning("Azure Search Index client init failed: %s", e)
        return None


# ── Index Management ───────────────────────────────────────────────────────────

def create_policy_index() -> dict:
    """
    Create the retail policy search index in Azure AI Search.
    Schema: id, content, source, chunk_id
    """
    client = _get_index_client()
    if not client:
        return {"status": "skipped", "reason": "Azure AI Search not configured"}

    try:
        from azure.search.documents.indexes.models import (  # type: ignore
            SearchIndex, SimpleField, SearchableField,
            SearchFieldDataType,
        )
        index = SearchIndex(
            name=env_settings.AZURE_SEARCH_INDEX,
            fields=[
                SimpleField(name="id",       type=SearchFieldDataType.String, key=True),
                SearchableField(name="content", type=SearchFieldDataType.String),
                SimpleField(name="source",   type=SearchFieldDataType.String, filterable=True),
                SimpleField(name="chunk_id", type=SearchFieldDataType.String),
            ],
        )
        client.create_or_update_index(index)
        logger.info("Azure AI Search index '%s' created/updated.", env_settings.AZURE_SEARCH_INDEX)
        return {"status": "created", "index": env_settings.AZURE_SEARCH_INDEX}
    except Exception as e:
        logger.warning("Index creation failed: %s", e)
        return {"status": "error", "detail": str(e)}


def upload_policy_documents() -> dict:
    """
    Load policy .txt files and upload them to Azure AI Search index.
    """
    client = _get_search_client()
    if not client:
        return {"status": "skipped", "reason": "Azure AI Search not configured"}

    try:
        documents = []
        doc_id = 0
        for file_name in sorted(os.listdir(_DOC_FOLDER)):
            if not file_name.endswith(".txt"):
                continue
            file_path = os.path.join(_DOC_FOLDER, file_name)
            with open(file_path, "r", encoding="utf-8") as f:
                raw_text = f.read()
            # Chunk into ~500-char pieces
            chunk_size, overlap = 500, 50
            start = 0
            while start < len(raw_text):
                chunk = raw_text[start: start + chunk_size].strip()
                if chunk:
                    documents.append({
                        "id":       str(doc_id),
                        "content":  chunk,
                        "source":   file_name,
                        "chunk_id": f"{file_name}_{doc_id}",
                    })
                    doc_id += 1
                start += chunk_size - overlap

        if documents:
            client.upload_documents(documents)
            logger.info("Uploaded %d documents to Azure AI Search.", len(documents))
            return {"status": "uploaded", "document_count": len(documents)}
        return {"status": "no_documents"}
    except Exception as e:
        logger.warning("Document upload failed: %s", e)
        return {"status": "error", "detail": str(e)}


# ── Search ─────────────────────────────────────────────────────────────────────

def search_policy(query: str, top_k: int = 3) -> dict:
    """
    Search policy documents using Azure AI Search.
    Falls back to local VectorStore if Azure not configured.
    """
    client = _get_search_client()

    if client:
        try:
            results = client.search(
                search_text=query,
                top=top_k,
                select=["content", "source", "chunk_id"],
            )
            hits = []
            for r in results:
                hits.append({
                    "content":  r["content"],
                    "source":   r["source"],
                    "chunk_id": r["chunk_id"],
                    "score":    r.get("@search.score", 0.0),
                })
            answer = hits[0]["content"] if hits else "No results found."
            sources = list({h["source"] for h in hits})
            logger.info("Azure AI Search: %d results for '%s'", len(hits), query[:50])
            return {
                "query": query,
                "answer": answer,
                "sources": sources,
                "chunks": hits,
                "service": "azure_ai_search",
            }
        except Exception as e:
            logger.warning("Azure AI Search query failed: %s", e)

    # ── Local VectorStore fallback ─────────────────────────────────────────────
    from server.retrieval.doc_retriever import retrieve_policy_answer
    result = retrieve_policy_answer(query, top_k=top_k)
    result["service"] = "local_vector_store"
    return result
