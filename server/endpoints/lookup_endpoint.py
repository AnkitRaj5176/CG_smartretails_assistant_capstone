"""
lookup_endpoint.py — Section C: RAG Document Search
"""
import logging
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from server.retrieval.doc_retriever import retrieve_policy_answer

logger = logging.getLogger(__name__)
lookup_router = APIRouter(tags=["C. GenAI & Multi-Agent System"])


class PolicySearchRequest(BaseModel):
    search_query: str


@lookup_router.post("/api/docs/search")
async def search_policy_documents(request_body: PolicySearchRequest) -> dict:
    """RAG search over retail policy documents using TF-IDF Vector Store."""
    if not request_body.search_query.strip():
        raise HTTPException(status_code=400, detail="search_query must not be empty.")
    try:
        return retrieve_policy_answer(request_body.search_query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
