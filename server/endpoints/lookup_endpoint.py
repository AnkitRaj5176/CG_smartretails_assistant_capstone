import logging

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from server.retrieval.doc_retriever import retrieve_policy_answer

logger = logging.getLogger(__name__)

lookup_router = APIRouter(tags=["C. GenAI & Multi-Agent System"])


class PolicySearchRequest(BaseModel):
    """Request body for policy document search."""

    search_query: str


@lookup_router.post("/api/docs/search", status_code=status.HTTP_200_OK)
async def search_policy_documents(request_body: PolicySearchRequest) -> dict:
    """Search retail policy documents and return a grounded answer."""
    if not request_body.search_query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="search_query must not be empty.",
        )

    try:
        retrieval_result = retrieve_policy_answer(request_body.search_query)
    except Exception as retrieval_error:
        logger.warning("Policy retrieval failed: %s", retrieval_error)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retrieval failed: {retrieval_error}",
        )

    return retrieval_result
