"""
policy_agent.py
───────────────
Document Assistant Agent — searches retail policy knowledge base
using the VectorStore (embeddings + cosine similarity RAG)
and generates grounded answers via Azure OpenAI.
"""

import logging

from server.genai.llm_client import call_llm
from server.retrieval.doc_retriever import retrieve_policy_answer

logger = logging.getLogger(__name__)

POLICY_SYSTEM_PROMPT = """You are a retail policy compliance assistant.
Your ONLY source of truth is the context provided below from the policy documents.
Rules you MUST follow:
- Answer ONLY using the provided context. Never use outside knowledge.
- If the answer is not in the context, say exactly:
  "This information is not available in the policy documents."
- Quote relevant policy sections when possible.
- Be concise and precise — policy answers must be unambiguous.
- Do NOT add disclaimers or suggest consulting a lawyer."""


def run_policy_agent(user_message: str) -> dict:
    """
    Execute the Policy Agent pipeline:
    1. Retrieve top-3 relevant policy chunks via VectorStore RAG.
    2. Pass full context + user question to Azure OpenAI for grounded answer.
    3. Return structured response with sources and chunk scores.
    """
    # Step 1 — Vector Store RAG retrieval
    retrieval_result = retrieve_policy_answer(user_message, top_k=3)
    full_context = retrieval_result.get("full_context") or retrieval_result.get("answer", "")
    sources = retrieval_result.get("sources", [])
    chunks = retrieval_result.get("chunks", [])

    logger.info(
        "PolicyAgent RAG: %d chunks retrieved, sources=%s",
        len(chunks), sources,
    )

    # Step 2 — grounded LLM answer using full context (all top-3 chunks)
    llm_response = call_llm(
        system_prompt=POLICY_SYSTEM_PROMPT,
        user_message=user_message,
        context=full_context,
        max_tokens=400,
        temperature=0.0,  # deterministic for policy answers
    )

    return {
        "agent": "PolicyAgent",
        "agent_description": "Retail policy and compliance document specialist",
        "tool_used": "vector_store_rag",
        "raw_data": full_context,
        "sources": sources,
        "chunks": chunks,
        "response": llm_response,
    }
