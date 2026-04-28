"""Vector Search with Re-ranking Tool — two-stage retrieval over Pinecone."""

import os
from typing import Dict, Any, Optional

import aiohttp
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from src.tools.pinecone_helpers import (
    load_pinecone_index,
    generate_embedding,
    query_pinecone,
    format_matches,
)


async def create_vector_search_with_reranking_tool(
    tool_config: Dict[str, Any],
    account_id: int,
    db: Any,
    auth_token: Optional[str] = None,
    **kwargs,
) -> Optional[StructuredTool]:
    setup = load_pinecone_index(tool_config, account_id, db, **kwargs)
    if not setup:
        return None
    index, namespace, index_name = setup

    description = tool_config.get("description", f"Search and rerank the {namespace} knowledge base")
    top_k_default = tool_config.get("topK", 20)
    top_n_default = tool_config.get("topN", 5)

    async def retrieval_with_reranking_impl(
        query: str,
        top_k: int = top_k_default,
        top_n: int = top_n_default,
    ) -> Dict:
        """Retrieve and rerank relevant documents from the knowledge base."""
        try:
            embedding = await generate_embedding(query, auth_token)
            if embedding is None:
                return {"error": "Failed to generate embedding"}

            matches = await query_pinecone(index, embedding, namespace, top_k)
            if not matches:
                return {"results": [], "message": "No relevant documents found", "namespace": namespace, "index": index_name}

            docs = [m.get("metadata", {}).get("content", "") or "No content available" for m in matches]
            similarity_scores = [m.get("score", 0.0) for m in matches]

            reranker_api_url = os.getenv("RERANKER_API_URL")
            if not reranker_api_url:
                return {
                    "results": format_matches(matches[:top_n]),
                    "namespace": namespace,
                    "index": index_name,
                    "reranking_applied": False,
                }

            reranked = await _call_reranker(reranker_api_url, query, docs, auth_token)
            if reranked is None:
                return {
                    "results": format_matches(matches[:top_n]),
                    "namespace": namespace,
                    "index": index_name,
                    "reranking_applied": False,
                }

            formatted = []
            for item in reranked[:top_n]:
                idx = item.get("index")
                if idx is not None and idx < len(matches):
                    entry = format_matches([matches[idx]])[0]
                    entry["score"] = item.get("relevance_score", 0.0)
                    entry["similarity_score"] = similarity_scores[idx]
                    formatted.append(entry)

            return {
                "results": formatted,
                "namespace": namespace,
                "index": index_name,
                "reranking_applied": True,
                "initial_candidates": len(docs),
                "final_results": len(formatted),
            }
        except Exception as exc:
            print(f"[VECTOR SEARCH RERANK] Error: {exc}")
            return {"error": str(exc)}

    class SearchWithRerankQuery(BaseModel):
        query: str = Field(description="The search query to find relevant documents")
        top_k: int = Field(default=top_k_default, description=f"Number of initial candidates to retrieve (default: {top_k_default})")
        top_n: int = Field(default=top_n_default, description=f"Number of final reranked results to return (default: {top_n_default})")

    return StructuredTool(
        func=retrieval_with_reranking_impl,
        coroutine=retrieval_with_reranking_impl,
        name=tool_config.get("name", "vector_search_with_reranking"),
        description=description,
        args_schema=SearchWithRerankQuery,
    )


async def _call_reranker(base_url: str, query: str, documents: list, auth_token: Optional[str]) -> Optional[list]:
    """Call the reranker microservice. Returns the ranked results list or None on failure."""
    headers = {}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"

    endpoint = f"{base_url.rstrip('/')}/huggingface/rerank"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint, json={"query": query, "documents": documents}, headers=headers) as resp:
                if resp.status != 200:
                    print(f"[VECTOR SEARCH RERANK] Reranker API error ({resp.status})")
                    return None
                result = await resp.json()
                return result.get("results", [])
    except Exception as exc:
        print(f"[VECTOR SEARCH RERANK] Reranker call failed: {exc}")
        return None
