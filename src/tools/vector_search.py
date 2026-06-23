"""Vector Search Tool — semantic search over Pinecone."""

import logging
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.tools.pinecone_helpers import (
    format_matches,
    generate_embedding,
    load_pinecone_index,
    query_pinecone,
)

logger = logging.getLogger(__name__)


async def create_vector_search_tool(
    tool_config: dict[str, Any],
    account_id: int,
    db: Session,
    auth_token: str | None = None,
    **kwargs,
) -> StructuredTool | None:
    setup = load_pinecone_index(tool_config, account_id, db, **kwargs)
    if not setup:
        return None
    index, namespace, index_name = setup

    description = tool_config.get("description", f"Search the {namespace} knowledge base")
    top_k_default = tool_config.get("topK", 10)

    async def retrieval_impl(query: str, top_k: int = top_k_default) -> dict:
        """Retrieve relevant documents from the knowledge base."""
        try:
            embedding = await generate_embedding(query, auth_token)
            if embedding is None:
                return {"error": "Failed to generate embedding"}

            matches = await query_pinecone(index, embedding, namespace, top_k)
            if not matches:
                return {"results": [], "message": "No relevant documents found"}

            return {
                "results": format_matches(matches),
                "namespace": namespace,
                "index": index_name,
            }
        except Exception as exc:
            logger.error(f"[VECTOR SEARCH] Error: {exc}")
            return {"error": str(exc)}

    class SearchQuery(BaseModel):
        query: str = Field(description="The search query to find relevant documents")
        top_k: int = Field(
            default=top_k_default,
            description=f"Number of results to return (default: {top_k_default})",
        )

    return StructuredTool(
        func=retrieval_impl,
        coroutine=retrieval_impl,
        name=tool_config.get("name", "vector_search"),
        description=description,
        args_schema=SearchQuery,
    )
