"""Search API schemas."""

from __future__ import annotations

from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    """Schema for search request."""

    query_text: Optional[str] = None
    query_embedding: Optional[list[float]] = None
    k: int = Field(default=10, ge=1, le=100)
    filters: dict[str, Any] = Field(default_factory=dict)

    class Config:
        """Pydantic config."""

        json_schema_extra = {
            "example": {
                "query_text": "What is machine learning?",
                "k": 5,
                "filters": {"author": "John Doe"},
            }
        }


class SearchResult(BaseModel):
    """Schema for a single search result."""

    chunk_id: UUID
    document_id: UUID
    content: str
    score: float
    metadata: dict[str, Any]


class SearchResponse(BaseModel):
    """Schema for search response."""

    results: list[SearchResult]
    total: int
    query_time_ms: float

    class Config:
        """Pydantic config."""

        json_schema_extra = {
            "example": {
                "results": [
                    {
                        "chunk_id": "123e4567-e89b-12d3-a456-426614174002",
                        "document_id": "123e4567-e89b-12d3-a456-426614174001",
                        "content": "Machine learning is...",
                        "score": 0.95,
                        "metadata": {"page": 1},
                    }
                ],
                "total": 1,
                "query_time_ms": 15.5,
            }
        }
