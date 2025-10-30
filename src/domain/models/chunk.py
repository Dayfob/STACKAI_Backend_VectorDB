"""Chunk domain model."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """Chunk domain model.

    A chunk represents a piece of text with its embedding vector.
    """

    id: UUID = Field(default_factory=uuid4)
    content: str = Field(min_length=1)
    embedding: list[float] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    document_id: UUID
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        """Pydantic config."""

        from_attributes = True
        json_schema_extra = {
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174002",
                "content": "This is a sample chunk of text.",
                "embedding": [0.1, 0.2, 0.3],
                "metadata": {"page": 1, "position": 0},
                "document_id": "123e4567-e89b-12d3-a456-426614174001",
            }
        }
