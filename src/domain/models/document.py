"""Document domain model."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class Document(BaseModel):
    """Document domain model.

    A document represents a text document that can be chunked for embedding.
    """

    id: UUID = Field(default_factory=uuid4)
    name: str = Field(min_length=1, max_length=255)
    metadata: dict[str, Any] = Field(default_factory=dict)
    library_id: UUID
    chunk_ids: list[UUID] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        """Pydantic config."""

        from_attributes = True
        json_schema_extra = {
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174001",
                "name": "Research Paper",
                "metadata": {"author": "John Doe", "year": 2024},
                "library_id": "123e4567-e89b-12d3-a456-426614174000",
                "chunk_ids": [],
            }
        }
