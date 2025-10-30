"""Library domain model."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from src.domain.enums import IndexType


class Library(BaseModel):
    """Library domain model.

    A library is a collection of documents with a specific vector index type.
    """

    id: UUID = Field(default_factory=uuid4)
    name: str = Field(min_length=1, max_length=255)
    description: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    index_type: IndexType = IndexType.BRUTE_FORCE
    document_ids: list[UUID] = Field(default_factory=list)
    is_indexed: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        """Pydantic config."""

        from_attributes = True
        json_schema_extra = {
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "name": "My Library",
                "description": "A collection of documents",
                "metadata": {"category": "research"},
                "index_type": "brute_force",
                "document_ids": [],
                "is_indexed": False,
            }
        }
