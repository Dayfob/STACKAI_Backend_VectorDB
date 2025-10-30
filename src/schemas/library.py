"""Library API schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from src.domain.enums import IndexType


class LibraryCreate(BaseModel):
    """Schema for creating a library."""

    name: str = Field(min_length=1, max_length=255)
    description: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    index_type: IndexType = IndexType.BRUTE_FORCE


class LibraryUpdate(BaseModel):
    """Schema for updating a library."""

    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None
    index_type: Optional[IndexType] = None


class LibraryResponse(BaseModel):
    """Schema for library response."""

    id: UUID
    name: str
    description: Optional[str]
    metadata: dict[str, Any]
    index_type: IndexType
    document_ids: list[UUID]
    is_indexed: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        """Pydantic config."""

        from_attributes = True
