"""Chunk API schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class ChunkCreate(BaseModel):
    """Schema for creating a chunk."""

    content: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChunkUpdate(BaseModel):
    """Schema for updating a chunk."""

    content: Optional[str] = Field(None, min_length=1)
    metadata: Optional[dict[str, Any]] = None


class ChunkResponse(BaseModel):
    """Schema for chunk response."""

    id: UUID
    content: str
    embedding: list[float]
    metadata: dict[str, Any]
    document_id: UUID
    created_at: datetime
    updated_at: datetime

    class Config:
        """Pydantic config."""

        from_attributes = True
