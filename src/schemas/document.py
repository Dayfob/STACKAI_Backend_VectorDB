"""Document API schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class DocumentCreate(BaseModel):
    """Schema for creating a document."""

    name: str = Field(min_length=1, max_length=255)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentUpdate(BaseModel):
    """Schema for updating a document."""

    name: Optional[str] = Field(None, min_length=1, max_length=255)
    metadata: Optional[dict[str, Any]] = None


class DocumentResponse(BaseModel):
    """Schema for document response."""

    id: UUID
    name: str
    metadata: dict[str, Any]
    library_id: UUID
    chunk_ids: list[UUID]
    created_at: datetime
    updated_at: datetime

    class Config:
        """Pydantic config."""

        from_attributes = True
