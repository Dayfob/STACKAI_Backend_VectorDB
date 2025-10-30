"""Chunk endpoints."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.v1.dependencies import get_chunk_service
from src.core.services import ChunkService
from src.schemas.chunk import ChunkCreate, ChunkResponse, ChunkUpdate

router = APIRouter(
    prefix="/documents/{document_id}/chunks",
    tags=["chunks"],
)


@router.post("/", response_model=ChunkResponse, status_code=status.HTTP_201_CREATED)
async def create_chunk(
    document_id: UUID,
    data: ChunkCreate,
    service: ChunkService = Depends(get_chunk_service),
) -> Any:
    """Create a new chunk in a document.

    Args:
        document_id: Document ID
        data: Chunk creation data
        service: Chunk service

    Returns:
        Created chunk with embedding

    Raises:
        HTTPException: If document not found
    """
    from src.core.exceptions import DocumentNotFoundError, EmbeddingError

    try:
        chunk = await service.create_chunk(document_id, data)
        return chunk
    except DocumentNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=e.message,
        )
    except EmbeddingError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate embedding: {e.message}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.get("/", response_model=list[ChunkResponse])
def list_chunks(
    document_id: UUID,
    service: ChunkService = Depends(get_chunk_service),
) -> Any:
    """List all chunks in a document.

    Args:
        document_id: Document ID
        service: Chunk service

    Returns:
        List of chunks
    """
    try:
        chunks = service.list_chunks(document_id=document_id)
        return chunks
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.get("/{chunk_id}", response_model=ChunkResponse)
def get_chunk(
    document_id: UUID,
    chunk_id: UUID,
    service: ChunkService = Depends(get_chunk_service),
) -> Any:
    """Get a chunk by ID.

    Args:
        document_id: Document ID (for consistency)
        chunk_id: Chunk ID
        service: Chunk service

    Returns:
        Chunk

    Raises:
        HTTPException: If chunk not found
    """
    from src.core.exceptions import ChunkNotFoundError

    try:
        chunk = service.get_chunk(chunk_id)
        return chunk
    except ChunkNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=e.message,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.put("/{chunk_id}", response_model=ChunkResponse)
async def update_chunk(
    document_id: UUID,
    chunk_id: UUID,
    data: ChunkUpdate,
    service: ChunkService = Depends(get_chunk_service),
) -> Any:
    """Update a chunk.

    Args:
        document_id: Document ID (for consistency)
        chunk_id: Chunk ID
        data: Update data
        service: Chunk service

    Returns:
        Updated chunk

    Raises:
        HTTPException: If chunk not found
    """
    from src.core.exceptions import ChunkNotFoundError, EmbeddingError

    try:
        chunk = await service.update_chunk(chunk_id, data)
        return chunk
    except ChunkNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=e.message,
        )
    except EmbeddingError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate embedding: {e.message}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.delete("/{chunk_id}", status_code=status.HTTP_204_NO_CONTENT, response_model=None)
def delete_chunk(
    document_id: UUID,
    chunk_id: UUID,
    service: ChunkService = Depends(get_chunk_service),
) -> None:
    """Delete a chunk.

    Args:
        document_id: Document ID (for consistency)
        chunk_id: Chunk ID
        service: Chunk service

    Raises:
        HTTPException: If chunk not found
    """
    from src.core.exceptions import ChunkNotFoundError

    try:
        service.delete_chunk(chunk_id)
    except ChunkNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=e.message,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )
