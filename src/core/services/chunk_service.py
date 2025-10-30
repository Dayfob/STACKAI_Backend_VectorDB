"""Chunk service for business logic."""

from __future__ import annotations

from typing import Any, Optional
from uuid import UUID

from src.domain.models.chunk import Chunk
from src.infrastructure.repositories.chunk_repository import ChunkRepository
from src.infrastructure.repositories.document_repository import DocumentRepository
from src.schemas.chunk import ChunkCreate, ChunkUpdate
from src.utils.embeddings import EmbeddingService


class ChunkService:
    """Service for managing chunks."""

    def __init__(
        self,
        repository: ChunkRepository,
        document_repository: DocumentRepository,
        embedding_service: EmbeddingService,
    ) -> None:
        """Initialize chunk service.

        Args:
            repository: Chunk repository
            document_repository: Document repository
            embedding_service: Embedding service
        """
        self.repository = repository
        self.document_repository = document_repository
        self.embedding_service = embedding_service

    async def create_chunk(self, document_id: UUID, data: ChunkCreate) -> Chunk:
        """Create a new chunk in a document.

        Args:
            document_id: Document ID
            data: Chunk creation data

        Returns:
            Created chunk with embedding

        Raises:
            DocumentNotFoundError: If document not found
            EmbeddingError: If embedding generation fails
        """
        from src.core.exceptions import DocumentNotFoundError

        # Check document exists
        document = self.document_repository.get(document_id)
        if not document:
            raise DocumentNotFoundError(str(document_id))

        # Generate embedding for chunk content
        embedding = await self.embedding_service.embed_text(data.content)

        # Create chunk with embedding
        chunk = Chunk(
            content=data.content,
            embedding=embedding,
            metadata=data.metadata,
            document_id=document_id,
        )
        created_chunk = self.repository.create(chunk)

        # Update document's chunk_ids
        updated_chunk_ids = document.chunk_ids + [created_chunk.id]
        self.document_repository.update(document_id, {"chunk_ids": updated_chunk_ids})

        return created_chunk

    async def get_embedding(self, text: str) -> list[float]:
        """Generate embedding for text.

        Args:
            text: Input text

        Returns:
            Embedding vector

        Raises:
            EmbeddingError: If embedding generation fails
        """
        return await self.embedding_service.embed_text(text)

    def get_chunk(self, chunk_id: UUID) -> Chunk:
        """Get a chunk by ID.

        Args:
            chunk_id: Chunk ID

        Returns:
            Chunk

        Raises:
            ChunkNotFoundError: If chunk not found
        """
        from src.core.exceptions import ChunkNotFoundError

        chunk = self.repository.get(chunk_id)
        if not chunk:
            raise ChunkNotFoundError(str(chunk_id))
        return chunk

    def list_chunks(
        self,
        document_id: Optional[UUID] = None,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[Chunk]:
        """List chunks.

        Args:
            document_id: Optional document ID to filter by
            filters: Optional filters

        Returns:
            List of chunks
        """
        if document_id:
            return self.repository.list_by_document(document_id)

        return self.repository.list(filters)

    async def update_chunk(self, chunk_id: UUID, data: ChunkUpdate) -> Chunk:
        """Update a chunk.

        Args:
            chunk_id: Chunk ID
            data: Update data

        Returns:
            Updated chunk

        Raises:
            ChunkNotFoundError: If chunk not found
        """
        from src.core.exceptions import ChunkNotFoundError

        # Filter out None values
        update_data = {k: v for k, v in data.model_dump().items() if v is not None}

        # If content changed, regenerate embedding
        if data.content is not None:
            embedding = await self.embedding_service.embed_text(data.content)
            update_data["embedding"] = embedding

        # If nothing to update, just return the existing chunk
        if not update_data:
            return self.get_chunk(chunk_id)

        updated_chunk = self.repository.update(chunk_id, update_data)
        if not updated_chunk:
            raise ChunkNotFoundError(str(chunk_id))
        return updated_chunk

    def delete_chunk(self, chunk_id: UUID) -> None:
        """Delete a chunk.

        Args:
            chunk_id: Chunk ID

        Raises:
            ChunkNotFoundError: If chunk not found
        """
        from src.core.exceptions import ChunkNotFoundError

        # Get chunk first to access document_id
        chunk = self.get_chunk(chunk_id)

        # Delete chunk
        success = self.repository.delete(chunk_id)
        if not success:
            raise ChunkNotFoundError(str(chunk_id))

        # Update document's chunk_ids
        document = self.document_repository.get(chunk.document_id)
        if document:
            updated_chunk_ids = [
                c_id for c_id in document.chunk_ids if c_id != chunk_id
            ]
            self.document_repository.update(chunk.document_id, {"chunk_ids": updated_chunk_ids})
