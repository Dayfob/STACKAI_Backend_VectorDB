"""Abstract base class for vector indexes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from uuid import UUID

from src.domain.models.chunk import Chunk


class VectorIndex(ABC):
    """Abstract base class for vector similarity search indexes.

    All index implementations must inherit from this class and implement
    the required methods for building, searching, adding, and removing vectors.
    """

    @abstractmethod
    def build(self, chunks: list[Chunk]) -> None:
        """Build the index from a list of chunks.

        Args:
            chunks: List of chunks with embeddings to index
        """
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        k: int = 10,
    ) -> list[tuple[UUID, float]]:
        """Search for k nearest neighbors.

        Args:
            query_embedding: Query vector
            k: Number of neighbors to return

        Returns:
            List of (chunk_id, similarity_score) tuples, sorted by score descending
        """
        pass

    @abstractmethod
    def add(self, chunk: Chunk) -> None:
        """Add a single chunk to the index.

        Args:
            chunk: Chunk to add
        """
        pass

    @abstractmethod
    def remove(self, chunk_id: UUID) -> None:
        """Remove a chunk from the index.

        Args:
            chunk_id: ID of chunk to remove
        """
        pass

    @abstractmethod
    def size(self) -> int:
        """Get the number of vectors in the index.

        Returns:
            Number of indexed vectors
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all vectors from the index."""
        pass
