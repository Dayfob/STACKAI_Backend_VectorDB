"""Brute force vector index implementation."""

from __future__ import annotations

from uuid import UUID

from src.domain.models.chunk import Chunk
from src.infrastructure.indexes.base import VectorIndex
from src.utils.math_utils import cosine_similarity
from src.utils.validators import validate_embedding_dimension


class BruteForceIndex(VectorIndex):
    """Brute force k-NN search implementation.

    Time Complexity:
        - Build: O(n) - just store all vectors
        - Search: O(n*d) - compare query with all vectors, d=dimension
        - Add: O(1) - append to list
        - Remove: O(n) - find and remove from list
        - Space: O(n*d)

    Pros:
        - Exact results (no approximation)
        - Simple implementation
        - No index building overhead

    Cons:
        - Slow search for large datasets
        - Not scalable
    """

    def __init__(self, dimension: int | None = None) -> None:
        """Initialize brute force index.

        Args:
            dimension: Expected embedding dimension (auto-detected if None)
        """
        self._chunks: list[Chunk] = []
        self._dimension: int | None = dimension
        self._initial_dimension: int | None = dimension  # For reset in clear()

    def build(self, chunks: list[Chunk]) -> None:
        """Build index by storing all chunks."""
        self._chunks = chunks.copy()

    def search(
        self,
        query_embedding: list[float],
        k: int = 10,
    ) -> list[tuple[UUID, float]]:
        """Search using brute force comparison."""
        if not self._chunks:
            return []

        # Calculate cosine similarity with all chunks
        similarities: list[tuple[UUID, float]] = []
        for chunk in self._chunks:
            if not chunk.embedding:
                continue
            similarity = cosine_similarity(query_embedding, chunk.embedding)
            similarities.append((chunk.id, similarity))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Return top k results
        return similarities[:k]

    def add(self, chunk: Chunk) -> None:
        """Add a chunk to the index."""
        if chunk.embedding:
            # Auto-detect dimension from first chunk
            if self._dimension is None:
                self._dimension = len(chunk.embedding)

            # Validate embedding dimension
            validate_embedding_dimension(chunk.embedding, self._dimension)

        self._chunks.append(chunk)

    def remove(self, chunk_id: UUID) -> None:
        """Remove a chunk from the index."""
        self._chunks = [chunk for chunk in self._chunks if chunk.id != chunk_id]

    def size(self) -> int:
        """Get number of indexed vectors."""
        return len(self._chunks)

    def clear(self) -> None:
        """Clear the index."""
        self._chunks = []
        self._dimension = self._initial_dimension  # Reset to initial value
