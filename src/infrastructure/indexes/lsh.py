"""Locality-Sensitive Hashing (LSH) index implementation."""

from __future__ import annotations

import numpy as np
from uuid import UUID

from src.domain.models.chunk import Chunk
from src.infrastructure.indexes.base import VectorIndex
from src.utils.math_utils import cosine_similarity


class LSHIndex(VectorIndex):
    """Locality-Sensitive Hashing (LSH) implementation.

    Time Complexity:
        - Build: O(n*L*k) where L=tables, k=hash functions
        - Search: O(1) average, O(n) worst case
        - Add: O(L*k)
        - Remove: O(L*k)
        - Space: O(n*L)

    Pros:
        - Very fast search (constant time average)
        - Memory efficient
        - Good for high-dimensional data

    Cons:
        - Approximate results
        - Requires parameter tuning
        - Quality depends on hash functions
    """

    def __init__(
        self,
        num_tables: int = 10,
        num_hyperplanes: int = 16,
    ) -> None:
        """Initialize LSH index.

        Args:
            num_tables: Number of hash tables (L)
            num_hyperplanes: Number of random hyperplanes per table (k)
        """
        self.num_tables = num_tables
        self.num_hyperplanes = num_hyperplanes
        self._hash_tables: list[dict[str, list[UUID]]] = []
        self._hyperplanes: list[list[list[float]]] = []
        self._chunks_map: dict[UUID, Chunk] = {}
        self._dimension: int | None = None

    def build(self, chunks: list[Chunk]) -> None:
        """Build LSH hash tables."""
        self.clear()

        if not chunks:
            return

        # Get embedding dimension from first chunk
        first_embedding = next((c.embedding for c in chunks if c.embedding), None)
        if not first_embedding:
            return

        self._dimension = len(first_embedding)

        # Generate random hyperplanes for each table
        self._hyperplanes = []
        for _ in range(self.num_tables):
            # Each table has num_hyperplanes random hyperplanes
            table_hyperplanes = []
            for _ in range(self.num_hyperplanes):
                # Random unit vector (hyperplane normal)
                hyperplane = np.random.randn(self._dimension)
                hyperplane = hyperplane / np.linalg.norm(hyperplane)
                table_hyperplanes.append(hyperplane.tolist())
            self._hyperplanes.append(table_hyperplanes)

        # Initialize hash tables
        self._hash_tables = [{} for _ in range(self.num_tables)]

        # Hash all chunks and insert into tables
        for chunk in chunks:
            if chunk.embedding and len(chunk.embedding) == self._dimension:
                self._add_to_tables(chunk)

    def search(
        self,
        query_embedding: list[float],
        k: int = 10,
    ) -> list[tuple[UUID, float]]:
        """Search using LSH hash tables."""
        if not self._hash_tables or not self._hyperplanes:
            return []

        # Collect candidates from all hash tables
        candidates = set()

        for table_idx in range(self.num_tables):
            # Hash query vector for this table
            hash_key = self._hash_vector(query_embedding, table_idx)

            # Get candidates with same hash
            if hash_key in self._hash_tables[table_idx]:
                candidates.update(self._hash_tables[table_idx][hash_key])

        # If no candidates found, return empty list
        if not candidates:
            return []

        # Re-rank candidates by actual cosine similarity
        results = []
        for chunk_id in candidates:
            chunk = self._chunks_map.get(chunk_id)
            if chunk and chunk.embedding:
                similarity = cosine_similarity(query_embedding, chunk.embedding)
                results.append((chunk_id, similarity))

        # Sort by similarity (descending) and return top k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def add(self, chunk: Chunk) -> None:
        """Add a chunk to LSH tables."""
        if not chunk.embedding:
            return

        # If hyperplanes not initialized, initialize them
        if not self._hyperplanes:
            self._dimension = len(chunk.embedding)
            self._hyperplanes = []
            for _ in range(self.num_tables):
                table_hyperplanes = []
                for _ in range(self.num_hyperplanes):
                    hyperplane = np.random.randn(self._dimension)
                    hyperplane = hyperplane / np.linalg.norm(hyperplane)
                    table_hyperplanes.append(hyperplane.tolist())
                self._hyperplanes.append(table_hyperplanes)
            self._hash_tables = [{} for _ in range(self.num_tables)]

        # Validate dimension
        if len(chunk.embedding) != self._dimension:
            raise ValueError(
                f"Chunk embedding dimension {len(chunk.embedding)} "
                f"does not match index dimension {self._dimension}"
            )

        # Add chunk
        self._add_to_tables(chunk)

    def remove(self, chunk_id: UUID) -> None:
        """Remove a chunk from LSH tables."""
        if chunk_id not in self._chunks_map:
            return

        chunk = self._chunks_map[chunk_id]

        # Remove from all hash tables
        for table_idx in range(self.num_tables):
            if chunk.embedding:
                hash_key = self._hash_vector(chunk.embedding, table_idx)

                if hash_key in self._hash_tables[table_idx]:
                    self._hash_tables[table_idx][hash_key] = [
                        cid for cid in self._hash_tables[table_idx][hash_key]
                        if cid != chunk_id
                    ]

                    # Remove empty buckets
                    if not self._hash_tables[table_idx][hash_key]:
                        del self._hash_tables[table_idx][hash_key]

        # Remove from chunks map
        del self._chunks_map[chunk_id]

    def size(self) -> int:
        """Get number of indexed vectors."""
        return len(self._chunks_map)

    def clear(self) -> None:
        """Clear the index."""
        self._hash_tables = [{} for _ in range(self.num_tables)]
        self._hyperplanes = []
        self._chunks_map = {}
        self._dimension = None

    def _add_to_tables(self, chunk: Chunk) -> None:
        """Internal method to add chunk to hash tables and chunks map."""
        # Store chunk
        self._chunks_map[chunk.id] = chunk

        # Add to all hash tables
        for table_idx in range(self.num_tables):
            hash_key = self._hash_vector(chunk.embedding, table_idx)

            if hash_key not in self._hash_tables[table_idx]:
                self._hash_tables[table_idx][hash_key] = []

            # Avoid duplicates
            if chunk.id not in self._hash_tables[table_idx][hash_key]:
                self._hash_tables[table_idx][hash_key].append(chunk.id)

    def _hash_vector(
        self,
        vector: list[float],
        table_idx: int,
    ) -> str:
        """Hash a vector using random hyperplanes."""
        if table_idx >= len(self._hyperplanes):
            return ""

        vec = np.array(vector)
        hash_bits = []

        # For each hyperplane, check which side of the plane the vector is on
        for hyperplane in self._hyperplanes[table_idx]:
            hp = np.array(hyperplane)
            # Dot product > 0 means same side as hyperplane normal
            dot_prod = np.dot(vec, hp)
            hash_bits.append('1' if dot_prod >= 0 else '0')

        # Convert binary string to hash key
        return ''.join(hash_bits)
