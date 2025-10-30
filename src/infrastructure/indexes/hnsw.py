"""Hierarchical Navigable Small World (HNSW) index implementation."""

from __future__ import annotations

import heapq
import math
import random
from uuid import UUID

from src.domain.models.chunk import Chunk
from src.infrastructure.indexes.base import VectorIndex
from src.utils.math_utils import cosine_similarity
from src.utils.validators import validate_embedding_dimension


class HNSWIndex(VectorIndex):
    """Simplified HNSW (Hierarchical Navigable Small World) implementation.

    Time Complexity:
        - Build: O(n*log(n)*d) amortized
        - Search: O(log(n)*d) average case
        - Add: O(log(n)*d) amortized
        - Remove: O(M) where M=connections per node
        - Space: O(n*M*d) where M=max connections

    Pros:
        - Very fast search
        - Good recall/accuracy
        - Supports dynamic updates

    Cons:
        - Complex implementation
        - Higher memory usage
        - Approximate results
    """

    def __init__(
        self,
        m: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
        dimension: int | None = None,
    ) -> None:
        """Initialize HNSW index.

        Args:
            m: Maximum number of bi-directional links per node
            ef_construction: Size of dynamic candidate list during construction
            ef_search: Size of dynamic candidate list during search
            dimension: Expected embedding dimension (auto-detected if None)
        """
        self.m = m
        self.m_max = m
        self.m_max0 = m * 2  # Layer 0 can have more connections
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self._graph: dict[int, dict[UUID, list[UUID]]] = {}  # layer -> node -> neighbors
        self._chunks_map: dict[UUID, Chunk] = {}
        self._entry_point: UUID | None = None
        self._node_layers: dict[UUID, int] = {}  # node -> max layer
        self._dimension: int | None = dimension
        self._initial_dimension: int | None = dimension  # For reset in clear()

    def build(self, chunks: list[Chunk]) -> None:
        """Build HNSW graph."""
        self.clear()
        for chunk in chunks:
            self.add(chunk)

    def search(
        self,
        query_embedding: list[float],
        k: int = 10,
    ) -> list[tuple[UUID, float]]:
        """Search using HNSW graph traversal."""
        if self._entry_point is None or not self._chunks_map:
            return []

        # Start from entry point at top layer
        current_nearest = [self._entry_point]

        # Navigate through layers from top to bottom
        for layer in range(max(self._node_layers.values()), 0, -1):
            current_nearest = self._search_layer(
                query_embedding, current_nearest, 1, layer
            )

        # Search at layer 0 with ef parameter
        candidates = self._search_layer(
            query_embedding, current_nearest, max(self.ef_search, k), 0
        )

        # Return top k results
        results = []
        for node_id in candidates[:k]:
            chunk = self._chunks_map.get(node_id)
            if chunk and chunk.embedding:
                similarity = cosine_similarity(query_embedding, chunk.embedding)
                results.append((node_id, similarity))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def add(self, chunk: Chunk) -> None:
        """Add a chunk to the HNSW graph."""
        if not chunk.embedding:
            return

        # Auto-detect dimension from first chunk
        if self._dimension is None:
            self._dimension = len(chunk.embedding)

        # Validate embedding dimension
        validate_embedding_dimension(chunk.embedding, self._dimension)

        node_id = chunk.id
        self._chunks_map[node_id] = chunk

        # Determine layer for new node (exponential decay)
        layer = self._get_random_layer()
        self._node_layers[node_id] = layer

        # Initialize graph structure for new layers
        for lc in range(layer + 1):
            if lc not in self._graph:
                self._graph[lc] = {}
            self._graph[lc][node_id] = []

        # If this is the first node
        if self._entry_point is None:
            self._entry_point = node_id
            return

        # Find nearest neighbors at each layer
        nearest = [self._entry_point]

        # Get current max layer in graph
        current_max_layer = max(self._node_layers.values()) if self._node_layers else 0

        # Search from top to target layer
        for lc in range(current_max_layer, layer, -1):
            nearest = self._search_layer(chunk.embedding, nearest, 1, lc)

        # Insert from target layer down to 0
        for lc in range(min(layer, current_max_layer), -1, -1):
            candidates = self._search_layer(
                chunk.embedding, nearest, self.ef_construction, lc
            )

            # Determine m for this layer
            m = self.m_max0 if lc == 0 else self.m_max

            # Select m nearest neighbors
            neighbors = self._get_neighbors(chunk.embedding, candidates, m)

            # Add bidirectional links
            self._graph[lc][node_id] = neighbors
            for neighbor_id in neighbors:
                # Ensure neighbor exists at this layer
                if neighbor_id not in self._graph[lc]:
                    self._graph[lc][neighbor_id] = []

                self._graph[lc][neighbor_id].append(node_id)

                # Prune neighbors if needed
                max_conn = self.m_max0 if lc == 0 else self.m_max
                if len(self._graph[lc][neighbor_id]) > max_conn:
                    neighbor_chunk = self._chunks_map[neighbor_id]
                    self._graph[lc][neighbor_id] = self._get_neighbors(
                        neighbor_chunk.embedding,
                        self._graph[lc][neighbor_id],
                        max_conn
                    )

        # Update entry point if necessary
        if layer > self._node_layers.get(self._entry_point, 0):
            self._entry_point = node_id

    def remove(self, chunk_id: UUID) -> None:
        """Remove a chunk from the HNSW graph."""
        if chunk_id not in self._chunks_map:
            return

        # Remove from all layers
        max_layer = self._node_layers.get(chunk_id, 0)
        for layer in range(max_layer + 1):
            if layer in self._graph and chunk_id in self._graph[layer]:
                # Remove connections to this node
                neighbors = self._graph[layer][chunk_id]
                for neighbor_id in neighbors:
                    if neighbor_id in self._graph[layer]:
                        self._graph[layer][neighbor_id] = [
                            n for n in self._graph[layer][neighbor_id]
                            if n != chunk_id
                        ]

                # Remove the node itself
                del self._graph[layer][chunk_id]

        # Remove from metadata
        del self._chunks_map[chunk_id]
        del self._node_layers[chunk_id]

        # Update entry point if needed
        if self._entry_point == chunk_id:
            if self._chunks_map:
                self._entry_point = next(iter(self._chunks_map.keys()))
            else:
                self._entry_point = None

    def size(self) -> int:
        """Get number of indexed vectors."""
        return len(self._chunks_map)

    def clear(self) -> None:
        """Clear the index."""
        self._graph = {}
        self._chunks_map = {}
        self._entry_point = None
        self._node_layers = {}
        self._dimension = self._initial_dimension  # Reset to initial value

    def _get_random_layer(self) -> int:
        """Get random layer using exponential decay."""
        ml = 1.0 / math.log(self.m) if self.m > 1 else 1.0
        return int(-math.log(random.uniform(0.0001, 1)) * ml)

    def _search_layer(
        self,
        query_embedding: list[float],
        entry_points: list[UUID],
        num_closest: int,
        layer: int,
    ) -> list[UUID]:
        """Search for nearest neighbors at a specific layer."""
        if layer not in self._graph:
            return entry_points[:num_closest]

        visited = set(entry_points)
        candidates = []
        w = []

        # Initialize with entry points
        for ep in entry_points:
            chunk = self._chunks_map.get(ep)
            if chunk and chunk.embedding:
                dist = -cosine_similarity(query_embedding, chunk.embedding)
                heapq.heappush(candidates, (dist, ep))
                heapq.heappush(w, (-dist, ep))

        while candidates:
            current_dist, current = heapq.heappop(candidates)

            # Stop if we've found enough good candidates
            if current_dist > -w[0][0]:
                break

            # Check neighbors
            neighbors = self._graph[layer].get(current, [])
            for neighbor_id in neighbors:
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    chunk = self._chunks_map.get(neighbor_id)
                    if chunk and chunk.embedding:
                        dist = -cosine_similarity(query_embedding, chunk.embedding)

                        if dist < -w[0][0] or len(w) < num_closest:
                            heapq.heappush(candidates, (dist, neighbor_id))
                            heapq.heappush(w, (-dist, neighbor_id))

                            if len(w) > num_closest:
                                heapq.heappop(w)

        # Return sorted results (best first)
        results = sorted(w, reverse=True)
        return [node_id for _, node_id in results]

    def _get_neighbors(
        self,
        embedding: list[float],
        candidates: list[UUID],
        m: int,
    ) -> list[UUID]:
        """Select m nearest neighbors from candidates."""
        # Filter out invalid candidates
        valid_candidates = [
            c for c in candidates
            if c in self._chunks_map and self._chunks_map[c].embedding
        ]

        if len(valid_candidates) <= m:
            return valid_candidates

        # Calculate similarities
        scored = []
        for candidate_id in valid_candidates:
            chunk = self._chunks_map[candidate_id]
            similarity = cosine_similarity(embedding, chunk.embedding)
            scored.append((similarity, candidate_id))

        # Sort by similarity (descending) and take top m
        scored.sort(reverse=True)
        return [node_id for _, node_id in scored[:m]]
