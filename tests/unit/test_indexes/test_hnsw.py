"""Unit tests for HNSW index."""

import pytest
from uuid import uuid4

from src.domain.models.chunk import Chunk
from src.infrastructure.indexes.hnsw import HNSWIndex


@pytest.mark.unit
class TestHNSWIndex:
    """Tests for HNSW index implementation."""

    def test_hnsw_init(self) -> None:
        """Test HNSW index initialization."""
        index = HNSWIndex(dimension=128, m=16, ef_construction=200)

        assert index._dimension == 128
        assert index.m == 16
        assert index.ef_construction == 200

    def test_hnsw_init_default_parameters(self) -> None:
        """Test HNSW index with default parameters."""
        index = HNSWIndex(dimension=128)

        assert index._dimension == 128
        assert index.m > 0
        assert index.ef_construction > 0

    def test_hnsw_build_empty(self) -> None:
        """Test building HNSW index with empty vectors."""
        index = HNSWIndex(dimension=128)
        index.build([])

        assert index.size() == 0

    def test_hnsw_build_single_vector(
        self, sample_embedding_small: list[float]
    ) -> None:
        """Test building HNSW index with single vector."""
        index = HNSWIndex(dimension=128)
        chunks = [
            Chunk(
                content="Test chunk 1",
                embedding=sample_embedding_small,
                document_id=uuid4()
            )
        ]

        index.build(chunks)

        assert index.size() == 1

    def test_hnsw_build_multiple_vectors(
        self, sample_embedding_small: list[float]
    ) -> None:
        """Test building HNSW index with multiple vectors."""
        index = HNSWIndex(dimension=128)

        # Create multiple chunks
        chunks = [
            Chunk(
                content=f"Test chunk {i}",
                embedding=[i * 0.01 + j * 0.001 for j in range(128)],
                document_id=uuid4()
            )
            for i in range(10)
        ]

        index.build(chunks)

        assert index.size() == 10

    def test_hnsw_search_empty_index(
        self, sample_embedding_small: list[float]
    ) -> None:
        """Test searching in empty HNSW index."""
        index = HNSWIndex(dimension=128)
        index.build([])

        results = index.search(sample_embedding_small, k=5)

        assert isinstance(results, list)
        assert len(results) == 0

    def test_hnsw_search_single_result(
        self, sample_embedding_small: list[float]
    ) -> None:
        """Test searching HNSW index with one result."""
        index = HNSWIndex(dimension=128)
        chunks = [
            Chunk(
                content="Test chunk 1",
                embedding=sample_embedding_small,
                document_id=uuid4()
            )
        ]

        index.build(chunks)
        results = index.search(sample_embedding_small, k=5)

        assert len(results) <= 1
        if len(results) > 0:
            assert isinstance(results[0], tuple)
            assert len(results[0]) == 2
            chunk_id, score = results[0]
            assert isinstance(score, float)

    def test_hnsw_search_multiple_results(
        self, sample_embedding_small: list[float]
    ) -> None:
        """Test searching HNSW index with multiple results."""
        index = HNSWIndex(dimension=128)

        # Create multiple similar chunks
        chunks = [
            Chunk(
                content=f"Test chunk {i}",
                embedding=[i * 0.01 + j * 0.001 for j in range(128)],
                document_id=uuid4()
            )
            for i in range(10)
        ]

        index.build(chunks)
        results = index.search(chunks[0].embedding, k=5)

        assert len(results) <= 5
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)

    def test_hnsw_search_k_larger_than_size(
        self, sample_embedding_small: list[float]
    ) -> None:
        """Test searching with k larger than index size."""
        index = HNSWIndex(dimension=128)

        chunks = [
            Chunk(
                content=f"Test chunk {i}",
                embedding=[i * 0.01 + j * 0.001 for j in range(128)],
                document_id=uuid4()
            )
            for i in range(3)
        ]

        index.build(chunks)
        results = index.search(chunks[0].embedding, k=10)

        # Should return at most 3 results
        assert len(results) <= 3

    def test_hnsw_search_results_sorted(
        self, sample_embedding_small: list[float]
    ) -> None:
        """Test that HNSW search results are sorted by score."""
        index = HNSWIndex(dimension=128)

        # Create chunks with varying similarity
        chunks = [
            Chunk(
                content=f"Test chunk {i}",
                embedding=[i * 0.1 + j * 0.01 for j in range(128)],
                document_id=uuid4()
            )
            for i in range(10)
        ]

        index.build(chunks)
        results = index.search(chunks[0].embedding, k=5)

        # Check scores are sorted (descending)
        if len(results) > 1:
            scores = [score for _, score in results]
            assert scores == sorted(scores, reverse=True)

    def test_hnsw_size(self, sample_embedding_small: list[float]) -> None:
        """Test HNSW index size method."""
        index = HNSWIndex(dimension=128)

        assert index.size() == 0

        chunks = [
            Chunk(
                content=f"Test chunk {i}",
                embedding=[i * 0.01 + j * 0.001 for j in range(128)],
                document_id=uuid4()
            )
            for i in range(5)
        ]
        index.build(chunks)

        assert index.size() == 5

    def test_hnsw_invalid_dimension(
        self, sample_embedding_small: list[float]
    ) -> None:
        """Test HNSW with mismatched dimensions."""
        index = HNSWIndex(dimension=64)  # Different dimension

        # Try to build with 128-dimensional chunk
        chunks = [
            Chunk(
                content="Test chunk",
                embedding=sample_embedding_small,  # 128-dimensional
                document_id=uuid4()
            )
        ]
        with pytest.raises((ValueError, AssertionError, Exception)):
            index.build(chunks)

    def test_hnsw_recall_quality(self) -> None:
        """Test HNSW recall quality on simple dataset."""
        index = HNSWIndex(dimension=128, m=16, ef_construction=200)

        # Create a simple dataset
        chunks = []
        for i in range(100):
            chunk = Chunk(
                content=f"Test chunk {i}",
                embedding=[i * 0.01 + j * 0.001 for j in range(128)],
                document_id=uuid4()
            )
            chunks.append(chunk)

        index.build(chunks)

        # Search for the first vector
        results = index.search(chunks[0].embedding, k=10)

        # Should find the exact match
        assert len(results) > 0
        # The first chunk should be in the results
        assert any(chunk_id == chunks[0].id for chunk_id, _ in results)

    def test_hnsw_rebuild(self, sample_embedding_small: list[float]) -> None:
        """Test rebuilding HNSW index."""
        index = HNSWIndex(dimension=128)

        # Build first time
        chunks1 = [
            Chunk(
                content=f"Test chunk {i}",
                embedding=[i * 0.01 + j * 0.001 for j in range(128)],
                document_id=uuid4()
            )
            for i in range(5)
        ]
        index.build(chunks1)
        assert index.size() == 5

        # Rebuild with different data
        chunks2 = [
            Chunk(
                content=f"New chunk {i}",
                embedding=[i * 0.02 + j * 0.001 for j in range(128)],
                document_id=uuid4()
            )
            for i in range(3)
        ]
        index.build(chunks2)
        assert index.size() == 3
