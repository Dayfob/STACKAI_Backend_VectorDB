"""Unit tests for LSH index."""

from uuid import uuid4

import pytest

from src.domain.models.chunk import Chunk
from src.infrastructure.indexes.lsh import LSHIndex


@pytest.mark.unit
class TestLSHIndex:
    """Tests for LSH index implementation."""

    def test_lsh_init(self) -> None:
        """Test LSH index initialization."""
        index = LSHIndex(num_tables=10, num_hyperplanes=16)

        assert index.num_tables == 10
        assert index.num_hyperplanes == 16
        assert index.size() == 0

    def test_lsh_init_default_parameters(self) -> None:
        """Test LSH index with default parameters."""
        index = LSHIndex()

        assert index.num_tables == 10
        assert index.num_hyperplanes == 16

    def test_lsh_build_empty(self) -> None:
        """Test building LSH index with empty chunks."""
        index = LSHIndex()
        index.build([])

        assert index.size() == 0

    def test_lsh_build_single_chunk(
        self, sample_embedding: list[float]
    ) -> None:
        """Test building LSH index with single chunk."""
        index = LSHIndex()

        chunk = Chunk(
            id=uuid4(),
            document_id=uuid4(),
            content="Test content",
            embedding=sample_embedding,
            metadata={},
        )

        index.build([chunk])

        assert index.size() == 1

    def test_lsh_build_multiple_chunks(
        self, sample_embedding: list[float]
    ) -> None:
        """Test building LSH index with multiple chunks."""
        index = LSHIndex()

        # Create multiple chunks with different embeddings
        chunks = []
        for i in range(10):
            embedding = [x + i * 0.01 for x in sample_embedding]
            chunk = Chunk(
                id=uuid4(),
                document_id=uuid4(),
                content=f"Content {i}",
                embedding=embedding,
                metadata={},
            )
            chunks.append(chunk)

        index.build(chunks)

        assert index.size() == 10

    def test_lsh_search_empty_index(
        self, sample_embedding: list[float]
    ) -> None:
        """Test searching in empty LSH index."""
        index = LSHIndex()
        index.build([])

        results = index.search(sample_embedding, k=5)

        assert isinstance(results, list)
        assert len(results) == 0

    def test_lsh_search_single_result(
        self, sample_embedding: list[float]
    ) -> None:
        """Test searching LSH index with one chunk."""
        index = LSHIndex()

        chunk = Chunk(
            id=uuid4(),
            document_id=uuid4(),
            content="Test content",
            embedding=sample_embedding,
            metadata={},
        )

        index.build([chunk])
        results = index.search(sample_embedding, k=5)

        # LSH is approximate, may or may not find it
        assert isinstance(results, list)
        if len(results) > 0:
            assert len(results[0]) == 2  # (UUID, float)
            chunk_id, score = results[0]
            assert isinstance(score, float)

    def test_lsh_search_multiple_results(
        self, sample_embedding: list[float]
    ) -> None:
        """Test searching LSH index with multiple chunks."""
        index = LSHIndex(num_tables=10, num_hyperplanes=16)

        # Create multiple similar chunks
        chunks = []
        for i in range(20):
            embedding = [x + i * 0.001 for x in sample_embedding]
            chunk = Chunk(
                id=uuid4(),
                document_id=uuid4(),
                content=f"Content {i}",
                embedding=embedding,
                metadata={},
            )
            chunks.append(chunk)

        index.build(chunks)
        results = index.search(sample_embedding, k=5)

        # LSH is approximate, so results may vary
        assert isinstance(results, list)
        for result in results:
            assert len(result) == 2
            chunk_id, score = result
            assert isinstance(score, float)
            assert 0 <= score <= 1

    def test_lsh_add_chunk(self, sample_embedding: list[float]) -> None:
        """Test adding a chunk to LSH index."""
        index = LSHIndex()

        chunk = Chunk(
            id=uuid4(),
            document_id=uuid4(),
            content="Test content",
            embedding=sample_embedding,
            metadata={},
        )

        index.add(chunk)

        assert index.size() == 1

    def test_lsh_remove_chunk(self, sample_embedding: list[float]) -> None:
        """Test removing a chunk from LSH index."""
        index = LSHIndex()

        chunk = Chunk(
            id=uuid4(),
            document_id=uuid4(),
            content="Test content",
            embedding=sample_embedding,
            metadata={},
        )

        index.add(chunk)
        assert index.size() == 1

        index.remove(chunk.id)
        assert index.size() == 0

    def test_lsh_clear(self, sample_embedding: list[float]) -> None:
        """Test clearing LSH index."""
        index = LSHIndex()

        chunks = []
        for i in range(5):
            chunk = Chunk(
                id=uuid4(),
                document_id=uuid4(),
                content=f"Content {i}",
                embedding=sample_embedding,
                metadata={},
            )
            chunks.append(chunk)

        index.build(chunks)
        assert index.size() == 5

        index.clear()
        assert index.size() == 0

    def test_lsh_rebuild(self, sample_embedding: list[float]) -> None:
        """Test rebuilding LSH index."""
        index = LSHIndex()

        # Build first time
        chunks1 = []
        for i in range(5):
            chunk = Chunk(
                id=uuid4(),
                document_id=uuid4(),
                content=f"Content {i}",
                embedding=sample_embedding,
                metadata={},
            )
            chunks1.append(chunk)

        index.build(chunks1)
        assert index.size() == 5

        # Rebuild with different data
        chunks2 = []
        for i in range(3):
            embedding = [x + i * 0.1 for x in sample_embedding]
            chunk = Chunk(
                id=uuid4(),
                document_id=uuid4(),
                content=f"New content {i}",
                embedding=embedding,
                metadata={},
            )
            chunks2.append(chunk)

        index.build(chunks2)
        assert index.size() == 3

    def test_lsh_dimension_validation(
        self, sample_embedding: list[float]
    ) -> None:
        """Test LSH with mismatched embedding dimensions."""
        index = LSHIndex()

        # Add first chunk with correct dimension
        chunk1 = Chunk(
            id=uuid4(),
            document_id=uuid4(),
            content="Content 1",
            embedding=sample_embedding,
            metadata={},
        )
        index.add(chunk1)

        # Try to add chunk with different dimension
        wrong_embedding = sample_embedding[:len(sample_embedding)//2]
        chunk2 = Chunk(
            id=uuid4(),
            document_id=uuid4(),
            content="Content 2",
            embedding=wrong_embedding,
            metadata={},
        )

        with pytest.raises(ValueError, match="dimension"):
            index.add(chunk2)
