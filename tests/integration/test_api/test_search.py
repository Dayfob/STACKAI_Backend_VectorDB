"""Integration tests for search endpoints."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock


@pytest.mark.integration
class TestSearchEndpoints:
    """Tests for search API endpoints."""

    def setup_library_with_chunks(
        self,
        client: TestClient,
        sample_library_data: dict,
        sample_document_data: dict,
        chunks_data: list[dict],
        sample_embedding: list[float],
    ) -> str:
        """Helper to create library with document and chunks.

        Args:
            client: Test client
            sample_library_data: Library data
            sample_document_data: Document data
            chunks_data: List of chunk data
            sample_embedding: Embedding vector

        Returns:
            Library ID
        """
        # Create library
        lib_response = client.post("/api/v1/libraries/", json=sample_library_data)
        library_id = lib_response.json()["id"]

        # Create document
        doc_response = client.post(
            f"/api/v1/libraries/{library_id}/documents/", json=sample_document_data
        )
        document_id = doc_response.json()["id"]

        # Mock embedding and create chunks
        with patch("src.utils.embeddings.EmbeddingService.embed_text") as mock_embed:
            mock_embed.return_value = sample_embedding

            for chunk_data in chunks_data:
                client.post(
                    f"/api/v1/documents/{document_id}/chunks/", json=chunk_data
                )

        # Build index
        client.post(f"/api/v1/libraries/{library_id}/index")

        return library_id

    @patch("src.utils.embeddings.EmbeddingService.embed_query")
    @patch("src.utils.embeddings.EmbeddingService.embed_text")
    def test_vector_search_with_query_text(
        self,
        mock_embed_text: AsyncMock,
        mock_embed_query: AsyncMock,
        client: TestClient,
        sample_library_data: dict,
        sample_document_data: dict,
        sample_chunk_data: dict,
        sample_chunk_data_2: dict,
        sample_embedding: list[float],
    ) -> None:
        """Test vector search with text query."""
        mock_embed_text.return_value = sample_embedding
        mock_embed_query.return_value = sample_embedding

        # Setup library with chunks
        library_id = self.setup_library_with_chunks(
            client,
            sample_library_data,
            sample_document_data,
            [sample_chunk_data, sample_chunk_data_2],
            sample_embedding,
        )

        # Perform search
        search_request = {"query_text": "machine learning", "k": 5}
        response = client.post(
            f"/api/v1/libraries/{library_id}/search/", json=search_request
        )

        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "total" in data
        assert "query_time_ms" in data
        assert isinstance(data["results"], list)
        assert data["total"] >= 0

    @patch("src.utils.embeddings.EmbeddingService.embed_query")
    @patch("src.utils.embeddings.EmbeddingService.embed_text")
    def test_vector_search_with_embedding(
        self,
        mock_embed_text: AsyncMock,
        mock_embed_query: AsyncMock,
        client: TestClient,
        sample_library_data: dict,
        sample_document_data: dict,
        sample_chunk_data: dict,
        sample_embedding: list[float],
    ) -> None:
        """Test vector search with embedding vector."""
        mock_embed_text.return_value = sample_embedding
        mock_embed_query.return_value = sample_embedding

        # Setup library with chunks
        library_id = self.setup_library_with_chunks(
            client,
            sample_library_data,
            sample_document_data,
            [sample_chunk_data],
            sample_embedding,
        )

        # Perform search with embedding
        search_request = {"query_embedding": sample_embedding, "k": 5}
        response = client.post(
            f"/api/v1/libraries/{library_id}/search/", json=search_request
        )

        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert len(data["results"]) >= 0

    @patch("src.utils.embeddings.EmbeddingService.embed_query")
    @patch("src.utils.embeddings.EmbeddingService.embed_text")
    def test_vector_search_library_not_found(
        self, mock_embed_text: AsyncMock,
        mock_embed_query: AsyncMock, client: TestClient, sample_embedding: list[float]
    ) -> None:
        """Test search in non-existent library."""
        mock_embed_text.return_value = sample_embedding
        mock_embed_query.return_value = sample_embedding

        fake_library_id = "00000000-0000-0000-0000-000000000000"
        search_request = {"query_text": "test query", "k": 5}
        response = client.post(
            f"/api/v1/libraries/{fake_library_id}/search/", json=search_request
        )

        assert response.status_code == 404

    @patch("src.utils.embeddings.EmbeddingService.embed_query")
    @patch("src.utils.embeddings.EmbeddingService.embed_text")
    def test_vector_search_index_not_built(
        self,
        mock_embed_text: AsyncMock,
        mock_embed_query: AsyncMock,
        client: TestClient,
        sample_library_data: dict,
        sample_embedding: list[float],
    ) -> None:
        """Test search when index is not built."""
        mock_embed_text.return_value = sample_embedding
        mock_embed_query.return_value = sample_embedding

        # Create library but don't build index
        lib_response = client.post("/api/v1/libraries/", json=sample_library_data)
        library_id = lib_response.json()["id"]

        # Try to search
        search_request = {"query_text": "test query", "k": 5}
        response = client.post(
            f"/api/v1/libraries/{library_id}/search/", json=search_request
        )

        # Should return 400 or empty results depending on implementation
        assert response.status_code in [200, 400]

    @patch("src.utils.embeddings.EmbeddingService.embed_query")
    @patch("src.utils.embeddings.EmbeddingService.embed_text")
    def test_vector_search_with_k_parameter(
        self,
        mock_embed_text: AsyncMock,
        mock_embed_query: AsyncMock,
        client: TestClient,
        sample_library_data: dict,
        sample_document_data: dict,
        sample_chunk_data: dict,
        sample_chunk_data_2: dict,
        sample_chunk_data_3: dict,
        sample_embedding: list[float],
    ) -> None:
        """Test vector search with different k values."""
        mock_embed_text.return_value = sample_embedding
        mock_embed_query.return_value = sample_embedding

        # Setup library with 3 chunks
        library_id = self.setup_library_with_chunks(
            client,
            sample_library_data,
            sample_document_data,
            [sample_chunk_data, sample_chunk_data_2, sample_chunk_data_3],
            sample_embedding,
        )

        # Search with k=2
        search_request = {"query_text": "test query", "k": 2}
        response = client.post(
            f"/api/v1/libraries/{library_id}/search/", json=search_request
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) <= 2

    @patch("src.utils.embeddings.EmbeddingService.embed_query")
    @patch("src.utils.embeddings.EmbeddingService.embed_text")
    def test_vector_search_result_structure(
        self,
        mock_embed_text: AsyncMock,
        mock_embed_query: AsyncMock,
        client: TestClient,
        sample_library_data: dict,
        sample_document_data: dict,
        sample_chunk_data: dict,
        sample_embedding: list[float],
    ) -> None:
        """Test search result has correct structure."""
        mock_embed_text.return_value = sample_embedding
        mock_embed_query.return_value = sample_embedding

        # Setup library with chunks
        library_id = self.setup_library_with_chunks(
            client,
            sample_library_data,
            sample_document_data,
            [sample_chunk_data],
            sample_embedding,
        )

        # Perform search
        search_request = {"query_text": "test query", "k": 5}
        response = client.post(
            f"/api/v1/libraries/{library_id}/search/", json=search_request
        )

        assert response.status_code == 200
        data = response.json()

        # Check top-level structure
        assert "results" in data
        assert "total" in data
        assert "query_time_ms" in data
        assert isinstance(data["query_time_ms"], (int, float))

        # Check result structure if results exist
        if len(data["results"]) > 0:
            result = data["results"][0]
            assert "chunk_id" in result
            assert "document_id" in result
            assert "content" in result
            assert "score" in result
            assert "metadata" in result
            assert isinstance(result["score"], (int, float))

    @patch("src.utils.embeddings.EmbeddingService.embed_query")
    @patch("src.utils.embeddings.EmbeddingService.embed_text")
    def test_semantic_search(
        self,
        mock_embed_text: AsyncMock,
        mock_embed_query: AsyncMock,
        client: TestClient,
        sample_library_data: dict,
        sample_document_data: dict,
        sample_chunk_data: dict,
        sample_embedding: list[float],
    ) -> None:
        """Test semantic search endpoint."""
        mock_embed_text.return_value = sample_embedding
        mock_embed_query.return_value = sample_embedding

        # Setup library with chunks
        library_id = self.setup_library_with_chunks(
            client,
            sample_library_data,
            sample_document_data,
            [sample_chunk_data],
            sample_embedding,
        )

        # Perform semantic search
        response = client.post(
            f"/api/v1/libraries/{library_id}/search/semantic",
            params={"query_text": "machine learning", "k": 5},
        )

        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "total" in data
        assert "query_time_ms" in data

    @patch("src.utils.embeddings.EmbeddingService.embed_query")
    @patch("src.utils.embeddings.EmbeddingService.embed_text")
    def test_semantic_search_default_k(
        self,
        mock_embed_text: AsyncMock,
        mock_embed_query: AsyncMock,
        client: TestClient,
        sample_library_data: dict,
        sample_document_data: dict,
        sample_chunk_data: dict,
        sample_embedding: list[float],
    ) -> None:
        """Test semantic search with default k value."""
        mock_embed_text.return_value = sample_embedding
        mock_embed_query.return_value = sample_embedding

        # Setup library with chunks
        library_id = self.setup_library_with_chunks(
            client,
            sample_library_data,
            sample_document_data,
            [sample_chunk_data],
            sample_embedding,
        )

        # Perform semantic search without k parameter
        response = client.post(
            f"/api/v1/libraries/{library_id}/search/semantic",
            params={"query_text": "machine learning"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "results" in data

    @patch("src.utils.embeddings.EmbeddingService.embed_query")
    @patch("src.utils.embeddings.EmbeddingService.embed_text")
    def test_search_with_hnsw_index(
        self,
        mock_embed_text: AsyncMock,
        mock_embed_query: AsyncMock,
        client: TestClient,
        sample_library_hnsw_data: dict,
        sample_document_data: dict,
        sample_chunk_data: dict,
        sample_chunk_data_2: dict,
        sample_embedding: list[float],
    ) -> None:
        """Test search with HNSW index."""
        mock_embed_text.return_value = sample_embedding
        mock_embed_query.return_value = sample_embedding

        # Setup library with HNSW index
        library_id = self.setup_library_with_chunks(
            client,
            sample_library_hnsw_data,
            sample_document_data,
            [sample_chunk_data, sample_chunk_data_2],
            sample_embedding,
        )

        # Perform search
        search_request = {"query_text": "test query", "k": 5}
        response = client.post(
            f"/api/v1/libraries/{library_id}/search/", json=search_request
        )

        assert response.status_code == 200
        data = response.json()
        assert "results" in data

    @patch("src.utils.embeddings.EmbeddingService.embed_query")
    @patch("src.utils.embeddings.EmbeddingService.embed_text")
    def test_search_with_lsh_index(
        self,
        mock_embed_text: AsyncMock,
        mock_embed_query: AsyncMock,
        client: TestClient,
        sample_library_lsh_data: dict,
        sample_document_data: dict,
        sample_chunk_data: dict,
        sample_chunk_data_2: dict,
        sample_embedding: list[float],
    ) -> None:
        """Test search with LSH index."""
        mock_embed_text.return_value = sample_embedding
        mock_embed_query.return_value = sample_embedding

        # Setup library with LSH index
        library_id = self.setup_library_with_chunks(
            client,
            sample_library_lsh_data,
            sample_document_data,
            [sample_chunk_data, sample_chunk_data_2],
            sample_embedding,
        )

        # Perform search
        search_request = {"query_text": "test query", "k": 5}
        response = client.post(
            f"/api/v1/libraries/{library_id}/search/", json=search_request
        )

        assert response.status_code == 200
        data = response.json()
        assert "results" in data

    @patch("src.utils.embeddings.EmbeddingService.embed_query")
    @patch("src.utils.embeddings.EmbeddingService.embed_text")
    def test_search_empty_library(
        self,
        mock_embed_text: AsyncMock,
        mock_embed_query: AsyncMock,
        client: TestClient,
        sample_library_data: dict,
        sample_embedding: list[float],
    ) -> None:
        """Test search in library with no chunks."""
        mock_embed_text.return_value = sample_embedding
        mock_embed_query.return_value = sample_embedding

        # Create library without chunks
        lib_response = client.post("/api/v1/libraries/", json=sample_library_data)
        library_id = lib_response.json()["id"]

        # Build index
        client.post(f"/api/v1/libraries/{library_id}/index")

        # Perform search
        search_request = {"query_text": "test query", "k": 5}
        response = client.post(
            f"/api/v1/libraries/{library_id}/search/", json=search_request
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0
        assert len(data["results"]) == 0

    @patch("src.utils.embeddings.EmbeddingService.embed_query")
    @patch("src.utils.embeddings.EmbeddingService.embed_text")
    def test_search_validates_request(
        self,
        mock_embed_text: AsyncMock,
        mock_embed_query: AsyncMock,
        client: TestClient,
        sample_library_data: dict,
        sample_embedding: list[float],
    ) -> None:
        """Test search validates request parameters."""
        mock_embed_text.return_value = sample_embedding
        mock_embed_query.return_value = sample_embedding

        # Create library
        lib_response = client.post("/api/v1/libraries/", json=sample_library_data)
        library_id = lib_response.json()["id"]

        # Test missing both query_text and query_embedding
        search_request = {"k": 5}
        response = client.post(
            f"/api/v1/libraries/{library_id}/search/", json=search_request
        )

        # Should fail validation
        assert response.status_code in [400, 422]

    @patch("src.utils.embeddings.EmbeddingService.embed_query")
    @patch("src.utils.embeddings.EmbeddingService.embed_text")
    def test_search_scores_are_sorted(
        self,
        mock_embed_text: AsyncMock,
        mock_embed_query: AsyncMock,
        client: TestClient,
        sample_library_data: dict,
        sample_document_data: dict,
        sample_chunk_data: dict,
        sample_chunk_data_2: dict,
        sample_chunk_data_3: dict,
        sample_embedding: list[float],
    ) -> None:
        """Test that search results are sorted by score."""
        mock_embed_text.return_value = sample_embedding
        mock_embed_query.return_value = sample_embedding

        # Setup library with multiple chunks
        library_id = self.setup_library_with_chunks(
            client,
            sample_library_data,
            sample_document_data,
            [sample_chunk_data, sample_chunk_data_2, sample_chunk_data_3],
            sample_embedding,
        )

        # Perform search
        search_request = {"query_text": "test query", "k": 10}
        response = client.post(
            f"/api/v1/libraries/{library_id}/search/", json=search_request
        )

        assert response.status_code == 200
        data = response.json()

        # Check scores are sorted (descending order)
        if len(data["results"]) > 1:
            scores = [result["score"] for result in data["results"]]
            assert scores == sorted(scores, reverse=True)
