"""Integration tests for chunk endpoints."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock


@pytest.mark.integration
class TestChunkEndpoints:
    """Tests for chunk API endpoints."""

    def setup_library_and_document(
        self, client: TestClient, sample_library_data: dict, sample_document_data: dict
    ) -> tuple[str, str]:
        """Helper to create library and document for testing.

        Args:
            client: Test client
            sample_library_data: Library data
            sample_document_data: Document data

        Returns:
            Tuple of (library_id, document_id)
        """
        lib_response = client.post("/api/v1/libraries/", json=sample_library_data)
        library_id = lib_response.json()["id"]

        doc_response = client.post(
            f"/api/v1/libraries/{library_id}/documents/", json=sample_document_data
        )
        document_id = doc_response.json()["id"]

        return library_id, document_id

    @patch("src.utils.embeddings.EmbeddingService.embed_text")
    def test_create_chunk(
        self,
        mock_embed_text: AsyncMock,
        client: TestClient,
        sample_library_data: dict,
        sample_document_data: dict,
        sample_chunk_data: dict,
        sample_embedding: list[float],
    ) -> None:
        """Test creating a chunk in a document."""
        # Mock embedding service
        mock_embed_text.return_value = sample_embedding

        # Setup
        _, document_id = self.setup_library_and_document(
            client, sample_library_data, sample_document_data
        )

        # Create chunk
        response = client.post(
            f"/api/v1/documents/{document_id}/chunks/", json=sample_chunk_data
        )

        assert response.status_code == 201
        data = response.json()
        assert "id" in data
        assert data["content"] == sample_chunk_data["content"]
        assert data["document_id"] == document_id
        assert "embedding" in data
        assert len(data["embedding"]) > 0
        assert "created_at" in data

    def test_create_chunk_document_not_found(
        self, client: TestClient, sample_chunk_data: dict
    ) -> None:
        """Test creating a chunk in non-existent document."""
        fake_document_id = "00000000-0000-0000-0000-000000000000"
        response = client.post(
            f"/api/v1/documents/{fake_document_id}/chunks/", json=sample_chunk_data
        )

        assert response.status_code == 404

    @patch("src.utils.embeddings.EmbeddingService.embed_text")
    def test_list_chunks_empty(
        self,
        mock_embed_text: AsyncMock,
        client: TestClient,
        sample_library_data: dict,
        sample_document_data: dict,
        sample_embedding: list[float],
    ) -> None:
        """Test listing chunks when none exist."""
        mock_embed_text.return_value = sample_embedding

        _, document_id = self.setup_library_and_document(
            client, sample_library_data, sample_document_data
        )

        response = client.get(f"/api/v1/documents/{document_id}/chunks/")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 0

    @patch("src.utils.embeddings.EmbeddingService.embed_text")
    def test_list_chunks(
        self,
        mock_embed_text: AsyncMock,
        client: TestClient,
        sample_library_data: dict,
        sample_document_data: dict,
        sample_chunk_data: dict,
        sample_chunk_data_2: dict,
        sample_embedding: list[float],
    ) -> None:
        """Test listing chunks in a document."""
        mock_embed_text.return_value = sample_embedding

        _, document_id = self.setup_library_and_document(
            client, sample_library_data, sample_document_data
        )

        # Create chunks
        client.post(f"/api/v1/documents/{document_id}/chunks/", json=sample_chunk_data)
        client.post(f"/api/v1/documents/{document_id}/chunks/", json=sample_chunk_data_2)

        # List chunks
        response = client.get(f"/api/v1/documents/{document_id}/chunks/")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 2
        assert any(chunk["content"] == sample_chunk_data["content"] for chunk in data)
        assert any(chunk["content"] == sample_chunk_data_2["content"] for chunk in data)

    @patch("src.utils.embeddings.EmbeddingService.embed_text")
    def test_get_chunk(
        self,
        mock_embed_text: AsyncMock,
        client: TestClient,
        sample_library_data: dict,
        sample_document_data: dict,
        sample_chunk_data: dict,
        sample_embedding: list[float],
    ) -> None:
        """Test getting a chunk by ID."""
        mock_embed_text.return_value = sample_embedding

        _, document_id = self.setup_library_and_document(
            client, sample_library_data, sample_document_data
        )

        # Create chunk
        create_response = client.post(
            f"/api/v1/documents/{document_id}/chunks/", json=sample_chunk_data
        )
        chunk_id = create_response.json()["id"]

        # Get chunk
        response = client.get(f"/api/v1/documents/{document_id}/chunks/{chunk_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == chunk_id
        assert data["content"] == sample_chunk_data["content"]
        assert "embedding" in data

    @patch("src.utils.embeddings.EmbeddingService.embed_text")
    def test_get_chunk_not_found(
        self,
        mock_embed_text: AsyncMock,
        client: TestClient,
        sample_library_data: dict,
        sample_document_data: dict,
        sample_embedding: list[float],
    ) -> None:
        """Test getting a non-existent chunk."""
        mock_embed_text.return_value = sample_embedding

        _, document_id = self.setup_library_and_document(
            client, sample_library_data, sample_document_data
        )

        fake_chunk_id = "00000000-0000-0000-0000-000000000000"
        response = client.get(f"/api/v1/documents/{document_id}/chunks/{fake_chunk_id}")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    @patch("src.utils.embeddings.EmbeddingService.embed_text")
    def test_get_chunk_invalid_id(
        self,
        mock_embed_text: AsyncMock,
        client: TestClient,
        sample_library_data: dict,
        sample_document_data: dict,
        sample_embedding: list[float],
    ) -> None:
        """Test getting a chunk with invalid ID format."""
        mock_embed_text.return_value = sample_embedding

        _, document_id = self.setup_library_and_document(
            client, sample_library_data, sample_document_data
        )

        response = client.get(f"/api/v1/documents/{document_id}/chunks/invalid-id")

        assert response.status_code == 422  # Validation error

    @patch("src.utils.embeddings.EmbeddingService.embed_text")
    def test_update_chunk(
        self,
        mock_embed_text: AsyncMock,
        client: TestClient,
        sample_library_data: dict,
        sample_document_data: dict,
        sample_chunk_data: dict,
        sample_embedding: list[float],
    ) -> None:
        """Test updating a chunk (content changes, embedding regenerates)."""
        mock_embed_text.return_value = sample_embedding

        _, document_id = self.setup_library_and_document(
            client, sample_library_data, sample_document_data
        )

        # Create chunk
        create_response = client.post(
            f"/api/v1/documents/{document_id}/chunks/", json=sample_chunk_data
        )
        chunk_id = create_response.json()["id"]

        # Update chunk
        update_data = {
            "content": "Updated chunk content",
            "metadata": {"page": 2, "updated": True},
        }
        response = client.put(
            f"/api/v1/documents/{document_id}/chunks/{chunk_id}", json=update_data
        )

        assert response.status_code == 200
        data = response.json()
        assert data["content"] == update_data["content"]
        assert data["metadata"]["page"] == 2
        assert data["metadata"]["updated"] is True

    @patch("src.utils.embeddings.EmbeddingService.embed_text")
    def test_update_chunk_partial(
        self,
        mock_embed_text: AsyncMock,
        client: TestClient,
        sample_library_data: dict,
        sample_document_data: dict,
        sample_chunk_data: dict,
        sample_embedding: list[float],
    ) -> None:
        """Test partial update of a chunk (only metadata)."""
        mock_embed_text.return_value = sample_embedding

        _, document_id = self.setup_library_and_document(
            client, sample_library_data, sample_document_data
        )

        # Create chunk
        create_response = client.post(
            f"/api/v1/documents/{document_id}/chunks/", json=sample_chunk_data
        )
        chunk_id = create_response.json()["id"]
        original_content = create_response.json()["content"]

        # Update only metadata
        update_data = {"metadata": {"new_field": "new_value"}}
        response = client.put(
            f"/api/v1/documents/{document_id}/chunks/{chunk_id}", json=update_data
        )

        assert response.status_code == 200
        data = response.json()
        assert data["content"] == original_content  # Content unchanged

    @patch("src.utils.embeddings.EmbeddingService.embed_text")
    def test_update_chunk_not_found(
        self,
        mock_embed_text: AsyncMock,
        client: TestClient,
        sample_library_data: dict,
        sample_document_data: dict,
        sample_embedding: list[float],
    ) -> None:
        """Test updating a non-existent chunk."""
        mock_embed_text.return_value = sample_embedding

        _, document_id = self.setup_library_and_document(
            client, sample_library_data, sample_document_data
        )

        fake_chunk_id = "00000000-0000-0000-0000-000000000000"
        update_data = {"content": "Updated content"}
        response = client.put(
            f"/api/v1/documents/{document_id}/chunks/{fake_chunk_id}", json=update_data
        )

        assert response.status_code == 404

    @patch("src.utils.embeddings.EmbeddingService.embed_text")
    def test_delete_chunk(
        self,
        mock_embed_text: AsyncMock,
        client: TestClient,
        sample_library_data: dict,
        sample_document_data: dict,
        sample_chunk_data: dict,
        sample_embedding: list[float],
    ) -> None:
        """Test deleting a chunk."""
        mock_embed_text.return_value = sample_embedding

        _, document_id = self.setup_library_and_document(
            client, sample_library_data, sample_document_data
        )

        # Create chunk
        create_response = client.post(
            f"/api/v1/documents/{document_id}/chunks/", json=sample_chunk_data
        )
        chunk_id = create_response.json()["id"]

        # Delete chunk
        response = client.delete(f"/api/v1/documents/{document_id}/chunks/{chunk_id}")

        assert response.status_code == 204

        # Verify deletion
        get_response = client.get(f"/api/v1/documents/{document_id}/chunks/{chunk_id}")
        assert get_response.status_code == 404

    @patch("src.utils.embeddings.EmbeddingService.embed_text")
    def test_delete_chunk_not_found(
        self,
        mock_embed_text: AsyncMock,
        client: TestClient,
        sample_library_data: dict,
        sample_document_data: dict,
        sample_embedding: list[float],
    ) -> None:
        """Test deleting a non-existent chunk."""
        mock_embed_text.return_value = sample_embedding

        _, document_id = self.setup_library_and_document(
            client, sample_library_data, sample_document_data
        )

        fake_chunk_id = "00000000-0000-0000-0000-000000000000"
        response = client.delete(
            f"/api/v1/documents/{document_id}/chunks/{fake_chunk_id}"
        )

        assert response.status_code == 404

    @patch("src.utils.embeddings.EmbeddingService.embed_text")
    def test_chunk_lifecycle(
        self,
        mock_embed_text: AsyncMock,
        client: TestClient,
        sample_library_data: dict,
        sample_document_data: dict,
        sample_chunk_data: dict,
        sample_embedding: list[float],
    ) -> None:
        """Test complete chunk lifecycle: create, read, update, delete."""
        mock_embed_text.return_value = sample_embedding

        _, document_id = self.setup_library_and_document(
            client, sample_library_data, sample_document_data
        )

        # Create chunk
        create_response = client.post(
            f"/api/v1/documents/{document_id}/chunks/", json=sample_chunk_data
        )
        assert create_response.status_code == 201
        chunk_id = create_response.json()["id"]

        # Read chunk
        get_response = client.get(f"/api/v1/documents/{document_id}/chunks/{chunk_id}")
        assert get_response.status_code == 200
        assert get_response.json()["content"] == sample_chunk_data["content"]

        # Update chunk
        update_data = {"content": "Updated chunk content"}
        update_response = client.put(
            f"/api/v1/documents/{document_id}/chunks/{chunk_id}", json=update_data
        )
        assert update_response.status_code == 200
        assert update_response.json()["content"] == "Updated chunk content"

        # Delete chunk
        delete_response = client.delete(
            f"/api/v1/documents/{document_id}/chunks/{chunk_id}"
        )
        assert delete_response.status_code == 204

        # Verify deletion
        get_after_delete = client.get(
            f"/api/v1/documents/{document_id}/chunks/{chunk_id}"
        )
        assert get_after_delete.status_code == 404

    @patch("src.utils.embeddings.EmbeddingService.embed_text")
    def test_multiple_chunks_in_document(
        self,
        mock_embed_text: AsyncMock,
        client: TestClient,
        sample_library_data: dict,
        sample_document_data: dict,
        sample_chunk_data: dict,
        sample_chunk_data_2: dict,
        sample_chunk_data_3: dict,
        sample_embedding: list[float],
    ) -> None:
        """Test managing multiple chunks in a single document."""
        mock_embed_text.return_value = sample_embedding

        _, document_id = self.setup_library_and_document(
            client, sample_library_data, sample_document_data
        )

        # Create multiple chunks
        chunk1_response = client.post(
            f"/api/v1/documents/{document_id}/chunks/", json=sample_chunk_data
        )
        chunk2_response = client.post(
            f"/api/v1/documents/{document_id}/chunks/", json=sample_chunk_data_2
        )
        chunk3_response = client.post(
            f"/api/v1/documents/{document_id}/chunks/", json=sample_chunk_data_3
        )

        assert chunk1_response.status_code == 201
        assert chunk2_response.status_code == 201
        assert chunk3_response.status_code == 201

        # List chunks
        list_response = client.get(f"/api/v1/documents/{document_id}/chunks/")
        assert list_response.status_code == 200
        chunks = list_response.json()
        assert len(chunks) == 3

        # Delete one chunk
        chunk1_id = chunk1_response.json()["id"]
        delete_response = client.delete(
            f"/api/v1/documents/{document_id}/chunks/{chunk1_id}"
        )
        assert delete_response.status_code == 204

        # Verify only two chunks remain
        list_after_delete = client.get(f"/api/v1/documents/{document_id}/chunks/")
        assert len(list_after_delete.json()) == 2
