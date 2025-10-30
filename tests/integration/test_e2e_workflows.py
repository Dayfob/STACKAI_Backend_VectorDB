"""End-to-end integration tests for complete workflows."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock, MagicMock


@pytest.mark.integration
class TestEndToEndWorkflows:
    """End-to-end tests for complete application workflows."""

    @patch("src.utils.embeddings.EmbeddingService.embed_query")
    @patch("src.utils.embeddings.EmbeddingService.embed_text")
    def test_complete_vector_search_workflow(
        self,
        mock_embed_text: AsyncMock,
        mock_embed_query: AsyncMock,
        client: TestClient,
        sample_embedding: list[float],
    ) -> None:
        """Test complete workflow: create library, add documents, chunks, build index, search."""
        mock_embed_text.return_value = sample_embedding
        mock_embed_query.return_value = sample_embedding

        # Step 1: Create a library
        library_data = {
            "name": "AI Research Library",
            "description": "Collection of AI research papers",
            "index_type": "brute_force",
        }
        lib_response = client.post("/api/v1/libraries/", json=library_data)
        assert lib_response.status_code == 201
        library_id = lib_response.json()["id"]

        # Step 2: Create a document
        document_data = {
            "name": "Machine Learning Basics",
            "metadata": {"author": "John Doe", "year": 2024},
        }
        doc_response = client.post(
            f"/api/v1/libraries/{library_id}/documents/", json=document_data
        )
        assert doc_response.status_code == 201
        document_id = doc_response.json()["id"]

        # Step 3: Add multiple chunks
        chunks = [
            {
                "content": "Machine learning is a subset of artificial intelligence.",
                "metadata": {"page": 1},
            },
            {
                "content": "Deep learning uses neural networks with multiple layers.",
                "metadata": {"page": 2},
            },
            {
                "content": "Supervised learning requires labeled training data.",
                "metadata": {"page": 3},
            },
        ]

        for chunk_data in chunks:
            chunk_response = client.post(
                f"/api/v1/documents/{document_id}/chunks/", json=chunk_data
            )
            assert chunk_response.status_code == 201

        # Step 4: Build the index
        index_response = client.post(f"/api/v1/libraries/{library_id}/index")
        assert index_response.status_code == 200

        # Step 5: Perform search
        search_request = {"query_text": "What is deep learning?", "k": 3}
        search_response = client.post(
            f"/api/v1/libraries/{library_id}/search/", json=search_request
        )
        assert search_response.status_code == 200

        search_data = search_response.json()
        assert "results" in search_data
        assert search_data["total"] > 0
        assert len(search_data["results"]) > 0

        # Step 6: Verify search results contain expected data
        result = search_data["results"][0]
        assert "chunk_id" in result
        assert "document_id" in result
        assert "content" in result
        assert "score" in result
        assert "metadata" in result

    @patch("src.utils.embeddings.EmbeddingService.embed_query")
    @patch("src.utils.embeddings.EmbeddingService.embed_text")
    def test_multi_document_library_workflow(
        self,
        mock_embed_text: AsyncMock,
        mock_embed_query: AsyncMock,
        client: TestClient,
        sample_embedding: list[float],
    ) -> None:
        """Test workflow with multiple documents in one library."""
        mock_embed_text.return_value = sample_embedding
        mock_embed_query.return_value = sample_embedding

        # Create library
        library_data = {
            "name": "Multi-Document Library",
            "description": "Testing multiple documents",
            "index_type": "hnsw",
        }
        lib_response = client.post("/api/v1/libraries/", json=library_data)
        library_id = lib_response.json()["id"]

        # Create multiple documents
        documents = [
            {"name": "Document 1", "metadata": {"topic": "AI"}},
            {"name": "Document 2", "metadata": {"topic": "ML"}},
            {"name": "Document 3", "metadata": {"topic": "DL"}},
        ]

        document_ids = []
        for doc_data in documents:
            doc_response = client.post(
                f"/api/v1/libraries/{library_id}/documents/", json=doc_data
            )
            assert doc_response.status_code == 201
            document_ids.append(doc_response.json()["id"])

        # Add chunks to each document
        for doc_id in document_ids:
            chunk_data = {
                "content": f"Content for document {doc_id}",
                "metadata": {"doc_id": doc_id},
            }
            chunk_response = client.post(
                f"/api/v1/documents/{doc_id}/chunks/", json=chunk_data
            )
            assert chunk_response.status_code == 201

        # List all documents
        list_response = client.get(f"/api/v1/libraries/{library_id}/documents/")
        assert list_response.status_code == 200
        assert len(list_response.json()) == 3

        # Build index
        index_response = client.post(f"/api/v1/libraries/{library_id}/index")
        assert index_response.status_code == 200

        # Search across all documents
        search_request = {"query_text": "content", "k": 5}
        search_response = client.post(
            f"/api/v1/libraries/{library_id}/search/", json=search_request
        )
        assert search_response.status_code == 200

    @patch("src.utils.embeddings.EmbeddingService.embed_query")
    @patch("src.utils.embeddings.EmbeddingService.embed_text")
    def test_update_and_reindex_workflow(
        self,
        mock_embed_text: AsyncMock,
        mock_embed_query: AsyncMock,
        client: TestClient,
        sample_embedding: list[float],
    ) -> None:
        """Test workflow with updates and reindexing."""
        mock_embed_text.return_value = sample_embedding
        mock_embed_query.return_value = sample_embedding

        # Create library and document
        lib_response = client.post(
            "/api/v1/libraries/",
            json={
                "name": "Update Test Library",
                "description": "Testing updates",
                "index_type": "brute_force",
            },
        )
        library_id = lib_response.json()["id"]

        doc_response = client.post(
            f"/api/v1/libraries/{library_id}/documents/",
            json={"name": "Test Document", "metadata": {}},
        )
        document_id = doc_response.json()["id"]

        # Add initial chunk
        chunk_response = client.post(
            f"/api/v1/documents/{document_id}/chunks/",
            json={"content": "Initial content", "metadata": {}},
        )
        chunk_id = chunk_response.json()["id"]

        # Build index
        client.post(f"/api/v1/libraries/{library_id}/index")

        # Update chunk
        update_response = client.put(
            f"/api/v1/documents/{document_id}/chunks/{chunk_id}",
            json={"content": "Updated content", "metadata": {}},
        )
        assert update_response.status_code == 200

        # Rebuild index
        reindex_response = client.post(f"/api/v1/libraries/{library_id}/index")
        assert reindex_response.status_code == 200

        # Search should work with updated content
        search_response = client.post(
            f"/api/v1/libraries/{library_id}/search/",
            json={"query_text": "content", "k": 5},
        )
        assert search_response.status_code == 200

    @patch("src.utils.embeddings.EmbeddingService.embed_query")
    @patch("src.utils.embeddings.EmbeddingService.embed_text")
    def test_delete_and_recreate_workflow(
        self,
        mock_embed_text: AsyncMock,
        mock_embed_query: AsyncMock,
        client: TestClient,
        sample_embedding: list[float],
    ) -> None:
        """Test workflow with deletions and recreation."""
        mock_embed_text.return_value = sample_embedding
        mock_embed_query.return_value = sample_embedding

        # Create library
        lib_response = client.post(
            "/api/v1/libraries/",
            json={
                "name": "Delete Test Library",
                "description": "Testing deletions",
                "index_type": "brute_force",
            },
        )
        library_id = lib_response.json()["id"]

        # Create document with chunks
        doc_response = client.post(
            f"/api/v1/libraries/{library_id}/documents/",
            json={"name": "Test Document", "metadata": {}},
        )
        document_id = doc_response.json()["id"]

        chunk_response = client.post(
            f"/api/v1/documents/{document_id}/chunks/",
            json={"content": "Test content", "metadata": {}},
        )
        chunk_id = chunk_response.json()["id"]

        # Delete chunk
        delete_chunk_response = client.delete(
            f"/api/v1/documents/{document_id}/chunks/{chunk_id}"
        )
        assert delete_chunk_response.status_code == 204

        # Verify chunk is deleted
        get_chunk_response = client.get(
            f"/api/v1/documents/{document_id}/chunks/{chunk_id}"
        )
        assert get_chunk_response.status_code == 404

        # Delete document
        delete_doc_response = client.delete(
            f"/api/v1/libraries/{library_id}/documents/{document_id}"
        )
        assert delete_doc_response.status_code == 204

        # Create new document
        new_doc_response = client.post(
            f"/api/v1/libraries/{library_id}/documents/",
            json={"name": "New Document", "metadata": {}},
        )
        assert new_doc_response.status_code == 201

    @patch("src.utils.embeddings.EmbeddingService.embed_query")
    @patch("src.utils.embeddings.EmbeddingService.embed_text")
    def test_multiple_libraries_workflow(
        self,
        mock_embed_text: AsyncMock,
        mock_embed_query: AsyncMock,
        client: TestClient,
        sample_embedding: list[float],
    ) -> None:
        """Test workflow with multiple independent libraries."""
        mock_embed_text.return_value = sample_embedding
        mock_embed_query.return_value = sample_embedding

        # Create multiple libraries with different index types
        library_types = [
            {"name": "Brute Force Library", "index_type": "brute_force"},
            {"name": "HNSW Library", "index_type": "hnsw"},
            {"name": "LSH Library", "index_type": "lsh"},
        ]

        library_ids = []
        for lib_data in library_types:
            lib_response = client.post(
                "/api/v1/libraries/",
                json={**lib_data, "description": f"Testing {lib_data['index_type']}"},
            )
            assert lib_response.status_code == 201
            library_ids.append(lib_response.json()["id"])

        # Add content to each library
        for library_id in library_ids:
            doc_response = client.post(
                f"/api/v1/libraries/{library_id}/documents/",
                json={"name": f"Document for {library_id}", "metadata": {}},
            )
            document_id = doc_response.json()["id"]

            chunk_response = client.post(
                f"/api/v1/documents/{document_id}/chunks/",
                json={"content": f"Content for {library_id}", "metadata": {}},
            )
            assert chunk_response.status_code == 201

            # Build index
            index_response = client.post(f"/api/v1/libraries/{library_id}/index")
            assert index_response.status_code == 200

        # List all libraries
        list_response = client.get("/api/v1/libraries/")
        assert list_response.status_code == 200
        libraries = list_response.json()
        assert len(libraries) >= 3

        # Search in each library
        for library_id in library_ids:
            search_response = client.post(
                f"/api/v1/libraries/{library_id}/search/",
                json={"query_text": "content", "k": 5},
            )
            assert search_response.status_code == 200

    @patch("src.utils.embeddings.EmbeddingService.embed_query")
    @patch("src.utils.embeddings.EmbeddingService.embed_text")
    def test_large_batch_workflow(
        self,
        mock_embed_text: AsyncMock,
        mock_embed_query: AsyncMock,
        client: TestClient,
        sample_embedding: list[float],
    ) -> None:
        """Test workflow with larger batch of chunks."""
        mock_embed_text.return_value = sample_embedding
        mock_embed_query.return_value = sample_embedding

        # Create library
        lib_response = client.post(
            "/api/v1/libraries/",
            json={
                "name": "Large Batch Library",
                "description": "Testing large batch",
                "index_type": "hnsw",
            },
        )
        library_id = lib_response.json()["id"]

        # Create document
        doc_response = client.post(
            f"/api/v1/libraries/{library_id}/documents/",
            json={"name": "Large Document", "metadata": {}},
        )
        document_id = doc_response.json()["id"]

        # Add multiple chunks
        num_chunks = 20
        for i in range(num_chunks):
            chunk_response = client.post(
                f"/api/v1/documents/{document_id}/chunks/",
                json={
                    "content": f"Chunk number {i} with some content",
                    "metadata": {"chunk_number": i},
                },
            )
            assert chunk_response.status_code == 201

        # List chunks
        list_response = client.get(f"/api/v1/documents/{document_id}/chunks/")
        assert list_response.status_code == 200
        assert len(list_response.json()) == num_chunks

        # Build index
        index_response = client.post(f"/api/v1/libraries/{library_id}/index")
        assert index_response.status_code == 200

        # Search with different k values
        for k in [5, 10, 15]:
            search_response = client.post(
                f"/api/v1/libraries/{library_id}/search/",
                json={"query_text": "chunk content", "k": k},
            )
            assert search_response.status_code == 200
            results = search_response.json()["results"]
            assert len(results) <= k

    @patch("src.utils.embeddings.EmbeddingService.embed_query")
    @patch("src.utils.embeddings.EmbeddingService.embed_text")
    def test_semantic_search_workflow(
        self,
        mock_embed_text: AsyncMock,
        mock_embed_query: AsyncMock,
        client: TestClient,
        sample_embedding: list[float],
    ) -> None:
        """Test semantic search workflow."""
        mock_embed_text.return_value = sample_embedding
        mock_embed_query.return_value = sample_embedding

        # Create library
        lib_response = client.post(
            "/api/v1/libraries/",
            json={
                "name": "Semantic Search Library",
                "description": "Testing semantic search",
                "index_type": "brute_force",
            },
        )
        library_id = lib_response.json()["id"]

        # Create document with chunks
        doc_response = client.post(
            f"/api/v1/libraries/{library_id}/documents/",
            json={"name": "Semantic Test Document", "metadata": {}},
        )
        document_id = doc_response.json()["id"]

        chunks = [
            "Artificial intelligence is transforming industries.",
            "Machine learning models require training data.",
            "Natural language processing enables text understanding.",
        ]

        for content in chunks:
            client.post(
                f"/api/v1/documents/{document_id}/chunks/",
                json={"content": content, "metadata": {}},
            )

        # Build index
        client.post(f"/api/v1/libraries/{library_id}/index")

        # Test semantic search endpoint
        search_response = client.post(
            f"/api/v1/libraries/{library_id}/search/semantic",
            params={"query_text": "AI and ML", "k": 5},
        )
        assert search_response.status_code == 200
        assert "results" in search_response.json()

    @patch("src.utils.embeddings.EmbeddingService.embed_query")
    @patch("src.utils.embeddings.EmbeddingService.embed_text")
    def test_error_recovery_workflow(
        self,
        mock_embed_text: AsyncMock,
        mock_embed_query: AsyncMock,
        client: TestClient,
        sample_embedding: list[float],
    ) -> None:
        """Test workflow with error conditions and recovery."""
        mock_embed_text.return_value = sample_embedding
        mock_embed_query.return_value = sample_embedding

        # Try to create document in non-existent library
        fake_library_id = "00000000-0000-0000-0000-000000000000"
        doc_response = client.post(
            f"/api/v1/libraries/{fake_library_id}/documents/",
            json={"name": "Test", "metadata": {}},
        )
        assert doc_response.status_code == 404

        # Create library properly
        lib_response = client.post(
            "/api/v1/libraries/",
            json={
                "name": "Recovery Test Library",
                "description": "Testing error recovery",
                "index_type": "brute_force",
            },
        )
        library_id = lib_response.json()["id"]

        # Try to search before building index
        search_response = client.post(
            f"/api/v1/libraries/{library_id}/search/",
            json={"query_text": "test", "k": 5},
        )
        # Should return empty results or error
        assert search_response.status_code in [200, 400]

        # Create document and chunk properly
        doc_response = client.post(
            f"/api/v1/libraries/{library_id}/documents/",
            json={"name": "Recovery Document", "metadata": {}},
        )
        document_id = doc_response.json()["id"]

        chunk_response = client.post(
            f"/api/v1/documents/{document_id}/chunks/",
            json={"content": "Recovery content", "metadata": {}},
        )
        assert chunk_response.status_code == 201

        # Build index
        index_response = client.post(f"/api/v1/libraries/{library_id}/index")
        assert index_response.status_code == 200

        # Search should now work
        search_response = client.post(
            f"/api/v1/libraries/{library_id}/search/",
            json={"query_text": "recovery", "k": 5},
        )
        assert search_response.status_code == 200
