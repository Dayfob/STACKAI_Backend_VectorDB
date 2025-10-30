"""Integration tests for document endpoints."""

import pytest
from fastapi.testclient import TestClient


@pytest.mark.integration
class TestDocumentEndpoints:
    """Tests for document API endpoints."""

    def test_create_document(
        self, client: TestClient, sample_library_data: dict, sample_document_data: dict
    ) -> None:
        """Test creating a document in a library."""
        # Create library first
        lib_response = client.post("/api/v1/libraries/", json=sample_library_data)
        library_id = lib_response.json()["id"]

        # Create document
        response = client.post(
            f"/api/v1/libraries/{library_id}/documents/", json=sample_document_data
        )

        assert response.status_code == 201
        data = response.json()
        assert "id" in data
        assert data["name"] == sample_document_data["name"]
        assert data["library_id"] == library_id
        assert "created_at" in data
        assert "updated_at" in data

    def test_create_document_library_not_found(
        self, client: TestClient, sample_document_data: dict
    ) -> None:
        """Test creating a document in non-existent library."""
        fake_library_id = "00000000-0000-0000-0000-000000000000"
        response = client.post(
            f"/api/v1/libraries/{fake_library_id}/documents/", json=sample_document_data
        )

        assert response.status_code == 404

    def test_list_documents_empty(
        self, client: TestClient, sample_library_data: dict
    ) -> None:
        """Test listing documents when none exist."""
        # Create library
        lib_response = client.post("/api/v1/libraries/", json=sample_library_data)
        library_id = lib_response.json()["id"]

        # List documents
        response = client.get(f"/api/v1/libraries/{library_id}/documents/")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 0

    def test_list_documents(
        self,
        client: TestClient,
        sample_library_data: dict,
        sample_document_data: dict,
        sample_document_data_2: dict,
    ) -> None:
        """Test listing documents in a library."""
        # Create library
        lib_response = client.post("/api/v1/libraries/", json=sample_library_data)
        library_id = lib_response.json()["id"]

        # Create documents
        client.post(
            f"/api/v1/libraries/{library_id}/documents/", json=sample_document_data
        )
        client.post(
            f"/api/v1/libraries/{library_id}/documents/", json=sample_document_data_2
        )

        # List documents
        response = client.get(f"/api/v1/libraries/{library_id}/documents/")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 2
        assert any(doc["name"] == sample_document_data["name"] for doc in data)
        assert any(doc["name"] == sample_document_data_2["name"] for doc in data)

    def test_get_document(
        self, client: TestClient, sample_library_data: dict, sample_document_data: dict
    ) -> None:
        """Test getting a document by ID."""
        # Create library and document
        lib_response = client.post("/api/v1/libraries/", json=sample_library_data)
        library_id = lib_response.json()["id"]

        doc_response = client.post(
            f"/api/v1/libraries/{library_id}/documents/", json=sample_document_data
        )
        document_id = doc_response.json()["id"]

        # Get document
        response = client.get(
            f"/api/v1/libraries/{library_id}/documents/{document_id}"
        )

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == document_id
        assert data["name"] == sample_document_data["name"]

    def test_get_document_not_found(
        self, client: TestClient, sample_library_data: dict
    ) -> None:
        """Test getting a non-existent document."""
        # Create library
        lib_response = client.post("/api/v1/libraries/", json=sample_library_data)
        library_id = lib_response.json()["id"]

        fake_doc_id = "00000000-0000-0000-0000-000000000000"
        response = client.get(f"/api/v1/libraries/{library_id}/documents/{fake_doc_id}")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_get_document_invalid_id(
        self, client: TestClient, sample_library_data: dict
    ) -> None:
        """Test getting a document with invalid ID format."""
        lib_response = client.post("/api/v1/libraries/", json=sample_library_data)
        library_id = lib_response.json()["id"]

        response = client.get(f"/api/v1/libraries/{library_id}/documents/invalid-id")

        assert response.status_code == 422  # Validation error

    def test_update_document(
        self, client: TestClient, sample_library_data: dict, sample_document_data: dict
    ) -> None:
        """Test updating a document."""
        # Create library and document
        lib_response = client.post("/api/v1/libraries/", json=sample_library_data)
        library_id = lib_response.json()["id"]

        doc_response = client.post(
            f"/api/v1/libraries/{library_id}/documents/", json=sample_document_data
        )
        document_id = doc_response.json()["id"]

        # Update document
        update_data = {
            "name": "Updated Document Name",
            "metadata": {"author": "Updated Author"},
        }
        response = client.put(
            f"/api/v1/libraries/{library_id}/documents/{document_id}", json=update_data
        )

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == update_data["name"]
        assert data["metadata"]["author"] == update_data["metadata"]["author"]

    def test_update_document_partial(
        self, client: TestClient, sample_library_data: dict, sample_document_data: dict
    ) -> None:
        """Test partial update of a document."""
        # Create library and document
        lib_response = client.post("/api/v1/libraries/", json=sample_library_data)
        library_id = lib_response.json()["id"]

        doc_response = client.post(
            f"/api/v1/libraries/{library_id}/documents/", json=sample_document_data
        )
        document_id = doc_response.json()["id"]
        original_name = doc_response.json()["name"]

        # Update only metadata
        update_data = {"metadata": {"new_field": "new_value"}}
        response = client.put(
            f"/api/v1/libraries/{library_id}/documents/{document_id}", json=update_data
        )

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == original_name  # Name unchanged

    def test_update_document_not_found(
        self, client: TestClient, sample_library_data: dict
    ) -> None:
        """Test updating a non-existent document."""
        lib_response = client.post("/api/v1/libraries/", json=sample_library_data)
        library_id = lib_response.json()["id"]

        fake_doc_id = "00000000-0000-0000-0000-000000000000"
        update_data = {"name": "Updated Name"}
        response = client.put(
            f"/api/v1/libraries/{library_id}/documents/{fake_doc_id}", json=update_data
        )

        assert response.status_code == 404

    def test_delete_document(
        self, client: TestClient, sample_library_data: dict, sample_document_data: dict
    ) -> None:
        """Test deleting a document."""
        # Create library and document
        lib_response = client.post("/api/v1/libraries/", json=sample_library_data)
        library_id = lib_response.json()["id"]

        doc_response = client.post(
            f"/api/v1/libraries/{library_id}/documents/", json=sample_document_data
        )
        document_id = doc_response.json()["id"]

        # Delete document
        response = client.delete(
            f"/api/v1/libraries/{library_id}/documents/{document_id}"
        )

        assert response.status_code == 204

        # Verify deletion
        get_response = client.get(
            f"/api/v1/libraries/{library_id}/documents/{document_id}"
        )
        assert get_response.status_code == 404

    def test_delete_document_not_found(
        self, client: TestClient, sample_library_data: dict
    ) -> None:
        """Test deleting a non-existent document."""
        lib_response = client.post("/api/v1/libraries/", json=sample_library_data)
        library_id = lib_response.json()["id"]

        fake_doc_id = "00000000-0000-0000-0000-000000000000"
        response = client.delete(
            f"/api/v1/libraries/{library_id}/documents/{fake_doc_id}"
        )

        assert response.status_code == 404

    def test_document_lifecycle(
        self,
        client: TestClient,
        sample_library_data: dict,
        sample_document_data: dict,
    ) -> None:
        """Test complete document lifecycle: create, read, update, delete."""
        # Create library
        lib_response = client.post("/api/v1/libraries/", json=sample_library_data)
        library_id = lib_response.json()["id"]

        # Create document
        create_response = client.post(
            f"/api/v1/libraries/{library_id}/documents/", json=sample_document_data
        )
        assert create_response.status_code == 201
        document_id = create_response.json()["id"]

        # Read document
        get_response = client.get(
            f"/api/v1/libraries/{library_id}/documents/{document_id}"
        )
        assert get_response.status_code == 200
        assert get_response.json()["name"] == sample_document_data["name"]

        # Update document
        update_data = {"name": "Updated Document"}
        update_response = client.put(
            f"/api/v1/libraries/{library_id}/documents/{document_id}", json=update_data
        )
        assert update_response.status_code == 200
        assert update_response.json()["name"] == "Updated Document"

        # Delete document
        delete_response = client.delete(
            f"/api/v1/libraries/{library_id}/documents/{document_id}"
        )
        assert delete_response.status_code == 204

        # Verify deletion
        get_after_delete = client.get(
            f"/api/v1/libraries/{library_id}/documents/{document_id}"
        )
        assert get_after_delete.status_code == 404

    def test_multiple_documents_in_library(
        self,
        client: TestClient,
        sample_library_data: dict,
        sample_document_data: dict,
        sample_document_data_2: dict,
    ) -> None:
        """Test managing multiple documents in a single library."""
        # Create library
        lib_response = client.post("/api/v1/libraries/", json=sample_library_data)
        library_id = lib_response.json()["id"]

        # Create multiple documents
        doc1_response = client.post(
            f"/api/v1/libraries/{library_id}/documents/", json=sample_document_data
        )
        doc2_response = client.post(
            f"/api/v1/libraries/{library_id}/documents/", json=sample_document_data_2
        )

        assert doc1_response.status_code == 201
        assert doc2_response.status_code == 201

        # List documents
        list_response = client.get(f"/api/v1/libraries/{library_id}/documents/")
        assert list_response.status_code == 200
        documents = list_response.json()
        assert len(documents) == 2

        # Delete one document
        doc1_id = doc1_response.json()["id"]
        delete_response = client.delete(
            f"/api/v1/libraries/{library_id}/documents/{doc1_id}"
        )
        assert delete_response.status_code == 204

        # Verify only one document remains
        list_after_delete = client.get(f"/api/v1/libraries/{library_id}/documents/")
        assert len(list_after_delete.json()) == 1
