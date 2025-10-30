"""Integration tests for library endpoints."""

import pytest
from fastapi.testclient import TestClient
from uuid import UUID


@pytest.mark.integration
class TestLibraryEndpoints:
    """Tests for library API endpoints."""

    def test_create_library(self, client: TestClient, sample_library_data: dict) -> None:
        """Test creating a library."""
        response = client.post("/api/v1/libraries/", json=sample_library_data)

        assert response.status_code == 201
        data = response.json()
        assert "id" in data
        assert data["name"] == sample_library_data["name"]
        assert data["description"] == sample_library_data["description"]
        assert data["index_type"] == sample_library_data["index_type"]
        assert "created_at" in data
        assert "updated_at" in data

    def test_create_library_with_hnsw(
        self, client: TestClient, sample_library_hnsw_data: dict
    ) -> None:
        """Test creating a library with HNSW index."""
        response = client.post("/api/v1/libraries/", json=sample_library_hnsw_data)

        assert response.status_code == 201
        data = response.json()
        assert data["index_type"] == "hnsw"

    def test_create_library_with_lsh(
        self, client: TestClient, sample_library_lsh_data: dict
    ) -> None:
        """Test creating a library with LSH index."""
        response = client.post("/api/v1/libraries/", json=sample_library_lsh_data)

        assert response.status_code == 201
        data = response.json()
        assert data["index_type"] == "lsh"

    def test_create_library_invalid_index_type(
        self, client: TestClient, sample_library_data: dict
    ) -> None:
        """Test creating a library with invalid index type."""
        invalid_data = sample_library_data.copy()
        invalid_data["index_type"] = "invalid_index"

        response = client.post("/api/v1/libraries/", json=invalid_data)

        assert response.status_code == 422  # Validation error

    def test_list_libraries_empty(self, client: TestClient) -> None:
        """Test listing libraries when none exist."""
        response = client.get("/api/v1/libraries/")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_list_libraries(self, client: TestClient, sample_library_data: dict) -> None:
        """Test listing libraries."""
        # Create a library first
        create_response = client.post("/api/v1/libraries/", json=sample_library_data)
        assert create_response.status_code == 201

        # List libraries
        response = client.get("/api/v1/libraries/")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1
        assert any(lib["name"] == sample_library_data["name"] for lib in data)

    def test_list_libraries_multiple(
        self,
        client: TestClient,
        sample_library_data: dict,
        sample_library_hnsw_data: dict,
    ) -> None:
        """Test listing multiple libraries."""
        # Create multiple libraries
        client.post("/api/v1/libraries/", json=sample_library_data)
        client.post("/api/v1/libraries/", json=sample_library_hnsw_data)

        # List libraries
        response = client.get("/api/v1/libraries/")

        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 2

    def test_get_library(self, client: TestClient, sample_library_data: dict) -> None:
        """Test getting a library by ID."""
        # Create a library first
        create_response = client.post("/api/v1/libraries/", json=sample_library_data)
        library_id = create_response.json()["id"]

        # Get the library
        response = client.get(f"/api/v1/libraries/{library_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == library_id
        assert data["name"] == sample_library_data["name"]

    def test_get_library_not_found(self, client: TestClient) -> None:
        """Test getting a non-existent library."""
        fake_id = "00000000-0000-0000-0000-000000000000"
        response = client.get(f"/api/v1/libraries/{fake_id}")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_get_library_invalid_id(self, client: TestClient) -> None:
        """Test getting a library with invalid ID format."""
        response = client.get("/api/v1/libraries/invalid-id")

        assert response.status_code == 422  # Validation error

    def test_update_library(self, client: TestClient, sample_library_data: dict) -> None:
        """Test updating a library."""
        # Create a library first
        create_response = client.post("/api/v1/libraries/", json=sample_library_data)
        library_id = create_response.json()["id"]

        # Update the library
        update_data = {
            "name": "Updated Library Name",
            "description": "Updated description",
        }
        response = client.put(f"/api/v1/libraries/{library_id}", json=update_data)

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == update_data["name"]
        assert data["description"] == update_data["description"]
        assert data["id"] == library_id

    def test_update_library_partial(
        self, client: TestClient, sample_library_data: dict
    ) -> None:
        """Test partial update of a library."""
        # Create a library
        create_response = client.post("/api/v1/libraries/", json=sample_library_data)
        library_id = create_response.json()["id"]
        original_name = create_response.json()["name"]

        # Update only description
        update_data = {"description": "New description only"}
        response = client.put(f"/api/v1/libraries/{library_id}", json=update_data)

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == original_name  # Name unchanged
        assert data["description"] == update_data["description"]

    def test_update_library_not_found(self, client: TestClient) -> None:
        """Test updating a non-existent library."""
        fake_id = "00000000-0000-0000-0000-000000000000"
        update_data = {"name": "Updated Name"}
        response = client.put(f"/api/v1/libraries/{fake_id}", json=update_data)

        assert response.status_code == 404

    def test_delete_library(self, client: TestClient, sample_library_data: dict) -> None:
        """Test deleting a library."""
        # Create a library
        create_response = client.post("/api/v1/libraries/", json=sample_library_data)
        library_id = create_response.json()["id"]

        # Delete the library
        response = client.delete(f"/api/v1/libraries/{library_id}")

        assert response.status_code == 204

        # Verify it's deleted
        get_response = client.get(f"/api/v1/libraries/{library_id}")
        assert get_response.status_code == 404

    def test_delete_library_not_found(self, client: TestClient) -> None:
        """Test deleting a non-existent library."""
        fake_id = "00000000-0000-0000-0000-000000000000"
        response = client.delete(f"/api/v1/libraries/{fake_id}")

        assert response.status_code == 404

    def test_index_library_without_chunks(
        self, client: TestClient, sample_library_data: dict
    ) -> None:
        """Test building index for a library without chunks."""
        # Create a library
        create_response = client.post("/api/v1/libraries/", json=sample_library_data)
        library_id = create_response.json()["id"]

        # Build index
        response = client.post(f"/api/v1/libraries/{library_id}/index")

        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert data["library_id"] == library_id

    def test_index_library_not_found(self, client: TestClient) -> None:
        """Test building index for non-existent library."""
        fake_id = "00000000-0000-0000-0000-000000000000"
        response = client.post(f"/api/v1/libraries/{fake_id}/index")

        assert response.status_code == 404

    def test_library_lifecycle(
        self, client: TestClient, sample_library_data: dict
    ) -> None:
        """Test complete library lifecycle: create, read, update, delete."""
        # Create
        create_response = client.post("/api/v1/libraries/", json=sample_library_data)
        assert create_response.status_code == 201
        library_id = create_response.json()["id"]

        # Read
        get_response = client.get(f"/api/v1/libraries/{library_id}")
        assert get_response.status_code == 200
        assert get_response.json()["name"] == sample_library_data["name"]

        # Update
        update_data = {"name": "Updated Name"}
        update_response = client.put(
            f"/api/v1/libraries/{library_id}", json=update_data
        )
        assert update_response.status_code == 200
        assert update_response.json()["name"] == "Updated Name"

        # Delete
        delete_response = client.delete(f"/api/v1/libraries/{library_id}")
        assert delete_response.status_code == 204

        # Verify deletion
        get_after_delete = client.get(f"/api/v1/libraries/{library_id}")
        assert get_after_delete.status_code == 404
