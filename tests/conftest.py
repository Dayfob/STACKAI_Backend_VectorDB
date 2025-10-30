"""Pytest configuration and fixtures."""

from __future__ import annotations

import os
from typing import Generator
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture(scope="function")
def client() -> Generator[TestClient, None, None]:
    """Create FastAPI test client with clean state for each test.

    Returns:
        Test client
    """
    # Clear any cached dependencies
    app.dependency_overrides = {}

    with TestClient(app) as test_client:
        yield test_client

    # Cleanup after test
    app.dependency_overrides = {}


@pytest.fixture
def sample_embedding() -> list[float]:
    """Create sample embedding vector.

    Returns:
        Sample embedding (1024-dimensional)
    """
    return [0.1 + i * 0.001 for i in range(1024)]


@pytest.fixture
def sample_embedding_small() -> list[float]:
    """Create small sample embedding vector for testing.

    Returns:
        Sample embedding (128-dimensional)
    """
    return [0.1 + i * 0.01 for i in range(128)]


@pytest.fixture
def sample_library_data() -> dict:
    """Create sample library data.

    Returns:
        Sample library data
    """
    return {
        "name": "Test Library",
        "description": "A test library for integration testing",
        "metadata": {"category": "test", "version": "1.0"},
        "index_type": "brute_force",
    }


@pytest.fixture
def sample_library_hnsw_data() -> dict:
    """Create sample library data with HNSW index.

    Returns:
        Sample library data with HNSW
    """
    return {
        "name": "Test Library HNSW",
        "description": "A test library with HNSW index",
        "metadata": {"category": "test"},
        "index_type": "hnsw",
    }


@pytest.fixture
def sample_library_lsh_data() -> dict:
    """Create sample library data with LSH index.

    Returns:
        Sample library data with LSH
    """
    return {
        "name": "Test Library LSH",
        "description": "A test library with LSH index",
        "metadata": {"category": "test"},
        "index_type": "lsh",
    }


@pytest.fixture
def sample_document_data() -> dict:
    """Create sample document data.

    Returns:
        Sample document data
    """
    return {
        "name": "Test Document",
        "metadata": {"author": "Test Author", "year": 2024},
    }


@pytest.fixture
def sample_document_data_2() -> dict:
    """Create second sample document data.

    Returns:
        Sample document data
    """
    return {
        "name": "Another Test Document",
        "metadata": {"author": "Another Author", "year": 2024},
    }


@pytest.fixture
def sample_chunk_data() -> dict:
    """Create sample chunk data.

    Returns:
        Sample chunk data
    """
    return {
        "content": "Machine learning is a subset of artificial intelligence.",
        "metadata": {"page": 1, "section": "introduction"},
    }


@pytest.fixture
def sample_chunk_data_2() -> dict:
    """Create second sample chunk data.

    Returns:
        Sample chunk data
    """
    return {
        "content": "Deep learning uses neural networks with multiple layers.",
        "metadata": {"page": 2, "section": "methods"},
    }


@pytest.fixture
def sample_chunk_data_3() -> dict:
    """Create third sample chunk data.

    Returns:
        Sample chunk data
    """
    return {
        "content": "Natural language processing enables computers to understand human language.",
        "metadata": {"page": 3, "section": "applications"},
    }


@pytest.fixture
def sample_search_request() -> dict:
    """Create sample search request.

    Returns:
        Sample search request
    """
    return {
        "query_text": "What is machine learning?",
        "k": 5,
    }


@pytest.fixture
def mock_embedding_service() -> Generator[MagicMock, None, None]:
    """Mock embedding service to avoid API calls.

    Returns:
        Mocked embedding service
    """
    with patch("src.utils.embeddings.EmbeddingService") as mock:
        instance = mock.return_value

        # Mock the async embed method
        async def mock_embed(text: str) -> list[float]:
            # Return deterministic embeddings based on text length
            base = hash(text) % 100 / 100.0
            return [base + i * 0.001 for i in range(1024)]

        instance.embed = AsyncMock(side_effect=mock_embed)

        # Mock the async embed_batch method
        async def mock_embed_batch(texts: list[str]) -> list[list[float]]:
            return [await mock_embed(text) for text in texts]

        instance.embed_batch = AsyncMock(side_effect=mock_embed_batch)

        yield instance


@pytest.fixture
def mock_cohere_api() -> Generator[None, None, None]:
    """Mock Cohere API to avoid real API calls in tests.

    Yields:
        None
    """
    with patch("httpx.AsyncClient") as mock_client:
        # Create mock response for successful API call
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "embeddings": {
                "float": [
                    [0.1 + i * 0.001 for i in range(1024)],  # Single embedding
                ]
            }
        }
        mock_response.raise_for_status = MagicMock()

        # Configure mock client
        mock_instance = MagicMock()
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=None)
        mock_instance.post = AsyncMock(return_value=mock_response)

        mock_client.return_value = mock_instance

        yield


@pytest.fixture
def test_library_id() -> UUID:
    """Generate test library ID.

    Returns:
        Test UUID
    """
    return uuid4()


@pytest.fixture
def test_document_id() -> UUID:
    """Generate test document ID.

    Returns:
        Test UUID
    """
    return uuid4()


@pytest.fixture
def test_chunk_id() -> UUID:
    """Generate test chunk ID.

    Returns:
        Test UUID
    """
    return uuid4()


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singleton instances between tests.

    This ensures each test starts with a clean state.
    """
    # Reset storage and repository singletons if they exist
    from src.infrastructure.persistence import memory_storage

    # Clear the in-memory storage
    if hasattr(memory_storage, "_storage"):
        memory_storage._storage.clear()

    yield

    # Cleanup after test
    if hasattr(memory_storage, "_storage"):
        memory_storage._storage.clear()


@pytest.fixture
def temp_test_dir(tmp_path):
    """Create temporary directory for test files.

    Args:
        tmp_path: Pytest temporary path fixture

    Returns:
        Path to temporary directory
    """
    test_dir = tmp_path / "test_data"
    test_dir.mkdir(exist_ok=True)
    return test_dir


@pytest.fixture
def disable_cohere_api(monkeypatch):
    """Disable real Cohere API calls by unsetting API key.

    Args:
        monkeypatch: Pytest monkeypatch fixture
    """
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
