"""API dependencies for dependency injection."""

from functools import lru_cache

from src.core.config import Settings, get_settings
from src.core.services import ChunkService, DocumentService, LibraryService, SearchService
from src.infrastructure.persistence import DiskStorage, InMemoryStorage, Storage
from src.infrastructure.repositories import (
    ChunkRepository,
    DocumentRepository,
    LibraryRepository,
)
from src.utils.embeddings import EmbeddingService


@lru_cache()
def get_storage() -> Storage:
    """Get storage instance based on settings.

    Returns:
        Storage instance (InMemoryStorage or DiskStorage)
    """
    settings = get_settings()

    if settings.storage_type == "disk":
        return DiskStorage(
            base_path=settings.storage_path,
            format=settings.storage_format,  # type: ignore
        )
    else:
        return InMemoryStorage()


def get_library_repository() -> LibraryRepository:
    """Get library repository.

    Returns:
        Library repository
    """
    return LibraryRepository(storage=get_storage())


def get_document_repository() -> DocumentRepository:
    """Get document repository.

    Returns:
        Document repository
    """
    return DocumentRepository(storage=get_storage())


def get_chunk_repository() -> ChunkRepository:
    """Get chunk repository.

    Returns:
        Chunk repository
    """
    return ChunkRepository(storage=get_storage())


def get_embedding_service() -> EmbeddingService:
    """Get embedding service.

    Returns:
        Embedding service
    """
    settings = get_settings()
    return EmbeddingService(api_key=settings.cohere_api_key)


def get_library_service() -> LibraryService:
    """Get library service.

    Returns:
        Library service
    """
    return LibraryService(repository=get_library_repository())


def get_document_service() -> DocumentService:
    """Get document service.

    Returns:
        Document service
    """
    return DocumentService(
        repository=get_document_repository(),
        library_repository=get_library_repository(),
    )


def get_chunk_service() -> ChunkService:
    """Get chunk service.

    Returns:
        Chunk service
    """
    return ChunkService(
        repository=get_chunk_repository(),
        document_repository=get_document_repository(),
        embedding_service=get_embedding_service(),
    )


def get_search_service() -> SearchService:
    """Get search service.

    Returns:
        Search service
    """
    return SearchService(
        library_repository=get_library_repository(),
        chunk_repository=get_chunk_repository(),
        embedding_service=get_embedding_service(),
    )
