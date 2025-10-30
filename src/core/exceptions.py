"""Custom exceptions for the application."""

from __future__ import annotations

from typing import Any, Optional


class VectorDBError(Exception):
    """Base exception for all vector DB errors."""

    def __init__(
        self,
        message: str,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize exception.

        Args:
            message: Error message
            details: Additional error details
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}


class NotFoundError(VectorDBError):
    """Base exception for not found errors."""

    pass


class LibraryNotFoundError(NotFoundError):
    """Raised when a library is not found."""

    def __init__(self, library_id: str) -> None:
        """Initialize exception.

        Args:
            library_id: ID of the library
        """
        super().__init__(
            message=f"Library with ID {library_id} not found",
            details={"library_id": library_id},
        )


class DocumentNotFoundError(NotFoundError):
    """Raised when a document is not found."""

    def __init__(self, document_id: str) -> None:
        """Initialize exception.

        Args:
            document_id: ID of the document
        """
        super().__init__(
            message=f"Document with ID {document_id} not found",
            details={"document_id": document_id},
        )


class ChunkNotFoundError(NotFoundError):
    """Raised when a chunk is not found."""

    def __init__(self, chunk_id: str) -> None:
        """Initialize exception.

        Args:
            chunk_id: ID of the chunk
        """
        super().__init__(
            message=f"Chunk with ID {chunk_id} not found",
            details={"chunk_id": chunk_id},
        )


class ValidationError(VectorDBError):
    """Raised when validation fails."""

    pass


class IndexNotBuiltError(VectorDBError):
    """Raised when trying to search an index that hasn't been built."""

    def __init__(self, library_id: str) -> None:
        """Initialize exception.

        Args:
            library_id: ID of the library
        """
        super().__init__(
            message=f"Index for library {library_id} has not been built. Call index_library() first.",
            details={"library_id": library_id},
        )


class EmbeddingError(VectorDBError):
    """Raised when embedding generation fails."""

    pass
