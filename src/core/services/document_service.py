"""Document service for business logic."""

from __future__ import annotations

from typing import Any, Optional
from uuid import UUID

from src.domain.models.document import Document
from src.infrastructure.repositories.document_repository import DocumentRepository
from src.infrastructure.repositories.library_repository import LibraryRepository
from src.schemas.document import DocumentCreate, DocumentUpdate


class DocumentService:
    """Service for managing documents."""

    def __init__(
        self,
        repository: DocumentRepository,
        library_repository: LibraryRepository,
    ) -> None:
        """Initialize document service.

        Args:
            repository: Document repository
            library_repository: Library repository
        """
        self.repository = repository
        self.library_repository = library_repository

    def create_document(self, library_id: UUID, data: DocumentCreate) -> Document:
        """Create a new document in a library.

        Args:
            library_id: Library ID
            data: Document creation data

        Returns:
            Created document

        Raises:
            LibraryNotFoundError: If library not found
        """
        from src.core.exceptions import LibraryNotFoundError

        # Check if library exists
        library = self.library_repository.get(library_id)
        if not library:
            raise LibraryNotFoundError(str(library_id))

        # Create document
        document = Document(
            name=data.name,
            metadata=data.metadata,
            library_id=library_id,
        )
        created_document = self.repository.create(document)

        # Update library's document_ids
        updated_document_ids = library.document_ids + [created_document.id]
        self.library_repository.update(library_id, {"document_ids": updated_document_ids})

        return created_document

    def get_document(self, document_id: UUID) -> Document:
        """Get a document by ID.

        Args:
            document_id: Document ID

        Returns:
            Document

        Raises:
            DocumentNotFoundError: If document not found
        """
        from src.core.exceptions import DocumentNotFoundError

        document = self.repository.get(document_id)
        if not document:
            raise DocumentNotFoundError(str(document_id))
        return document

    def list_documents(
        self,
        library_id: Optional[UUID] = None,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[Document]:
        """List documents.

        Args:
            library_id: Optional library ID to filter by
            filters: Optional filters

        Returns:
            List of documents
        """
        if library_id:
            return self.repository.list_by_library(library_id)

        return self.repository.list(filters)

    def update_document(self, document_id: UUID, data: DocumentUpdate) -> Document:
        """Update a document.

        Args:
            document_id: Document ID
            data: Update data

        Returns:
            Updated document

        Raises:
            DocumentNotFoundError: If document not found
        """
        from src.core.exceptions import DocumentNotFoundError

        # Filter out None values
        update_data = {k: v for k, v in data.model_dump().items() if v is not None}

        # If nothing to update, just return the existing document
        if not update_data:
            return self.get_document(document_id)

        updated_document = self.repository.update(document_id, update_data)
        if not updated_document:
            raise DocumentNotFoundError(str(document_id))
        return updated_document

    def delete_document(self, document_id: UUID) -> None:
        """Delete a document.

        Args:
            document_id: Document ID

        Raises:
            DocumentNotFoundError: If document not found
        """
        from src.core.exceptions import DocumentNotFoundError

        # Get document first to access library_id
        document = self.get_document(document_id)

        # Delete document
        success = self.repository.delete(document_id)
        if not success:
            raise DocumentNotFoundError(str(document_id))

        # Update library's document_ids
        library = self.library_repository.get(document.library_id)
        if library:
            updated_document_ids = [
                doc_id for doc_id in library.document_ids if doc_id != document_id
            ]
            self.library_repository.update(document.library_id, {"document_ids": updated_document_ids})
