"""Document repository implementation."""

from __future__ import annotations

from typing import Any, Optional
from uuid import UUID

from src.domain.models.document import Document
from src.infrastructure.concurrency.rwlock import RWLock
from src.infrastructure.persistence.storage import Storage
from src.infrastructure.repositories.base import BaseRepository


class DocumentRepository(BaseRepository[Document]):
    """Repository for Document entities with thread-safe operations."""

    def __init__(self, storage: Storage) -> None:
        """Initialize document repository.

        Args:
            storage: Storage backend
        """
        self.storage = storage
        self.lock = RWLock()
        self._storage_key = "documents"

    def create(self, entity: Document) -> Document:
        """Create a new document."""
        with self.lock.writer():
            # Load existing data
            data = self.storage.load(self._storage_key) or {}

            # Add new entity (use mode='json' for JSON-compatible serialization)
            data[str(entity.id)] = entity.model_dump(mode='json')

            # Save back to storage
            self.storage.save(self._storage_key, data)

            return entity

    def get(self, entity_id: UUID) -> Optional[Document]:
        """Get document by ID."""
        with self.lock.reader():
            data = self.storage.load(self._storage_key) or {}
            entity_data = data.get(str(entity_id))

            if entity_data:
                return Document(**entity_data)
            return None

    def list(self, filters: Optional[dict[str, Any]] = None) -> list[Document]:
        """List all documents."""
        with self.lock.reader():
            data = self.storage.load(self._storage_key) or {}
            documents = [Document(**entity_data) for entity_data in data.values()]

            # Apply filters if provided
            if filters:
                filtered = []
                for document in documents:
                    match = True
                    for key, value in filters.items():
                        if not hasattr(document, key) or getattr(document, key) != value:
                            match = False
                            break
                    if match:
                        filtered.append(document)
                return filtered

            return documents

    def update(self, entity_id: UUID, data: dict[str, Any]) -> Optional[Document]:
        """Update a document."""
        with self.lock.writer():
            storage_data = self.storage.load(self._storage_key) or {}
            entity_data = storage_data.get(str(entity_id))

            if not entity_data:
                return None

            # Deserialize to Pydantic model
            entity = Document(**entity_data)

            # Update fields using Pydantic's model_copy with update
            from datetime import datetime
            update_data = {**data, "updated_at": datetime.utcnow()}
            updated_entity = entity.model_copy(update=update_data)

            # Serialize back with JSON-compatible format
            storage_data[str(entity_id)] = updated_entity.model_dump(mode='json')
            self.storage.save(self._storage_key, storage_data)

            return updated_entity

    def delete(self, entity_id: UUID) -> bool:
        """Delete a document."""
        with self.lock.writer():
            data = self.storage.load(self._storage_key) or {}

            if str(entity_id) not in data:
                return False

            del data[str(entity_id)]
            self.storage.save(self._storage_key, data)

            return True

    def exists(self, entity_id: UUID) -> bool:
        """Check if document exists."""
        with self.lock.reader():
            data = self.storage.load(self._storage_key) or {}
            return str(entity_id) in data

    def list_by_library(self, library_id: UUID) -> list[Document]:
        """List all documents in a library."""
        with self.lock.reader():
            data = self.storage.load(self._storage_key) or {}
            documents = []

            for entity_data in data.values():
                document = Document(**entity_data)
                if document.library_id == library_id:
                    documents.append(document)

            return documents
