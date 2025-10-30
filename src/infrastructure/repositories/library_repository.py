"""Library repository implementation."""

from __future__ import annotations

from typing import Any, Optional
from uuid import UUID

from src.domain.models.library import Library
from src.infrastructure.concurrency.rwlock import RWLock
from src.infrastructure.persistence.storage import Storage
from src.infrastructure.repositories.base import BaseRepository


class LibraryRepository(BaseRepository[Library]):
    """Repository for Library entities with thread-safe operations."""

    def __init__(self, storage: Storage) -> None:
        """Initialize library repository.

        Args:
            storage: Storage backend
        """
        self.storage = storage
        self.lock = RWLock()
        self._storage_key = "libraries"

    def create(self, entity: Library) -> Library:
        """Create a new library."""
        with self.lock.writer():
            # Load existing data
            data = self.storage.load(self._storage_key) or {}

            # Add new entity (use mode='json' for JSON-compatible serialization)
            data[str(entity.id)] = entity.model_dump(mode='json')

            # Save back to storage
            self.storage.save(self._storage_key, data)

            return entity

    def get(self, entity_id: UUID) -> Optional[Library]:
        """Get library by ID."""
        with self.lock.reader():
            data = self.storage.load(self._storage_key) or {}
            entity_data = data.get(str(entity_id))

            if entity_data:
                return Library(**entity_data)
            return None

    def list(self, filters: Optional[dict[str, Any]] = None) -> list[Library]:
        """List all libraries."""
        with self.lock.reader():
            data = self.storage.load(self._storage_key) or {}
            libraries = [Library(**entity_data) for entity_data in data.values()]

            # Apply filters if provided
            if filters:
                filtered = []
                for library in libraries:
                    match = True
                    for key, value in filters.items():
                        if not hasattr(library, key) or getattr(library, key) != value:
                            match = False
                            break
                    if match:
                        filtered.append(library)
                return filtered

            return libraries

    def update(self, entity_id: UUID, data: dict[str, Any]) -> Optional[Library]:
        """Update a library."""
        with self.lock.writer():
            storage_data = self.storage.load(self._storage_key) or {}
            entity_data = storage_data.get(str(entity_id))

            if not entity_data:
                return None

            # Deserialize to Pydantic model
            entity = Library(**entity_data)

            # Update fields using Pydantic's model_copy with update
            from datetime import datetime
            update_data = {**data, "updated_at": datetime.utcnow()}
            updated_entity = entity.model_copy(update=update_data)

            # Serialize back with JSON-compatible format
            storage_data[str(entity_id)] = updated_entity.model_dump(mode='json')
            self.storage.save(self._storage_key, storage_data)

            return updated_entity

    def delete(self, entity_id: UUID) -> bool:
        """Delete a library."""
        with self.lock.writer():
            data = self.storage.load(self._storage_key) or {}

            if str(entity_id) not in data:
                return False

            del data[str(entity_id)]
            self.storage.save(self._storage_key, data)

            return True

    def exists(self, entity_id: UUID) -> bool:
        """Check if library exists."""
        with self.lock.reader():
            data = self.storage.load(self._storage_key) or {}
            return str(entity_id) in data
