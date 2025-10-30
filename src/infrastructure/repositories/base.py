"""Base repository interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, Optional, TypeVar
from uuid import UUID

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class BaseRepository(ABC, Generic[T]):
    """Abstract base repository for CRUD operations."""

    @abstractmethod
    def create(self, entity: T) -> T:
        """Create a new entity.

        Args:
            entity: Entity to create

        Returns:
            Created entity
        """
        pass

    @abstractmethod
    def get(self, entity_id: UUID) -> Optional[T]:
        """Get entity by ID.

        Args:
            entity_id: Entity ID

        Returns:
            Entity if found, None otherwise
        """
        pass

    @abstractmethod
    def list(self, filters: Optional[dict[str, Any]] = None) -> list[T]:
        """List all entities with optional filters.

        Args:
            filters: Optional filter criteria

        Returns:
            List of entities
        """
        pass

    @abstractmethod
    def update(self, entity_id: UUID, data: dict[str, Any]) -> Optional[T]:
        """Update an entity.

        Args:
            entity_id: Entity ID
            data: Update data

        Returns:
            Updated entity if found, None otherwise
        """
        pass

    @abstractmethod
    def delete(self, entity_id: UUID) -> bool:
        """Delete an entity.

        Args:
            entity_id: Entity ID

        Returns:
            True if deleted, False if not found
        """
        pass

    @abstractmethod
    def exists(self, entity_id: UUID) -> bool:
        """Check if entity exists.

        Args:
            entity_id: Entity ID

        Returns:
            True if exists, False otherwise
        """
        pass
