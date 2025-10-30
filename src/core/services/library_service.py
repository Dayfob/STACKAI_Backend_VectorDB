"""Library service for business logic."""

from __future__ import annotations

from typing import Any, Optional
from uuid import UUID

from src.domain.models.library import Library
from src.infrastructure.repositories.library_repository import LibraryRepository
from src.schemas.library import LibraryCreate, LibraryUpdate


class LibraryService:
    """Service for managing libraries."""

    def __init__(self, repository: LibraryRepository) -> None:
        """Initialize library service.

        Args:
            repository: Library repository
        """
        self.repository = repository

    def create_library(self, data: LibraryCreate) -> Library:
        """Create a new library.

        Args:
            data: Library creation data

        Returns:
            Created library
        """
        library = Library(
            name=data.name,
            description=data.description,
            metadata=data.metadata,
            index_type=data.index_type,
        )
        return self.repository.create(library)

    def get_library(self, library_id: UUID) -> Library:
        """Get a library by ID.

        Args:
            library_id: Library ID

        Returns:
            Library

        Raises:
            LibraryNotFoundError: If library not found
        """
        from src.core.exceptions import LibraryNotFoundError

        library = self.repository.get(library_id)
        if not library:
            raise LibraryNotFoundError(str(library_id))
        return library

    def list_libraries(self, filters: Optional[dict[str, Any]] = None) -> list[Library]:
        """List all libraries.

        Args:
            filters: Optional filters

        Returns:
            List of libraries
        """
        return self.repository.list(filters)

    def update_library(self, library_id: UUID, data: LibraryUpdate) -> Library:
        """Update a library.

        Args:
            library_id: Library ID
            data: Update data

        Returns:
            Updated library

        Raises:
            LibraryNotFoundError: If library not found
        """
        from src.core.exceptions import LibraryNotFoundError

        # Filter out None values
        update_data = {k: v for k, v in data.model_dump().items() if v is not None}

        # If nothing to update, just return the existing library
        if not update_data:
            return self.get_library(library_id)

        updated_library = self.repository.update(library_id, update_data)
        if not updated_library:
            raise LibraryNotFoundError(str(library_id))
        return updated_library

    def delete_library(self, library_id: UUID) -> None:
        """Delete a library.

        Args:
            library_id: Library ID

        Raises:
            LibraryNotFoundError: If library not found
        """
        from src.core.exceptions import LibraryNotFoundError

        success = self.repository.delete(library_id)
        if not success:
            raise LibraryNotFoundError(str(library_id))

    def index_library(self, library_id: UUID) -> None:
        """Build index for a library.

        Args:
            library_id: Library ID

        Raises:
            LibraryNotFoundError: If library not found
        """
        from src.core.exceptions import LibraryNotFoundError

        # Get library
        library = self.get_library(library_id)

        # Mark library as indexed
        self.repository.update(library_id, {"is_indexed": True})
