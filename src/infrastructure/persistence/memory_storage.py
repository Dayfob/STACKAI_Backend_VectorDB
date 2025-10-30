"""In-memory storage implementation."""

from __future__ import annotations

from typing import Any, Optional

from src.infrastructure.persistence.storage import Storage


class InMemoryStorage(Storage):
    """In-memory storage implementation using a dictionary."""

    def __init__(self) -> None:
        """Initialize in-memory storage."""
        self._data: dict[str, Any] = {}

    def save(self, key: str, data: Any) -> None:
        """Save data to memory.

        Args:
            key: Storage key
            data: Data to save (can be any Python object)
        """
        self._data[key] = data

    def load(self, key: str) -> Optional[Any]:
        """Load data from memory.

        Args:
            key: Storage key

        Returns:
            Loaded data or None if key not found
        """
        return self._data.get(key)

    def delete(self, key: str) -> None:
        """Delete data from memory.

        Args:
            key: Storage key
        """
        self._data.pop(key, None)

    def exists(self, key: str) -> bool:
        """Check if key exists in memory.

        Args:
            key: Storage key

        Returns:
            True if key exists, False otherwise
        """
        return key in self._data

    def list_keys(self) -> list[str]:
        """List all keys in memory.

        Returns:
            List of all keys in storage
        """
        return list(self._data.keys())
