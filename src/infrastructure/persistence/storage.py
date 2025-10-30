"""Abstract storage interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional


class Storage(ABC):
    """Abstract base class for storage backends."""

    @abstractmethod
    def save(self, key: str, data: Any) -> None:
        """Save data to storage.

        Args:
            key: Storage key
            data: Data to save
        """
        pass

    @abstractmethod
    def load(self, key: str) -> Optional[Any]:
        """Load data from storage.

        Args:
            key: Storage key

        Returns:
            Loaded data or None if not found
        """
        pass

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete data from storage.

        Args:
            key: Storage key
        """
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists in storage.

        Args:
            key: Storage key

        Returns:
            True if key exists, False otherwise
        """
        pass

    @abstractmethod
    def list_keys(self) -> list[str]:
        """List all keys in storage.

        Returns:
            List of all keys
        """
        pass
