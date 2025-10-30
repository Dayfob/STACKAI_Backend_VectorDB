"""Disk-based storage implementation."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Literal, Optional

from src.infrastructure.persistence.storage import Storage


class DiskStorage(Storage):
    """Disk-based storage implementation.

    Supports JSON and Pickle serialization formats.
    """

    def __init__(
        self,
        base_path: str = "./data",
        format: Literal["json", "pickle"] = "json",
    ) -> None:
        """Initialize disk storage.

        Args:
            base_path: Base directory for storage
            format: Serialization format (json or pickle)
        """
        self.base_path = Path(base_path)
        self.format = format
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _get_file_path(self, key: str) -> Path:
        """Get file path for a key."""
        extension = "json" if self.format == "json" else "pkl"
        return self.base_path / f"{key}.{extension}"

    def save(self, key: str, data: Any) -> None:
        """Save data to disk.

        Args:
            key: Storage key
            data: Data to save

        Raises:
            IOError: If unable to write to disk
            TypeError: If data is not serializable (JSON mode)
        """
        file_path = self._get_file_path(key)

        try:
            if self.format == "json":
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            else:  # pickle
                with open(file_path, "wb") as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        except (IOError, OSError) as e:
            raise IOError(f"Failed to save data to {file_path}: {e}") from e
        except (TypeError, ValueError) as e:
            raise TypeError(f"Data is not serializable in {self.format} format: {e}") from e

    def load(self, key: str) -> Optional[Any]:
        """Load data from disk.

        Args:
            key: Storage key

        Returns:
            Loaded data or None if key not found

        Raises:
            IOError: If unable to read from disk
            ValueError: If data is corrupted or invalid
        """
        file_path = self._get_file_path(key)

        if not file_path.exists():
            return None

        try:
            if self.format == "json":
                with open(file_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            else:  # pickle
                with open(file_path, "rb") as f:
                    return pickle.load(f)
        except (IOError, OSError) as e:
            raise IOError(f"Failed to load data from {file_path}: {e}") from e
        except (json.JSONDecodeError, pickle.UnpicklingError, ValueError) as e:
            raise ValueError(f"Corrupted or invalid data in {file_path}: {e}") from e

    def delete(self, key: str) -> None:
        """Delete data from disk.

        Args:
            key: Storage key

        Note:
            Does nothing if key doesn't exist (idempotent operation)
        """
        file_path = self._get_file_path(key)
        try:
            file_path.unlink(missing_ok=True)
        except (IOError, OSError):
            # Silently ignore errors (file might be already deleted)
            pass

    def exists(self, key: str) -> bool:
        """Check if key exists on disk.

        Args:
            key: Storage key

        Returns:
            True if key exists, False otherwise
        """
        file_path = self._get_file_path(key)
        return file_path.exists() and file_path.is_file()

    def list_keys(self) -> list[str]:
        """List all keys on disk.

        Returns:
            List of all keys in storage

        Note:
            Only returns keys for files matching the current format
        """
        extension = "json" if self.format == "json" else "pkl"
        keys = []

        if not self.base_path.exists():
            return keys

        for file_path in self.base_path.glob(f"*.{extension}"):
            if file_path.is_file():
                # Remove extension to get the key
                keys.append(file_path.stem)

        return sorted(keys)
