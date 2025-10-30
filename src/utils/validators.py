"""Validation utilities."""

from __future__ import annotations

from typing import Any


def validate_embedding_dimension(
    embedding: list[float],
    expected_dim: int,
) -> bool:
    """Validate that embedding has expected dimension.

    Args:
        embedding: Embedding vector
        expected_dim: Expected dimension

    Returns:
        True if valid, False otherwise

    Raises:
        ValueError: If embedding is None, empty, or has wrong dimension
    """
    if embedding is None:
        raise ValueError("Embedding cannot be None")

    if not isinstance(embedding, list):
        raise ValueError(f"Embedding must be a list, got {type(embedding).__name__}")

    if len(embedding) == 0:
        raise ValueError("Embedding cannot be empty")

    if not all(isinstance(x, (int, float)) for x in embedding):
        raise ValueError("Embedding must contain only numeric values")

    if len(embedding) != expected_dim:
        raise ValueError(
            f"Embedding dimension mismatch: expected {expected_dim}, got {len(embedding)}"
        )

    return True


def validate_metadata(metadata: dict[str, Any]) -> bool:
    """Validate metadata structure.

    Ensures metadata contains only JSON-serializable types and no circular references.

    Args:
        metadata: Metadata dictionary

    Returns:
        True if valid, False otherwise

    Raises:
        ValueError: If metadata contains invalid types or structures
    """
    if metadata is None:
        raise ValueError("Metadata cannot be None")

    if not isinstance(metadata, dict):
        raise ValueError(f"Metadata must be a dict, got {type(metadata).__name__}")

    # Check for valid JSON-serializable types
    def _check_value(value: Any, path: str = "metadata", seen: set | None = None) -> None:
        """Recursively check if value is JSON-serializable."""
        if seen is None:
            seen = set()

        # Check for circular references
        value_id = id(value)
        if isinstance(value, (dict, list)) and value_id in seen:
            raise ValueError(f"Circular reference detected at {path}")

        if isinstance(value, (dict, list)):
            seen.add(value_id)

        # Check types
        if value is None or isinstance(value, (bool, int, float, str)):
            return

        if isinstance(value, dict):
            for key, val in value.items():
                if not isinstance(key, str):
                    raise ValueError(
                        f"Dictionary keys must be strings at {path}, got {type(key).__name__}"
                    )
                _check_value(val, f"{path}.{key}", seen.copy())

        elif isinstance(value, (list, tuple)):
            for idx, item in enumerate(value):
                _check_value(item, f"{path}[{idx}]", seen.copy())

        else:
            raise ValueError(
                f"Invalid metadata type at {path}: {type(value).__name__}. "
                f"Only JSON-serializable types are allowed (str, int, float, bool, dict, list, None)"
            )

    _check_value(metadata)
    return True
