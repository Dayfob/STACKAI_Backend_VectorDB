"""Mathematical utilities for vector operations."""

from __future__ import annotations

import numpy as np


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Calculate cosine similarity between two vectors.

    Cosine similarity = (A · B) / (||A|| * ||B||)

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity score between -1 and 1
        (1 = identical, 0 = orthogonal, -1 = opposite)
    """
    a = np.array(vec1)
    b = np.array(vec2)

    dot_prod = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(dot_prod / (norm_a * norm_b))


def euclidean_distance(vec1: list[float], vec2: list[float]) -> float:
    """Calculate Euclidean distance between two vectors.

    Distance = sqrt(sum((a_i - b_i)^2))

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Euclidean distance (0 = identical, larger = more different)
    """
    a = np.array(vec1)
    b = np.array(vec2)
    return float(np.linalg.norm(a - b))


def normalize_vector(vec: list[float]) -> list[float]:
    """Normalize a vector to unit length.

    Normalized vector = v / ||v||

    Args:
        vec: Input vector

    Returns:
        Normalized vector with length 1
    """
    arr = np.array(vec)
    norm = np.linalg.norm(arr)

    if norm == 0:
        return vec

    return (arr / norm).tolist()


def dot_product(vec1: list[float], vec2: list[float]) -> float:
    """Calculate dot product of two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Dot product
    """
    a = np.array(vec1)
    b = np.array(vec2)
    return float(np.dot(a, b))


def vector_magnitude(vec: list[float]) -> float:
    """Calculate magnitude (L2 norm) of a vector.

    Magnitude = sqrt(sum(x_i^2))

    Args:
        vec: Input vector

    Returns:
        Vector magnitude
    """
    arr = np.array(vec)
    return float(np.linalg.norm(arr))
