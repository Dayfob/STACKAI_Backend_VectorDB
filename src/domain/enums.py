"""Domain enums for the vector database."""

from enum import Enum


class IndexType(str, Enum):
    """Vector index algorithm types."""

    BRUTE_FORCE = "brute_force"
    HNSW = "hnsw"
    LSH = "lsh"


class StorageType(str, Enum):
    """Storage backend types."""

    MEMORY = "memory"
    DISK = "disk"
