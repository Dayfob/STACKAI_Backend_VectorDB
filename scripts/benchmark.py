"""Benchmark script to compare index performance."""

from __future__ import annotations

import time
from typing import Any

from src.domain.models.chunk import Chunk
from src.infrastructure.indexes import BruteForceIndex, HNSWIndex, LSHIndex


def generate_sample_chunks(n: int, dimension: int = 1024) -> list[Chunk]:
    """Generate sample chunks with random embeddings.

    Args:
        n: Number of chunks
        dimension: Embedding dimension

    Returns:
        List of chunks
    """
    # TODO: Implement generation with random embeddings
    pass


def benchmark_index(
    index_class: type,
    chunks: list[Chunk],
    query_embedding: list[float],
    k: int = 10,
) -> dict[str, Any]:
    """Benchmark an index implementation.

    Args:
        index_class: Index class to benchmark
        chunks: Chunks to index
        query_embedding: Query embedding
        k: Number of neighbors

    Returns:
        Benchmark results
    """
    # Build time
    index = index_class()
    start_time = time.time()
    index.build(chunks)
    build_time = time.time() - start_time

    # Search time (average of multiple queries)
    search_times = []
    for _ in range(100):
        start_time = time.time()
        results = index.search(query_embedding, k)
        search_time = time.time() - start_time
        search_times.append(search_time)

    avg_search_time = sum(search_times) / len(search_times)

    return {
        "index_type": index_class.__name__,
        "build_time": build_time,
        "avg_search_time": avg_search_time,
        "index_size": index.size(),
    }


def main() -> None:
    """Run benchmark comparison."""
    print("Vector Index Benchmark")
    print("=" * 50)

    # Generate test data
    print("\nGenerating test data...")
    chunks = generate_sample_chunks(n=10000, dimension=1024)
    query_embedding = [0.1] * 1024

    # Benchmark each index
    indexes = [BruteForceIndex, HNSWIndex, LSHIndex]

    results = []
    for index_class in indexes:
        print(f"\nBenchmarking {index_class.__name__}...")
        result = benchmark_index(index_class, chunks, query_embedding)
        results.append(result)

        print(f"Build time: {result['build_time']:.4f}s")
        print(f"Avg search time: {result['avg_search_time']*1000:.4f}ms")

    # Print comparison
    print("\n" + "=" * 50)
    print("Comparison Table:")
    print("=" * 50)
    print(f"{'Index':<20} {'Build Time':<15} {'Search Time':<15}")
    print("-" * 50)
    for result in results:
        print(
            f"{result['index_type']:<20} "
            f"{result['build_time']:.4f}s{' '*7} "
            f"{result['avg_search_time']*1000:.4f}ms"
        )


if __name__ == "__main__":
    main()
