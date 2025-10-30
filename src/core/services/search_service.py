"""Search service for vector similarity search."""

from __future__ import annotations

from typing import Any, Optional
from uuid import UUID

from src.domain.enums import IndexType
from src.infrastructure.indexes import BruteForceIndex, HNSWIndex, LSHIndex, VectorIndex
from src.infrastructure.repositories.chunk_repository import ChunkRepository
from src.infrastructure.repositories.library_repository import LibraryRepository
from src.schemas.search import SearchRequest, SearchResult
from src.utils.embeddings import EmbeddingService


class SearchService:
    """Service for vector similarity search."""

    def __init__(
        self,
        library_repository: LibraryRepository,
        chunk_repository: ChunkRepository,
        embedding_service: EmbeddingService,
    ) -> None:
        """Initialize search service.

        Args:
            library_repository: Library repository
            chunk_repository: Chunk repository
            embedding_service: Embedding service
        """
        self.library_repository = library_repository
        self.chunk_repository = chunk_repository
        self.embedding_service = embedding_service
        self._indexes: dict[UUID, VectorIndex] = {}

    def invalidate_index(self, library_id: UUID) -> None:
        """Invalidate cached index for a library.

        Args:
            library_id: Library ID
        """
        if library_id in self._indexes:
            del self._indexes[library_id]

    def clear_all_indexes(self) -> None:
        """Clear all cached indexes."""
        self._indexes.clear()

    def _get_or_create_index(self, library_id: UUID) -> VectorIndex:
        """Get or create index for a library.

        Args:
            library_id: Library ID

        Returns:
            Vector index

        Raises:
            LibraryNotFoundError: If library not found
        """
        from src.core.exceptions import LibraryNotFoundError

        # Check if index exists in cache
        if library_id in self._indexes:
            return self._indexes[library_id]

        # Get library
        library = self.library_repository.get(library_id)
        if not library:
            raise LibraryNotFoundError(str(library_id))

        # Create appropriate index based on library.index_type
        index = self._create_index(library.index_type)

        # Get all chunks for this library's documents
        all_chunks = []
        for document_id in library.document_ids:
            chunks = self.chunk_repository.list_by_document(document_id)
            all_chunks.extend(chunks)

        # Validate that all chunks have embeddings
        chunks_without_embeddings = [
            chunk.id for chunk in all_chunks if not chunk.embedding
        ]
        if chunks_without_embeddings:
            from src.core.exceptions import ValidationError
            raise ValidationError(
                f"Cannot build index: {len(chunks_without_embeddings)} chunks missing embeddings",
                details={"chunk_ids": [str(cid) for cid in chunks_without_embeddings[:10]]},
            )

        # Build index with chunks
        if all_chunks:
            index.build(all_chunks)

        # Cache the index
        self._indexes[library_id] = index

        return index

    def _create_index(self, index_type: IndexType) -> VectorIndex:
        """Create index based on type.

        Args:
            index_type: Index type

        Returns:
            Vector index
        """
        if index_type == IndexType.BRUTE_FORCE:
            return BruteForceIndex()
        elif index_type == IndexType.HNSW:
            return HNSWIndex()
        elif index_type == IndexType.LSH:
            return LSHIndex()
        else:
            raise ValueError(f"Unknown index type: {index_type}")

    async def search(
        self,
        library_id: UUID,
        request: SearchRequest,
    ) -> list[SearchResult]:
        """Perform vector similarity search.

        Args:
            library_id: Library ID
            request: Search request

        Returns:
            List of search results

        Raises:
            LibraryNotFoundError: If library not found
            IndexNotBuiltError: If index not built
        """
        from src.core.exceptions import IndexNotBuiltError, LibraryNotFoundError

        # Get library
        library = self.library_repository.get(library_id)
        if not library:
            raise LibraryNotFoundError(str(library_id))

        # Check library.is_indexed
        if not library.is_indexed:
            raise IndexNotBuiltError(str(library_id))

        # Get or create index
        index = self._get_or_create_index(library_id)

        # Determine query embedding
        query_embedding = request.query_embedding
        if request.query_text:
            query_embedding = await self.embedding_service.embed_query(request.query_text)
        elif not query_embedding:
            raise ValueError("Either query_text or query_embedding must be provided")

        # If filters provided, request more results to account for filtering
        # Use 2x or max 100 to have buffer for filtering
        search_k = request.k
        if request.filters:
            search_k = min(request.k * 2, 100)

        # Search index
        search_results = index.search(query_embedding, search_k)

        # Build search results with chunk data
        results = []
        for chunk_id, score in search_results:
            chunk = self.chunk_repository.get(chunk_id)
            if chunk:
                # Apply metadata filters if provided
                if request.filters:
                    match = True
                    for key, value in request.filters.items():
                        if key not in chunk.metadata or chunk.metadata[key] != value:
                            match = False
                            break
                    if not match:
                        continue

                results.append(
                    SearchResult(
                        chunk_id=chunk.id,
                        document_id=chunk.document_id,
                        content=chunk.content,
                        score=score,
                        metadata=chunk.metadata,
                    )
                )

                # Stop when we have enough filtered results
                if len(results) >= request.k:
                    break

        return results

    async def semantic_search(
        self,
        library_id: UUID,
        query_text: str,
        k: int = 10,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[SearchResult]:
        """Perform semantic search with text query.

        Args:
            library_id: Library ID
            query_text: Text query
            k: Number of results
            filters: Optional metadata filters

        Returns:
            List of search results

        Raises:
            LibraryNotFoundError: If library not found
            IndexNotBuiltError: If index not built
        """
        request = SearchRequest(
            query_text=query_text,
            k=k,
            filters=filters or {},
        )
        return await self.search(library_id, request)
