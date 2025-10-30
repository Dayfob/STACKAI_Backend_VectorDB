"""Search endpoints."""

import time
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.v1.dependencies import get_search_service
from src.core.services import SearchService
from src.schemas.search import SearchRequest, SearchResponse

router = APIRouter(
    prefix="/libraries/{library_id}/search",
    tags=["search"],
)


@router.post("/", response_model=SearchResponse)
async def vector_search(
    library_id: UUID,
    request: SearchRequest,
    service: SearchService = Depends(get_search_service),
) -> Any:
    """Perform vector similarity search.

    Args:
        library_id: Library ID
        request: Search request (with embedding or text)
        service: Search service

    Returns:
        Search results

    Raises:
        HTTPException: If library not found or index not built
    """
    from src.core.exceptions import IndexNotBuiltError, LibraryNotFoundError, ValidationError

    try:
        start_time = time.time()
        results = await service.search(library_id, request)
        query_time_ms = (time.time() - start_time) * 1000

        return SearchResponse(
            results=results,
            total=len(results),
            query_time_ms=query_time_ms,
        )
    except LibraryNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=e.message,
        )
    except IndexNotBuiltError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=e.message,
        )
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=e.message,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.post("/semantic", response_model=SearchResponse)
async def semantic_search(
    library_id: UUID,
    query_text: str,
    k: int = 10,
    service: SearchService = Depends(get_search_service),
) -> Any:
    """Perform semantic search with text query.

    Args:
        library_id: Library ID
        query_text: Text query
        k: Number of results
        service: Search service

    Returns:
        Search results

    Raises:
        HTTPException: If library not found or index not built
    """
    from src.core.exceptions import IndexNotBuiltError, LibraryNotFoundError, ValidationError

    try:
        start_time = time.time()
        results = await service.semantic_search(library_id, query_text, k)
        query_time_ms = (time.time() - start_time) * 1000

        return SearchResponse(
            results=results,
            total=len(results),
            query_time_ms=query_time_ms,
        )
    except LibraryNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=e.message,
        )
    except IndexNotBuiltError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=e.message,
        )
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=e.message,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )
