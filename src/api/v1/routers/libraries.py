"""Library endpoints."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.v1.dependencies import get_library_service
from src.core.services import LibraryService
from src.schemas.library import LibraryCreate, LibraryResponse, LibraryUpdate

router = APIRouter(prefix="/libraries", tags=["libraries"])


@router.post("/", response_model=LibraryResponse, status_code=status.HTTP_201_CREATED)
def create_library(
    data: LibraryCreate,
    service: LibraryService = Depends(get_library_service),
) -> Any:
    """Create a new library.

    Args:
        data: Library creation data
        service: Library service

    Returns:
        Created library
    """
    try:
        library = service.create_library(data)
        return library
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.get("/", response_model=list[LibraryResponse])
def list_libraries(
    service: LibraryService = Depends(get_library_service),
) -> Any:
    """List all libraries.

    Args:
        service: Library service

    Returns:
        List of libraries
    """
    try:
        libraries = service.list_libraries()
        return libraries
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.get("/{library_id}", response_model=LibraryResponse)
def get_library(
    library_id: UUID,
    service: LibraryService = Depends(get_library_service),
) -> Any:
    """Get a library by ID.

    Args:
        library_id: Library ID
        service: Library service

    Returns:
        Library

    Raises:
        HTTPException: If library not found
    """
    from src.core.exceptions import LibraryNotFoundError

    try:
        library = service.get_library(library_id)
        return library
    except LibraryNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=e.message,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.put("/{library_id}", response_model=LibraryResponse)
def update_library(
    library_id: UUID,
    data: LibraryUpdate,
    service: LibraryService = Depends(get_library_service),
) -> Any:
    """Update a library.

    Args:
        library_id: Library ID
        data: Update data
        service: Library service

    Returns:
        Updated library

    Raises:
        HTTPException: If library not found
    """
    from src.core.exceptions import LibraryNotFoundError

    try:
        library = service.update_library(library_id, data)
        return library
    except LibraryNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=e.message,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.delete("/{library_id}", status_code=status.HTTP_204_NO_CONTENT, response_model=None)
def delete_library(
    library_id: UUID,
    service: LibraryService = Depends(get_library_service),
) -> None:
    """Delete a library.

    Args:
        library_id: Library ID
        service: Library service

    Raises:
        HTTPException: If library not found
    """
    from src.core.exceptions import LibraryNotFoundError

    try:
        service.delete_library(library_id)
    except LibraryNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=e.message,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.post("/{library_id}/index", status_code=status.HTTP_200_OK)
def index_library(
    library_id: UUID,
    service: LibraryService = Depends(get_library_service),
) -> Any:
    """Build index for a library.

    Args:
        library_id: Library ID
        service: Library service

    Returns:
        Success message

    Raises:
        HTTPException: If library not found
    """
    from src.core.exceptions import LibraryNotFoundError

    try:
        service.index_library(library_id)
        return {"message": "Index built successfully", "library_id": str(library_id)}
    except LibraryNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=e.message,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )
