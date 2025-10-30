"""Document endpoints."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.v1.dependencies import get_document_service
from src.core.services import DocumentService
from src.schemas.document import DocumentCreate, DocumentResponse, DocumentUpdate

router = APIRouter(
    prefix="/libraries/{library_id}/documents",
    tags=["documents"],
)


@router.post("/", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
def create_document(
    library_id: UUID,
    data: DocumentCreate,
    service: DocumentService = Depends(get_document_service),
) -> Any:
    """Create a new document in a library.

    Args:
        library_id: Library ID
        data: Document creation data
        service: Document service

    Returns:
        Created document

    Raises:
        HTTPException: If library not found
    """
    from src.core.exceptions import LibraryNotFoundError

    try:
        document = service.create_document(library_id, data)
        return document
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


@router.get("/", response_model=list[DocumentResponse])
def list_documents(
    library_id: UUID,
    service: DocumentService = Depends(get_document_service),
) -> Any:
    """List all documents in a library.

    Args:
        library_id: Library ID
        service: Document service

    Returns:
        List of documents
    """
    try:
        documents = service.list_documents(library_id=library_id)
        return documents
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.get("/{document_id}", response_model=DocumentResponse)
def get_document(
    library_id: UUID,
    document_id: UUID,
    service: DocumentService = Depends(get_document_service),
) -> Any:
    """Get a document by ID.

    Args:
        library_id: Library ID (for consistency)
        document_id: Document ID
        service: Document service

    Returns:
        Document

    Raises:
        HTTPException: If document not found
    """
    from src.core.exceptions import DocumentNotFoundError

    try:
        document = service.get_document(document_id)
        return document
    except DocumentNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=e.message,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.put("/{document_id}", response_model=DocumentResponse)
def update_document(
    library_id: UUID,
    document_id: UUID,
    data: DocumentUpdate,
    service: DocumentService = Depends(get_document_service),
) -> Any:
    """Update a document.

    Args:
        library_id: Library ID (for consistency)
        document_id: Document ID
        data: Update data
        service: Document service

    Returns:
        Updated document

    Raises:
        HTTPException: If document not found
    """
    from src.core.exceptions import DocumentNotFoundError

    try:
        document = service.update_document(document_id, data)
        return document
    except DocumentNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=e.message,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT, response_model=None)
def delete_document(
    library_id: UUID,
    document_id: UUID,
    service: DocumentService = Depends(get_document_service),
) -> None:
    """Delete a document.

    Args:
        library_id: Library ID (for consistency)
        document_id: Document ID
        service: Document service

    Raises:
        HTTPException: If document not found
    """
    from src.core.exceptions import DocumentNotFoundError

    try:
        service.delete_document(document_id)
    except DocumentNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=e.message,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )
