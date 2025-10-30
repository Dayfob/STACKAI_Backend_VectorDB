"""FastAPI application entry point."""

from __future__ import annotations

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.api.v1.routers import chunks, documents, libraries, search
from src.core.config import get_settings
from src.core.exceptions import NotFoundError, VectorDBError

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="REST API for vector similarity search with multiple indexing algorithms",
    docs_url="/docs",
    redoc_url="/redoc",
)


# Exception handlers
@app.exception_handler(VectorDBError)
async def vector_db_error_handler(request: Request, exc: VectorDBError) -> JSONResponse:
    """Handle custom VectorDB errors.

    Args:
        request: Request object
        exc: Exception

    Returns:
        JSON response with error details
    """
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    if isinstance(exc, NotFoundError):
        status_code = status.HTTP_404_NOT_FOUND

    return JSONResponse(
        status_code=status_code,
        content={
            "error": exc.__class__.__name__,
            "message": exc.message,
            "details": exc.details,
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Handle validation errors.

    Args:
        request: Request object
        exc: Exception

    Returns:
        JSON response with validation errors
    """
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "ValidationError",
            "message": "Request validation failed",
            "details": exc.errors(),
        },
    )


# Health check endpoint
@app.get("/health", tags=["health"])
async def health_check() -> dict[str, str]:
    """Health check endpoint.

    Returns:
        Health status
    """
    return {"status": "healthy"}


# Include routers
app.include_router(libraries.router, prefix="/api/v1")
app.include_router(documents.router, prefix="/api/v1")
app.include_router(chunks.router, prefix="/api/v1")
app.include_router(search.router, prefix="/api/v1")


@app.get("/", tags=["root"])
async def root() -> dict[str, str]:
    """Root endpoint.

    Returns:
        Welcome message
    """
    return {
        "message": "Vector Database API",
        "version": settings.app_version,
        "docs": "/docs",
    }
