"""Embedding service using Cohere API."""

from __future__ import annotations

import httpx

from src.core.exceptions import EmbeddingError

# Cohere API limits
MAX_BATCH_SIZE = 96  # Maximum number of texts per API call


class EmbeddingService:
    """Service for generating embeddings using Cohere API."""

    def __init__(self, api_key: str, model: str = "embed-english-v3.0") -> None:
        """Initialize embedding service.

        Args:
            api_key: Cohere API key
            model: Cohere embedding model name
        """
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.cohere.ai/v1"

    async def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Input text

        Returns:
            Embedding vector

        Raises:
            EmbeddingError: If API request fails or embedding generation fails
        """
        embeddings = await self.embed_texts([text])
        return embeddings[0]

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of input texts (max 96 per request)

        Returns:
            List of embedding vectors

        Raises:
            EmbeddingError: If API request fails or embedding generation fails
            ValueError: If texts list is empty or exceeds maximum batch size
        """
        if not texts:
            raise ValueError("texts list cannot be empty")

        if len(texts) > MAX_BATCH_SIZE:
            raise ValueError(
                f"Cannot embed {len(texts)} texts. Maximum batch size is {MAX_BATCH_SIZE}. "
                f"Please split your request into smaller batches."
            )

        try:
            async with httpx.AsyncClient() as client:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }

                payload = {
                    "texts": texts,
                    "model": self.model,
                    "input_type": "search_document",
                    "embedding_types": ["float"],
                }

                response = await client.post(
                    f"{self.base_url}/embed",
                    headers=headers,
                    json=payload,
                    timeout=30.0,
                )

                response.raise_for_status()
                data = response.json()

                # Extract embeddings from response
                # Cohere API returns two possible formats:
                # 1. EmbedByTypeResponse: {"embeddings": {"float": [[...]], ...}}
                # 2. EmbedFloatsResponse: {"embeddings": [[...]]}
                embeddings_data = data.get("embeddings")

                if not embeddings_data:
                    raise EmbeddingError(
                        "No embeddings found in API response",
                        details={"response": data},
                    )

                # Handle EmbedByTypeResponse format
                if isinstance(embeddings_data, dict) and "float" in embeddings_data:
                    return embeddings_data["float"]

                # Handle EmbedFloatsResponse format
                if isinstance(embeddings_data, list):
                    return embeddings_data

                raise EmbeddingError(
                    "Unexpected embeddings format in API response",
                    details={"embeddings_type": type(embeddings_data).__name__},
                )

        except EmbeddingError:
            # Re-raise EmbeddingError without wrapping
            raise

        except httpx.HTTPStatusError as e:
            error_detail = ""
            try:
                error_data = e.response.json()
                error_detail = error_data.get("message", str(e))
            except Exception:
                error_detail = str(e)

            raise EmbeddingError(
                f"Failed to generate embeddings: {error_detail}",
                details={"status_code": e.response.status_code, "texts_count": len(texts)},
            ) from e

        except httpx.RequestError as e:
            raise EmbeddingError(
                f"Network error while generating embeddings: {str(e)}",
                details={"texts_count": len(texts)},
            ) from e

        except Exception as e:
            raise EmbeddingError(
                f"Unexpected error during embedding generation: {str(e)}",
                details={"texts_count": len(texts)},
            ) from e

    async def embed_query(self, query: str) -> list[float]:
        """Generate embedding for a search query.

        Args:
            query: Search query text

        Returns:
            Embedding vector

        Raises:
            EmbeddingError: If API request fails or embedding generation fails
        """
        try:
            async with httpx.AsyncClient() as client:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }

                payload = {
                    "texts": [query],
                    "model": self.model,
                    "input_type": "search_query",
                    "embedding_types": ["float"],
                }

                response = await client.post(
                    f"{self.base_url}/embed",
                    headers=headers,
                    json=payload,
                    timeout=30.0,
                )

                response.raise_for_status()
                data = response.json()

                # Extract embedding from response
                # Cohere API returns two possible formats:
                # 1. EmbedByTypeResponse: {"embeddings": {"float": [[...]]}}
                # 2. EmbedFloatsResponse: {"embeddings": [[...]]}
                embeddings_data = data.get("embeddings")

                if not embeddings_data:
                    raise EmbeddingError(
                        "No embeddings found in API response",
                        details={"response": data},
                    )

                # Handle EmbedByTypeResponse format
                if isinstance(embeddings_data, dict) and "float" in embeddings_data:
                    float_embeddings = embeddings_data["float"]
                    if float_embeddings and len(float_embeddings) > 0:
                        return float_embeddings[0]
                    raise EmbeddingError(
                        "Empty embeddings list in API response",
                        details={"response": data},
                    )

                # Handle EmbedFloatsResponse format
                if isinstance(embeddings_data, list):
                    if embeddings_data and len(embeddings_data) > 0:
                        return embeddings_data[0]
                    raise EmbeddingError(
                        "Empty embeddings list in API response",
                        details={"response": data},
                    )

                raise EmbeddingError(
                    "Unexpected embeddings format in API response",
                    details={"embeddings_type": type(embeddings_data).__name__},
                )

        except EmbeddingError:
            # Re-raise EmbeddingError without wrapping
            raise

        except httpx.HTTPStatusError as e:
            error_detail = ""
            try:
                error_data = e.response.json()
                error_detail = error_data.get("message", str(e))
            except Exception:
                error_detail = str(e)

            raise EmbeddingError(
                f"Failed to generate query embedding: {error_detail}",
                details={"status_code": e.response.status_code, "query": query},
            ) from e

        except httpx.RequestError as e:
            raise EmbeddingError(
                f"Network error while generating query embedding: {str(e)}",
                details={"query": query},
            ) from e

        except Exception as e:
            raise EmbeddingError(
                f"Unexpected error during query embedding generation: {str(e)}",
                details={"query": query},
            ) from e
