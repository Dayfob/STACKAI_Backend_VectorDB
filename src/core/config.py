"""Application configuration."""

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    app_name: str = "Vector Database API"
    app_version: str = "1.0.0"
    debug: bool = False

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Cohere API
    cohere_api_key: str

    # Storage
    storage_type: str = "memory"  # Options: memory, disk
    storage_path: str = "./data"
    storage_format: str = "json"  # Options: json, pickle (only for disk storage)

    # Index
    default_index_type: str = "brute_force"  # Options: brute_force, hnsw, lsh

    class Config:
        """Pydantic config."""

        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance.

    Returns:
        Settings instance
    """
    return Settings()
