"""Centralised settings loaded from environment / .env file."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )

    # ── Groq LLM ─────────────────────────────────────────────────────
    groq_api_key: str = Field(..., alias="GROQ_API_KEY")
    groq_model: str = Field("llama-3.3-70b-versatile", alias="GROQ_MODEL")

    # ── Local embeddings (sentence-transformers) ──────────────────────
    embedding_model: str = Field("all-MiniLM-L6-v2", alias="EMBEDDING_MODEL")

    # ── Neo4j ─────────────────────────────────────────────────────────
    neo4j_uri: str = Field("bolt://localhost:7687", alias="NEO4J_URI")
    neo4j_username: str = Field("neo4j", alias="NEO4J_USERNAME")
    neo4j_password: str = Field(..., alias="NEO4J_PASSWORD")
    neo4j_database: str = Field("neo4j", alias="NEO4J_DATABASE")

    # ── ChromaDB ──────────────────────────────────────────────────────
    chroma_host: str = Field("localhost", alias="CHROMA_HOST")
    chroma_port: int = Field(8000, alias="CHROMA_PORT")
    chroma_collection: str = Field("rag_documents", alias="CHROMA_COLLECTION")

    # ── App ───────────────────────────────────────────────────────────
    app_host: str = Field("0.0.0.0", alias="APP_HOST")
    app_port: int = Field(8080, alias="APP_PORT")
    app_env: str = Field("development", alias="APP_ENV")
    log_level: str = Field("INFO", alias="LOG_LEVEL")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings singleton."""
    return Settings()
