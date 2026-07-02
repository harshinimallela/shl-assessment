"""
Configuration: pydantic-settings validates all env vars at import time.

Design: fail-fast at startup beats runtime KeyError mid-request.
Alternative considered: python-dotenv with manual validation — less type-safe,
no automatic env coercion, no field-level documentation.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ── Required ─────────────────────────────────────────────────────────────
    # Get your free key at: https://console.groq.com
    groq_api_key: str = Field(..., description="Groq API key (free at console.groq.com)")

    # ── Optional with sensible defaults ──────────────────────────────────────
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    cors_origins: str = "*"

    # LLM model — all free on Groq:
    #   llama-3.3-70b-versatile  → best quality, recommended
    #   llama-3.1-8b-instant     → fastest, lower quality
    #   mixtral-8x7b-32768       → long context (32k tokens)
    llm_model: str = "llama-3.3-70b-versatile"
    llm_max_tokens: int = 1024
    llm_max_retries: int = 2

    # Retrieval backend
    retrieval_backend: Literal["tfidf", "faiss"] = "faiss"
    embedding_model: str = "all-MiniLM-L6-v2"
    faiss_top_k_retrieve: int = 20
    faiss_top_k_rerank: int = 5
    enable_reranker: bool = True

    # Catalog
    catalog_path: str = "catalog.json"

    @field_validator("groq_api_key")
    @classmethod
    def api_key_must_be_set(cls, v: str) -> str:
        if (not v or v.startswith("your") or "your_groq_api_key_here" in v.lower()):
            raise ValueError("GROQ_API_KEY must be set. Get a free key at https://console.groq.com")
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton settings — evaluated once, cached forever."""
    return Settings()
