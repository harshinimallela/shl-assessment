"""
Dependencies: composition root for dependency injection.

All singletons are built once at startup and cached via lru_cache.
FastAPI's Depends() calls these functions; they return the cached instances.

Design rationale:
  - lru_cache(maxsize=1) simulates singleton scope without a global variable
  - The Supervisor receives injected dependencies, not globals
  - Swapping the retrieval backend = one line change in build_retriever()
  - Testable: override any dependency in tests via app.dependency_overrides
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from groq import Groq

from app.config import get_settings
from app.prompt_manager import PromptManager
from app.retrieval import build_retriever
from app.supervisor import Supervisor

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"
_CATALOG_PATH = Path("catalog.json")


@lru_cache(maxsize=1)
def get_groq_client() -> Groq:
    """
    Singleton Groq client — one connection pool for the process.

    Groq's client is thread-safe and reuse is strongly recommended.
    Get a free API key at https://console.groq.com
    """
    settings = get_settings()
    return Groq(api_key=settings.groq_api_key)


@lru_cache(maxsize=1)
def get_prompt_manager() -> PromptManager:
    """Singleton PromptManager — loads all prompts once at startup."""
    return PromptManager(_PROMPTS_DIR)


@lru_cache(maxsize=1)
def get_catalog_store():
    """
    Singleton retriever — builds index once at startup.

    For FAISS: embedding model loads + catalog encoded here (~2-5s cold start).
    For TF-IDF: sub-second startup.
    """
    settings = get_settings()
    logger.info("Building retriever backend=%s", settings.retrieval_backend)
    return build_retriever(settings, _CATALOG_PATH)


@lru_cache(maxsize=1)
def get_supervisor() -> Supervisor:
    """Singleton Supervisor — constructed with all injected dependencies."""
    return Supervisor(
        client=get_groq_client(),
        catalog_store=get_catalog_store(),
        prompt_manager=get_prompt_manager(),
    )


def supervisor_dep() -> Supervisor:
    """
    FastAPI dependency function.

    Not cached itself (FastAPI calls this per-request) but delegates to
    cached get_supervisor() — effectively zero cost per request.
    """
    return get_supervisor()
