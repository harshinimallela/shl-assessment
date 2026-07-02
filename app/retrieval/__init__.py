"""
Retrieval factory: selects backend based on config.

Using a factory function rather than conditional imports in dependencies.py
keeps the backend selection in one place and makes it easy to add new backends.
"""
from __future__ import annotations

from pathlib import Path

from app.config import Settings


def build_retriever(settings: Settings, catalog_path: Path):
    """
    Build and return the configured retrieval backend.

    FAISS is the default and recommended backend.
    TF-IDF is the fallback for environments without torch/faiss-cpu.
    """
    if settings.retrieval_backend == "faiss":
        try:
            from app.retrieval.faiss_store import FAISSCatalogStore
            return FAISSCatalogStore(
                catalog_path=catalog_path,
                top_k_retrieve=settings.faiss_top_k_retrieve,
                top_k_rerank=settings.faiss_top_k_rerank,
                enable_reranker=settings.enable_reranker,
                embedding_model=settings.embedding_model,
            )
        except ImportError:
            import logging
            logging.getLogger(__name__).warning(
                "FAISS unavailable, falling back to TF-IDF backend"
            )

    from app.retrieval.tfidf_store import TFIDFCatalogStore
    return TFIDFCatalogStore(catalog_path=catalog_path)
