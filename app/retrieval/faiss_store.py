"""
FAISSCatalogStore: dense semantic retrieval with CrossEncoder reranking.

Pipeline:
  1. Optional metadata pre-filter (mask candidates before ANN search)
  2. Encode query with sentence-transformers (all-MiniLM-L6-v2)
  3. FAISS IndexFlatIP cosine search → top-20 candidates
  4. CrossEncoder reranker (ms-marco-MiniLM-L-6-v2) → top-5
  5. Apply metadata filter to reranked results

Why all-MiniLM-L6-v2?
  - 384-dim embeddings: fast encoding (~5ms CPU), small index footprint
  - Strong semantic coverage for HR/hiring domain
  - No API calls: fully local inference

Why CrossEncoder reranking?
  - Bi-encoder retrieval maximizes recall (top-20)
  - CrossEncoder sees (query, doc) jointly → much higher precision
  - ms-marco-MiniLM-L-6-v2 is fast enough for top-20 re-scoring (~15ms CPU)
  - Result: recall of broad retrieval + precision of deep comparison

Why not just CrossEncoder from scratch?
  - CrossEncoder is O(N) at query time — scanning 89 items is fine now,
    but at 1000+ items it becomes a bottleneck
  - Bi-encoder + CrossEncoder is the industry-standard two-tower pattern

Alternatives considered:
  - OpenAI text-embedding-3-small: lower latency on big catalogs but
    adds API cost, network latency, and a hard runtime dependency
  - BM25 + dense hybrid: marginally better recall on this dataset, not
    worth the added complexity at 89 items
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from app.models.catalog import CatalogEntry

logger = logging.getLogger(__name__)

# Lazy imports — only loaded if FAISS backend is selected
# This keeps startup fast if tfidf backend is chosen
try:
    import faiss  # type: ignore
    from sentence_transformers import CrossEncoder, SentenceTransformer  # type: ignore
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False
    logger.warning("FAISS/sentence-transformers not installed. Use retrieval_backend=tfidf")


class FAISSCatalogStore:
    """
    Semantic retrieval: sentence-transformers → FAISS → CrossEncoder reranker.

    Implements the Retriever protocol.
    """

    _EMBED_MODEL = "all-MiniLM-L6-v2"
    _RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(
        self,
        catalog_path: Path,
        top_k_retrieve: int = 20,
        top_k_rerank: int = 5,
        enable_reranker: bool = True,
        embedding_model: str = _EMBED_MODEL,
    ) -> None:
        if not _FAISS_AVAILABLE:
            raise ImportError(
                "Install faiss-cpu and sentence-transformers to use the FAISS backend:\n"
                "  pip install faiss-cpu sentence-transformers"
            )

        self._top_k_retrieve = top_k_retrieve
        self._top_k_rerank = top_k_rerank
        self._enable_reranker = enable_reranker

        # ── Load catalog ───────────────────────────────────────────────────────
        with open(catalog_path) as f:
            raw = json.load(f)

        self.entries: List[CatalogEntry] = [CatalogEntry(**item) for item in raw]
        self._url_set: Set[str] = {e.url for e in self.entries}
        self._name_map: Dict[str, CatalogEntry] = {
            e.name.lower(): e for e in self.entries
        }

        # ── Build embedding index ──────────────────────────────────────────────
        t0 = time.perf_counter()
        logger.info("Loading embedding model: %s", embedding_model)
        self._embed_model = SentenceTransformer(embedding_model)

        texts = [e.searchable_text for e in self.entries]
        # encode_batch returns (N, dim) float32 numpy array
        embeddings: np.ndarray = self._embed_model.encode(
            texts,
            normalize_embeddings=True,  # cosine similarity via dot product
            show_progress_bar=False,
            batch_size=64,
        )
        dim = embeddings.shape[1]

        # IndexFlatIP = exact inner-product search on L2-normalized vectors
        # = cosine similarity. Exact (no approximation error) — fine at 89 items.
        # At >10k items, swap to IndexIVFFlat for sub-linear search.
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings.astype(np.float32))

        logger.info(
            "FAISS index built: %d entries, dim=%d, elapsed=%.0fms",
            len(self.entries),
            dim,
            (time.perf_counter() - t0) * 1000,
        )

        # ── Load reranker ──────────────────────────────────────────────────────
        if enable_reranker:
            logger.info("Loading reranker: %s", self._RERANK_MODEL)
            self._reranker = CrossEncoder(self._RERANK_MODEL)
            logger.info("Reranker ready")
        else:
            self._reranker = None

    # ── Retriever protocol ─────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 15,
        test_types: Optional[List[str]] = None,
        job_level: Optional[str] = None,
        language: Optional[str] = None,
    ) -> List[CatalogEntry]:
        """
        Two-stage retrieval:
          1. Dense FAISS search → top_k_retrieve candidates
          2. CrossEncoder rerank → top_k_rerank (if reranker enabled)
          3. Metadata filter applied to final results
        """
        t0 = time.perf_counter()

        # Encode query (normalize for cosine similarity)
        q_vec: np.ndarray = self._embed_model.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        ).astype(np.float32)

        # FAISS search — retrieve generously before reranking
        retrieve_k = min(self._top_k_retrieve, self._index.ntotal)
        scores, indices = self._index.search(q_vec, retrieve_k)

        candidates: List[Tuple[float, CatalogEntry]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            entry = self.entries[idx]
            candidates.append((float(score), entry))

        logger.debug(
            "FAISS retrieved %d candidates in %.0fms",
            len(candidates),
            (time.perf_counter() - t0) * 1000,
        )

        # ── CrossEncoder reranking ─────────────────────────────────────────────
        if self._reranker and self._enable_reranker and candidates:
            t1 = time.perf_counter()
            pairs = [(query, e.searchable_text) for _, e in candidates]
            rerank_scores: List[float] = self._reranker.predict(pairs).tolist()
            candidates = sorted(
                zip(rerank_scores, [e for _, e in candidates]),
                key=lambda x: x[0],
                reverse=True,
            )
            logger.debug(
                "CrossEncoder reranked %d → %.0fms",
                len(candidates),
                (time.perf_counter() - t1) * 1000,
            )

        # ── Metadata filter ────────────────────────────────────────────────────
        # Applied AFTER reranking to preserve semantic ordering.
        # Filtering before FAISS search would require rebuilding sub-indexes.
        results: List[CatalogEntry] = []
        for _, entry in candidates:
            if test_types and entry.test_type not in test_types:
                continue
            if job_level and not any(
                job_level.lower() in lvl.lower() for lvl in entry.job_levels
            ):
                continue
            if language and not any(
                language.lower() in lang.lower() for lang in entry.languages
            ):
                continue
            results.append(entry)
            if len(results) >= top_k:
                break

        logger.debug(
            "search('%s') → %d results, total=%.0fms",
            query[:60],
            len(results),
            (time.perf_counter() - t0) * 1000,
        )
        return results

    def get_by_names(self, names: List[str]) -> List[CatalogEntry]:
        result = []
        for name in names:
            entry = self._name_map.get(name.lower())
            if entry:
                result.append(entry)
            else:
                for key, e in self._name_map.items():
                    if name.lower() in key or key in name.lower():
                        result.append(e)
                        break
        return result

    def is_valid_url(self, url: str) -> bool:
        return url in self._url_set

    def is_valid_name(self, name: str) -> bool:
        return name.lower() in self._name_map

    @property
    def all_entries(self) -> List[CatalogEntry]:
        return self.entries

    @property
    def all_urls(self) -> Set[str]:
        return self._url_set
