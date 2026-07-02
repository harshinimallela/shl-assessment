"""
CatalogStore (TF-IDF backend): zero-dependency retrieval.

Pipeline:
  1. Optional metadata pre-filter (test_type, job_level, language)
  2. TF-IDF cosine similarity ranking
  3. Top-K selection

Why TF-IDF at all (vs. just FAISS)?
  - Keeps the service bootable on any machine with no GPU/internet
  - Useful as a fast smoke-test baseline
  - At ≤500 catalog items, recall difference vs. dense is marginal on
    domain-specific keyword queries

When to switch to FAISS backend:
  - Catalog > 500 items
  - Queries use paraphrasing / synonyms the TF-IDF vocabulary misses
  - Multilingual queries

The Supervisor receives a Retriever protocol object, not a concrete class,
so swapping backends is a one-line change in dependencies.py.
"""
from __future__ import annotations

import json
import logging
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from app.models.catalog import CatalogEntry

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> List[str]:
    """Whitespace + punctuation tokenizer, lowercase. Preserves C++, C#."""
    return re.findall(r"[a-z0-9#+.]+", text.lower())


class TFIDFCatalogStore:
    """
    Semantic + metadata-filtered retrieval over the SHL catalog.

    Implements the Retriever protocol.
    """

    def __init__(self, catalog_path: Path) -> None:
        with open(catalog_path) as f:
            raw = json.load(f)

        self.entries: List[CatalogEntry] = [CatalogEntry(**item) for item in raw]
        self._url_set: Set[str] = {e.url for e in self.entries}
        self._name_map: Dict[str, CatalogEntry] = {
            e.name.lower(): e for e in self.entries
        }

        # Build TF-IDF index at startup — O(n) one-time cost
        self._corpus: List[List[str]] = [
            _tokenize(e.searchable_text) for e in self.entries
        ]
        self._idf: Dict[str, float] = self._build_idf(self._corpus)
        self._doc_vectors: List[Dict[str, float]] = [
            self._tfidf_vector(tokens) for tokens in self._corpus
        ]

        logger.info(
            "TFIDFCatalogStore ready: %d entries, %d unique terms",
            len(self.entries),
            len(self._idf),
        )

    # ── Index construction ─────────────────────────────────────────────────────

    def _build_idf(self, corpus: List[List[str]]) -> Dict[str, float]:
        n = len(corpus)
        df: Dict[str, int] = defaultdict(int)
        for tokens in corpus:
            for t in set(tokens):
                df[t] += 1
        # Smoothed IDF: log((N+1)/(df+1)) + 1 — prevents zero IDF for universal terms
        return {t: math.log((n + 1) / (freq + 1)) + 1.0 for t, freq in df.items()}

    def _tfidf_vector(self, tokens: List[str]) -> Dict[str, float]:
        tf: Dict[str, float] = defaultdict(float)
        for t in tokens:
            tf[t] += 1.0
        n = len(tokens) or 1
        vec: Dict[str, float] = {}
        for t, count in tf.items():
            vec[t] = (count / n) * self._idf.get(t, 1.0)
        # L2-normalize for cosine similarity
        norm = math.sqrt(sum(v**2 for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}

    def _cosine(self, q: Dict[str, float], d: Dict[str, float]) -> float:
        return sum(q.get(t, 0.0) * v for t, v in d.items())

    # ── Retriever protocol ─────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 15,
        test_types: Optional[List[str]] = None,
        job_level: Optional[str] = None,
        language: Optional[str] = None,
    ) -> List[CatalogEntry]:
        query_vec = self._tfidf_vector(_tokenize(query))

        scored: List[Tuple[float, CatalogEntry]] = []
        for entry, doc_vec in zip(self.entries, self._doc_vectors):
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
            scored.append((self._cosine(query_vec, doc_vec), entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:top_k]]

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

    def canonical_entry(self, name: str) -> Optional[CatalogEntry]:
        return self._name_map.get(name.lower())

    @property
    def all_entries(self) -> List[CatalogEntry]:
        return self.entries

    @property
    def all_urls(self) -> Set[str]:
        return self._url_set
