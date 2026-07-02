"""
Retriever protocol: the interface both backends implement.

Using a Protocol (structural subtyping) rather than ABC means:
  - No forced inheritance chain
  - Any object with the right methods satisfies the interface
  - Easy to mock in tests without subclassing
"""
from __future__ import annotations

from typing import List, Optional, Protocol

from app.models.catalog import CatalogEntry


class Retriever(Protocol):
    def search(
        self,
        query: str,
        top_k: int = 15,
        test_types: Optional[List[str]] = None,
        job_level: Optional[str] = None,
        language: Optional[str] = None,
    ) -> List[CatalogEntry]:
        """Retrieve top-K entries for query with optional metadata filters."""
        ...

    def get_by_names(self, names: List[str]) -> List[CatalogEntry]:
        """Fetch specific entries by name (for comparison queries)."""
        ...

    def is_valid_url(self, url: str) -> bool:
        """Guard against hallucinated URLs."""
        ...

    def is_valid_name(self, name: str) -> bool:
        """Guard against hallucinated names."""
        ...
