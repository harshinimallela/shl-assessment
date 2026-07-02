"""
CatalogEntry: domain model for a single SHL assessment.

searchable_text is the field the retriever indexes — it concatenates
all human-readable fields so both keyword and semantic search have
maximum signal.
"""
from __future__ import annotations

from typing import List

from pydantic import BaseModel, computed_field


class CatalogEntry(BaseModel):
    name: str
    url: str
    description: str
    test_type: str          # A=Ability, P=Personality, K=Knowledge, S=Simulation, B=Behavioral
    test_type_label: str = ""
    job_levels: List[str] = []
    duration_minutes: int = 0
    languages: List[str] = []
    keywords: List[str] = []

    @computed_field  # type: ignore[misc]
    @property
    def searchable_text(self) -> str:
        """Concatenated text for retrieval indexing."""
        parts = [
            self.name,
            self.description,
            self.test_type_label,
            " ".join(self.keywords),
            " ".join(self.job_levels),
        ]
        return " ".join(p for p in parts if p)
