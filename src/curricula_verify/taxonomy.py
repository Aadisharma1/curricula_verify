"""Safety taxonomy (config-driven)."""

from __future__ import annotations

import json
from pathlib import Path


class Taxonomy:
    def __init__(self, categories: dict[str, int], severity_scale: dict[str, int] | None = None):
        if not categories:
            raise ValueError("taxonomy needs at least one category")
        for name, sev in categories.items():
            if not isinstance(sev, int) or sev < 0:
                raise ValueError(f"severity for {name} must be a non-negative int")
        self.categories = dict(categories)
        self.severity_scale = severity_scale or {"min": 0, "max": max(categories.values())}

    @classmethod
    def from_file(cls, path: str | Path) -> "Taxonomy":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(raw["categories"], raw.get("severity_scale"))

    def severity(self, category: str) -> int:
        if category not in self.categories:
            raise KeyError(f"unknown category: {category}")
        return self.categories[category]

    def unsafe_categories(self) -> list[str]:
        return [c for c, s in self.categories.items() if s > 0]
