"""Problem instance: demand points, candidate sites, existing shelters and walking distances."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_distances, load_sections, load_shelters

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def _minmax(x: np.ndarray) -> np.ndarray:
    span = x.max() - x.min()
    return (x - x.min()) / span if span > 0 else np.zeros_like(x, dtype=float)


@dataclass(frozen=True)
class VulnerabilityWeights:
    """Weights of the composite vulnerability index (each criterion min-max normalised)."""

    age: float = 1.0   # share of residents aged 70+
    park: float = 1.0  # distance to the nearest park (lack of green/cool space)
    heat: float = 0.0  # summer thermal discomfort (not used in the original study)

    def index(self, sections: pd.DataFrame) -> np.ndarray:
        total = self.age + self.park + self.heat
        if total <= 0:
            return np.ones(len(sections))
        v = (self.age * _minmax(sections["pct70"].to_numpy())
             + self.park * _minmax(sections["dist_park"].to_numpy())
             + self.heat * _minmax(sections["heat"].to_numpy()))
        return v / total


@dataclass(frozen=True)
class Instance:
    """Barcelona climate-shelter location instance.

    Demand points ``i`` are the census-section centroids. Candidate sites ``j`` are the same
    centroids (vertex location problem). Existing shelters are always open.

    Attributes
    ----------
    D : (n, n) walking distance in metres from section ``i`` to candidate site ``j``.
    D_exist : (n, m) walking distance from section ``i`` to existing shelter ``s``.
    """

    sections: pd.DataFrame
    shelters: pd.DataFrame
    D: np.ndarray
    D_exist: np.ndarray
    weights: VulnerabilityWeights = field(default_factory=VulnerabilityWeights)

    @classmethod
    def load(cls, data_dir: Path = DATA_DIR, weights: VulnerabilityWeights | None = None) -> Instance:
        sections = load_sections(data_dir)
        D, D_exist, codes = load_distances(data_dir)
        assert np.array_equal(codes, sections["code"].to_numpy()), "distance matrix / sections misaligned"
        return cls(sections, load_shelters(data_dir), D, D_exist, weights or VulnerabilityWeights())

    def with_weights(self, weights: VulnerabilityWeights) -> Instance:
        return Instance(self.sections, self.shelters, self.D, self.D_exist, weights)

    def subset(self, idx: np.ndarray) -> Instance:
        """Sub-instance on a subset of sections (demand = candidates = ``idx``); used for tests/benchmarks."""
        idx = np.asarray(idx)
        return Instance(self.sections.iloc[idx].reset_index(drop=True), self.shelters,
                        self.D[np.ix_(idx, idx)], self.D_exist[idx], self.weights)

    @property
    def n(self) -> int:
        return len(self.sections)

    @cached_property
    def base(self) -> np.ndarray:
        """Distance from every section to its nearest *existing* shelter."""
        return self.D_exist.min(axis=1)

    @cached_property
    def vulnerability(self) -> np.ndarray:
        return self.weights.index(self.sections)

    @cached_property
    def pop70(self) -> np.ndarray:
        return self.sections["pop70"].to_numpy(dtype=float)

    @property
    def codes(self) -> np.ndarray:
        return self.sections["code"].to_numpy()
