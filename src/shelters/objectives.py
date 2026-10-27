"""Location objectives and solution KPIs.

Every objective is expressed as *minimisation* of a function of ``dmin``, the walking distance
from each demand point to its nearest open shelter (existing ones are always open):

* ``center``   (equity)     : max_i  v_i * dmin_i                 -- weighted p-center
* ``median``   (efficiency) : sum_i  w_i * dmin_i / sum_i w_i     -- p-median, average metres per 70+ resident
* ``coverage`` (service)    : share of 70+ residents with dmin_i > R  -- maximal covering (as uncovered share)

``score`` adds a tiny lexicographic tie-break so that local search can move across the large
plateaus of min-max / coverage objectives; ``value`` is the true objective that is reported.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .instance import Instance

MODELS = {
    "center": "Weighted p-center (equity)",
    "median": "p-median (efficiency)",
    "coverage": "Maximal covering (service level)",
}


def dmin_of(inst: Instance, sites) -> np.ndarray:
    """Distance from each section to its nearest open shelter (existing + ``sites``)."""
    sites = np.asarray(sites, dtype=int)
    if sites.size == 0:
        return inst.base.astype(np.float64)
    return np.minimum(inst.base, inst.D[:, sites].min(axis=1)).astype(np.float64)


@dataclass
class Objective:
    inst: Instance
    model: str = "center"
    radius: float = 500.0

    def __post_init__(self) -> None:
        if self.model not in MODELS:
            raise ValueError(f"unknown model {self.model!r}; choose from {list(MODELS)}")
        if self.model == "center":
            self.w = self.inst.vulnerability.astype(np.float32)
        else:
            self.w = (self.inst.pop70 / self.inst.pop70.sum()).astype(np.float32)

    # ------------------------------------------------------------------ scores
    def score_matrix(self, dmin: np.ndarray) -> np.ndarray:
        """Score of many solutions at once. ``dmin`` has shape (n,) or (n, c)."""
        w = self.w if dmin.ndim == 1 else self.w[:, None]
        if self.model == "center":
            wd = w * dmin
            return wd.max(axis=0) + 1e-3 * wd.mean(axis=0)
        if self.model == "median":
            return (w * dmin).sum(axis=0)
        uncovered = (w * (dmin > self.radius)).sum(axis=0)
        return uncovered + 1e-8 * (w * dmin).sum(axis=0)

    def value_of_dmin(self, dmin: np.ndarray) -> float:
        if self.model == "center":
            return float((self.w * dmin).max())
        if self.model == "median":
            return float((self.w * dmin).sum())
        return float((self.w * (dmin > self.radius)).sum())

    def score(self, sites) -> float:
        return float(self.score_matrix(dmin_of(self.inst, sites)))

    def value(self, sites) -> float:
        return self.value_of_dmin(dmin_of(self.inst, sites))

    @property
    def unit(self) -> str:
        return {"center": "weighted m", "median": "m", "coverage": "share uncovered"}[self.model]


def kpis(inst: Instance, sites, radius: float = 500.0) -> dict[str, float]:
    """Decision-maker friendly indicators for a set of new shelters."""
    d = dmin_of(inst, sites)
    w70 = inst.pop70
    v = inst.vulnerability
    return {
        "new_shelters": int(np.asarray(sites).size),
        "weighted_max_distance": float((v * d).max()),
        "max_walk_m": float(d.max()),
        "avg_walk_70plus_m": float((w70 * d).sum() / w70.sum()),
        "p95_walk_70plus_m": float(_weighted_quantile(d, w70, 0.95)),
        "coverage_70plus": float(w70[d <= radius].sum() / w70.sum()),
        "residents_70plus_uncovered": int(w70[d > radius].sum()),
        "sections_uncovered": int((d > radius).sum()),
    }


def _weighted_quantile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    order = np.argsort(x)
    cw = np.cumsum(w[order])
    return float(x[order][np.searchsorted(cw, q * cw[-1])])
