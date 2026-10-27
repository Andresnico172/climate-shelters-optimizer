"""Multi-objective location: equity (weighted p-center) vs efficiency (p-median).

``nsga2`` is a from-scratch NSGA-II (Deb et al., 2002) on fixed-size site sets: fast
non-dominated sorting, crowding distance, binary tournament and (mu + lambda) survival.
The initial population is seeded with the single-objective optima so the front spans
both extremes; ``hypervolume_2d`` measures front quality.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .heuristics import greedy, local_search
from .instance import Instance
from .objectives import Objective


@dataclass
class ParetoFront:
    sites: list[np.ndarray]
    F: np.ndarray            # (m, 2): [p-center value, p-median value], both minimised
    history: list[float]     # hypervolume per generation
    runtime_s: float
    ref_point: np.ndarray


def non_dominated_sort(F: np.ndarray) -> list[np.ndarray]:
    n = F.shape[0]
    le = (F[:, None, :] <= F[None, :, :]).all(axis=2)
    lt = (F[:, None, :] < F[None, :, :]).any(axis=2)
    dominates = le & lt                                   # dominates[a, b]: a dominates b
    dom_count = dominates.sum(axis=0)
    fronts, current = [], np.flatnonzero(dom_count == 0)
    while current.size:
        fronts.append(current)
        dom_count = dom_count - dominates[current].sum(axis=0)
        dom_count[current] = -1
        current = np.flatnonzero(dom_count == 0)
    assert sum(f.size for f in fronts) == n
    return fronts


def crowding_distance(F: np.ndarray) -> np.ndarray:
    n, m = F.shape
    d = np.zeros(n)
    if n <= 2:
        return np.full(n, np.inf)
    for j in range(m):
        order = np.argsort(F[:, j])
        span = F[order[-1], j] - F[order[0], j]
        d[order[[0, -1]]] = np.inf
        if span > 0:
            d[order[1:-1]] += (F[order[2:], j] - F[order[:-2], j]) / span
    return d


def hypervolume_2d(F: np.ndarray, ref: np.ndarray) -> float:
    pts = F[(F < ref).all(axis=1)]
    if pts.size == 0:
        return 0.0
    pts = pts[np.argsort(pts[:, 0])]
    hv, best_y = 0.0, ref[1]
    for x, y in pts:
        if y < best_y:
            hv += (ref[0] - x) * (best_y - y)
            best_y = y
    return hv


def nsga2(inst: Instance, k: int, pop_size: int = 60, generations: int = 80, mutation_rate: float = 0.15,
          near: int = 15, seed: int = 0) -> ParetoFront:
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    n = inst.n
    center, median = Objective(inst, "center"), Objective(inst, "median")
    wc, wm = center.w.astype(np.float64), median.w.astype(np.float64)
    nearest = np.argsort(inst.D, axis=1)[:, 1:near + 1]

    def evaluate(pop: np.ndarray) -> np.ndarray:
        dm = np.minimum(inst.base[:, None], inst.D[:, pop].min(axis=2))
        return np.column_stack([(wc[:, None] * dm).max(axis=0), wm @ dm])

    def make_child(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        common = np.intersect1d(a, b)
        pool = np.setdiff1d(np.union1d(a, b), common)
        child = np.concatenate([common, rng.choice(pool, size=k - common.size, replace=False)])
        for p in np.flatnonzero(rng.random(k) < mutation_rate):
            child[p] = rng.integers(n) if rng.random() < 0.1 else rng.choice(nearest[child[p]])
        child = np.unique(child)
        while child.size < k:
            child = np.unique(np.append(child, rng.integers(n)))
        return np.sort(child)

    seeds = [local_search(o, greedy(o, k).sites).sites for o in (center, median)]
    pop = np.array([np.sort(s) for s in seeds] + [np.sort(rng.choice(n, k, replace=False))
                                                  for _ in range(pop_size - 2)])
    F = evaluate(pop)
    ref = F.max(axis=0) * 1.05
    history = []
    for _ in range(generations):
        rank = np.empty(len(pop), dtype=int)
        crowd = np.empty(len(pop))
        for r, front in enumerate(non_dominated_sort(F)):
            rank[front] = r
            crowd[front] = crowding_distance(F[front])

        def tournament() -> np.ndarray:
            i, j = rng.choice(len(pop), 2, replace=False)
            better = i if (rank[i], -crowd[i]) < (rank[j], -crowd[j]) else j
            return pop[better]

        children = np.array([make_child(tournament(), tournament()) for _ in range(pop_size)])
        union = np.vstack([pop, children])
        union, idx = np.unique(union, axis=0, return_index=True)          # drop duplicate genomes
        FU = np.vstack([F, evaluate(children)])[idx]
        survivors = []
        for front in non_dominated_sort(FU):
            if len(survivors) + front.size <= pop_size:
                survivors.extend(front)
            else:
                cd = crowding_distance(FU[front])
                survivors.extend(front[np.argsort(-cd)[: pop_size - len(survivors)]])
                break
        pop, F = union[survivors], FU[survivors]
        history.append(hypervolume_2d(F[non_dominated_sort(F)[0]], ref))

    first = non_dominated_sort(F)[0]
    first = first[np.argsort(F[first, 0])]
    return ParetoFront([pop[i] for i in first], F[first], history, time.perf_counter() - t0, ref)
