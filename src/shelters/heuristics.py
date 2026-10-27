"""Constructive heuristics and metaheuristics for selecting ``k`` new shelter sites.

All algorithms share the same representation (an array of ``k`` candidate indices; existing
shelters are implicit and always open) and the same vectorised swap-neighbourhood evaluator,
so their results are directly comparable.

* ``greedy``        -- Greedy-Add
* ``local_search``  -- Teitz & Bart vertex interchange, best improvement
* ``tabu_search``   -- swap-based tabu search with aspiration and short-term memory
* ``vns``           -- basic Variable Neighbourhood Search (Hansen & Mladenovic, 1997)
* ``genetic``       -- steady elitist GA with set-preserving crossover and spatial mutation
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .objectives import Objective, dmin_of


@dataclass
class Solution:
    sites: np.ndarray
    value: float
    algorithm: str
    runtime_s: float = 0.0
    history: list[float] = field(default_factory=list)  # best true objective per iteration
    info: dict = field(default_factory=dict)

    @property
    def k(self) -> int:
        return int(self.sites.size)


class _Clock:
    def __init__(self, time_limit: float | None):
        self.t0 = time.perf_counter()
        self.limit = time_limit

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.t0

    def expired(self) -> bool:
        return self.limit is not None and self.elapsed >= self.limit


# --------------------------------------------------------------------------- neighbourhood
def swap_scores(obj: Objective, sites: np.ndarray, cand: np.ndarray) -> np.ndarray:
    """Score of every 1-swap ``sites[p] -> cand[q]``; returns a (k, len(cand)) matrix.

    Uses the classic nearest / second-nearest bookkeeping (Whitaker's fast interchange):
    removing facility ``p`` only changes demand points whose nearest open facility is ``p``.
    """
    inst = obj.inst
    M = np.column_stack([inst.base, inst.D[:, sites]])           # column 0 = existing network
    order = np.argsort(M, axis=1)[:, :2]
    rows = np.arange(M.shape[0])
    c1, d1, d2 = order[:, 0], M[rows, order[:, 0]], M[rows, order[:, 1]]
    Dc = inst.D[:, cand]
    out = np.empty((sites.size, cand.size))
    for p in range(sites.size):
        d_without = np.where(c1 == p + 1, d2, d1)
        out[p] = obj.score_matrix(np.minimum(d_without[:, None], Dc))
    return out


def _complement(n: int, sites: np.ndarray) -> np.ndarray:
    mask = np.ones(n, dtype=bool)
    mask[sites] = False
    return np.flatnonzero(mask)


def _sample(cand: np.ndarray, frac: float, rng: np.random.Generator) -> np.ndarray:
    if frac >= 1.0:
        return cand
    return rng.choice(cand, size=max(1, int(frac * cand.size)), replace=False)


# --------------------------------------------------------------------------- algorithms
def greedy(obj: Objective, k: int) -> Solution:
    """Greedy-Add: open, one at a time, the site with the largest marginal improvement."""
    clock = _Clock(None)
    inst = obj.inst
    sites: list[int] = []
    d = inst.base.astype(np.float64)
    history = []
    for _ in range(k):
        cand = _complement(inst.n, np.array(sites, dtype=int))
        scores = obj.score_matrix(np.minimum(d[:, None], inst.D[:, cand]))
        j = int(cand[np.argmin(scores)])
        sites.append(j)
        d = np.minimum(d, inst.D[:, j])
        history.append(obj.value_of_dmin(d))
    s = np.array(sites, dtype=int)
    return Solution(s, obj.value(s), "Greedy", clock.elapsed, history)


def local_search(obj: Objective, sites, sample: float = 1.0, max_iter: int = 1000,
                 rng: np.random.Generator | None = None, time_limit: float | None = None) -> Solution:
    """Vertex-interchange local search (best improvement) until a local optimum."""
    clock = _Clock(time_limit)
    rng = rng or np.random.default_rng(0)
    sites = np.array(sites, dtype=int).copy()
    cur = obj.score(sites)
    history = [obj.value(sites)]
    for _ in range(max_iter):
        cand = _sample(_complement(obj.inst.n, sites), sample, rng)
        sc = swap_scores(obj, sites, cand)
        p, q = np.unravel_index(np.argmin(sc), sc.shape)
        if sc[p, q] >= cur - 1e-9:
            break
        sites[p], cur = cand[q], sc[p, q]
        history.append(obj.value(sites))
        if clock.expired():
            break
    return Solution(sites, obj.value(sites), "Local search", clock.elapsed, history)


def tabu_search(obj: Objective, k: int, iters: int = 150, tenure: int = 10, sample: float = 0.5,
                seed: int = 0, time_limit: float | None = None, start=None) -> Solution:
    """Tabu search over the swap neighbourhood.

    A site that leaves the solution cannot re-enter (and one that enters cannot leave) for
    ``tenure`` iterations; tabu moves are allowed if they beat the best solution (aspiration).
    The best admissible move is always taken, even if it worsens the solution.
    """
    clock = _Clock(time_limit)
    rng = np.random.default_rng(seed)
    sites = greedy(obj, k).sites if start is None else np.array(start, dtype=int)
    cur = obj.score(sites)
    best, best_score = sites.copy(), cur
    tabu_until = np.zeros(obj.inst.n, dtype=int)
    history = [obj.value(best)]
    for it in range(1, iters + 1):
        cand = _sample(_complement(obj.inst.n, sites), sample, rng)
        sc = swap_scores(obj, sites, cand)
        tabu = (tabu_until[sites][:, None] > it) | (tabu_until[cand][None, :] > it)
        allowed = ~tabu | (sc < best_score - 1e-9)
        if not allowed.any():
            continue
        sc = np.where(allowed, sc, np.inf)
        p, q = np.unravel_index(np.argmin(sc), sc.shape)
        out_site, in_site = sites[p], cand[q]
        sites[p], cur = in_site, sc[p, q]
        t = tenure + int(rng.integers(0, 3))
        tabu_until[out_site] = it + t
        tabu_until[in_site] = it + t
        if cur < best_score - 1e-9:
            best, best_score = sites.copy(), cur
        history.append(obj.value(best))
        if clock.expired():
            break
    return Solution(best, obj.value(best), "Tabu search", clock.elapsed, history,
                    {"iterations": len(history) - 1})


def vns(obj: Objective, k: int, iters: int = 30, r_max: int = 4, sample: float = 0.35,
        seed: int = 0, time_limit: float | None = None) -> Solution:
    """Basic VNS: shake by ``r`` random swaps, local search, move or widen the neighbourhood."""
    clock = _Clock(time_limit)
    rng = np.random.default_rng(seed)
    best = local_search(obj, greedy(obj, k).sites, sample=sample, rng=rng).sites
    best_score = obj.score(best)
    history = [obj.value(best)]
    r = 1
    for _ in range(iters):
        trial = best.copy()
        pos = rng.choice(k, size=min(r, k), replace=False)
        trial[pos] = rng.choice(_complement(obj.inst.n, best), size=pos.size, replace=False)
        trial = local_search(obj, trial, sample=sample, rng=rng).sites
        s = obj.score(trial)
        if s < best_score - 1e-9:
            best, best_score, r = trial, s, 1
        else:
            r = r + 1 if r < r_max else 1
        history.append(obj.value(best))
        if clock.expired():
            break
    return Solution(best, obj.value(best), "VNS", clock.elapsed, history)


def genetic(obj: Objective, k: int, pop_size: int = 40, generations: int = 120,
            mutation_rate: float = 0.15, elite: int = 2, tournament: int = 3, near: int = 15,
            seed: int = 0, memetic: bool = True, time_limit: float | None = None) -> Solution:
    """Elitist genetic algorithm on fixed-size site sets.

    * crossover keeps the sites shared by both parents and fills up from their symmetric difference;
    * mutation replaces a site by one of its ``near`` closest candidates (spatial move) or,
      with low probability, by a random site (diversification);
    * ``memetic=True`` polishes the final best individual with local search.
    """
    clock = _Clock(time_limit)
    rng = np.random.default_rng(seed)
    inst = obj.inst
    n = inst.n
    nearest = np.argsort(inst.D, axis=1)[:, 1:near + 1]

    def fitness(pop: np.ndarray) -> np.ndarray:
        dm = np.minimum(inst.base[:, None], inst.D[:, pop].min(axis=2))   # (n, P)
        return obj.score_matrix(dm)

    def random_individual() -> np.ndarray:
        return rng.choice(n, size=k, replace=False)

    def repair(ind: np.ndarray) -> np.ndarray:
        ind = np.unique(ind)
        while ind.size < k:
            ind = np.unique(np.append(ind, rng.integers(n)))
        return ind

    pop = np.array([greedy(obj, k).sites] + [random_individual() for _ in range(pop_size - 1)])
    fit = fitness(pop)
    history = [obj.value(pop[np.argmin(fit)])]
    for _ in range(generations):
        order = np.argsort(fit)
        children = [pop[i].copy() for i in order[:elite]]
        while len(children) < pop_size:
            a, b = (pop[min(rng.choice(pop_size, tournament), key=lambda i: fit[i])] for _ in range(2))
            common = np.intersect1d(a, b)
            pool = np.setdiff1d(np.union1d(a, b), common)
            child = np.concatenate([common, rng.choice(pool, size=k - common.size, replace=False)])
            for p in np.flatnonzero(rng.random(k) < mutation_rate):
                child[p] = rng.integers(n) if rng.random() < 0.1 else rng.choice(nearest[child[p]])
            children.append(repair(child))
        pop = np.array(children)
        fit = fitness(pop)
        history.append(min(history[-1], obj.value(pop[np.argmin(fit)])))
        if clock.expired():
            break
    best = pop[np.argmin(fit)]
    if memetic:
        best = local_search(obj, best, sample=0.5, rng=rng).sites
        history.append(obj.value(best))
    return Solution(np.sort(best), obj.value(best), "Genetic algorithm", clock.elapsed, history)


ALGORITHMS = {
    "Greedy": lambda obj, k, seed=0, **kw: greedy(obj, k),
    "Local search": lambda obj, k, seed=0, **kw: _ls_from_greedy(obj, k, seed, **kw),
    "Tabu search": lambda obj, k, seed=0, **kw: tabu_search(obj, k, seed=seed, **kw),
    "VNS": lambda obj, k, seed=0, **kw: vns(obj, k, seed=seed, **kw),
    "Genetic algorithm": lambda obj, k, seed=0, **kw: genetic(obj, k, seed=seed, **kw),
}


def _ls_from_greedy(obj: Objective, k: int, seed: int, **kw) -> Solution:
    clock = _Clock(None)
    g = greedy(obj, k)
    sol = local_search(obj, g.sites, rng=np.random.default_rng(seed), **kw)
    sol.history = g.history[-1:] + sol.history[1:]
    sol.runtime_s = clock.elapsed
    return sol


def run(name: str, obj: Objective, k: int, seed: int = 0, **kw) -> Solution:
    sol = ALGORITHMS[name](obj, k, seed=seed, **kw)
    sol.sites = np.sort(sol.sites)
    assert sol.sites.size == k and np.unique(sol.sites).size == k
    # recompute to make sure the reported value matches the returned set
    sol.value = obj.value_of_dmin(dmin_of(obj.inst, sol.sites))
    return sol
