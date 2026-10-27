"""Exact models with Google OR-Tools (CP-SAT) and a Lagrangian bound for the p-median.

* ``min_sites_for_radius`` -- set covering (Toregas et al., 1971): fewest new shelters so that
  every section is within ``R`` metres (or within weighted radius ``R``).
* ``p_center``             -- exact weighted vertex p-center via binary search over candidate radii,
  solving a set-covering feasibility problem at each step (Handler & Mirchandani; Daskin, 2000).
  This avoids the O(n^2) assignment variables of the textbook MIP, which do not scale to n = 1068.
* ``max_covering``         -- maximal covering location problem (Church & ReVelle, 1974).
* ``p_median_lagrangian``  -- Lagrangian relaxation of the assignment constraints with subgradient
  optimisation; gives a lower bound and a feasible solution, hence a certified optimality gap.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from ortools.sat.python import cp_model

from .heuristics import Solution, greedy, local_search
from .instance import Instance
from .objectives import Objective


@dataclass
class ExactResult:
    sites: np.ndarray | None
    value: float | None
    status: str
    runtime_s: float
    lower_bound: float | None = None
    info: dict = field(default_factory=dict)

    @property
    def gap(self) -> float | None:
        if self.value is None or self.lower_bound is None or self.value == 0:
            return None
        return (self.value - self.lower_bound) / self.value


def _solver(time_limit: float, workers: int = 8) -> cp_model.CpSolver:
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = time_limit
    s.parameters.num_workers = workers
    return s


def _cover_sets(weighted_D: np.ndarray, weighted_base: np.ndarray, radius: float):
    """Rows that still need covering and, for each, the candidate sites that cover it."""
    need = np.flatnonzero(weighted_base > radius)
    return need, [np.flatnonzero(weighted_D[i] <= radius) for i in need]


# --------------------------------------------------------------------------- set covering
def min_sites_for_radius(inst: Instance, radius: float, weighted: bool = False,
                         time_limit: float = 30.0) -> ExactResult:
    t0 = time.perf_counter()
    v = inst.vulnerability if weighted else np.ones(inst.n)
    need, covers = _cover_sets(v[:, None] * inst.D, v * inst.base, radius)
    if any(c.size == 0 for c in covers):
        return ExactResult(None, None, "INFEASIBLE", time.perf_counter() - t0)
    m = cp_model.CpModel()
    x = [m.NewBoolVar(f"x{j}") for j in range(inst.n)]
    for c in covers:
        m.AddBoolOr([x[j] for j in c])
    m.Minimize(sum(x))
    s = _solver(time_limit)
    st = s.Solve(m)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return ExactResult(None, None, s.StatusName(st), time.perf_counter() - t0)
    sites = np.array([j for j in range(inst.n) if s.Value(x[j])], dtype=int)
    return ExactResult(sites, float(sites.size), s.StatusName(st), time.perf_counter() - t0,
                       lower_bound=s.BestObjectiveBound(), info={"rows": int(need.size)})


# --------------------------------------------------------------------------- p-center
def _cover_feasible(WD, wb, radius, k, time_limit, hint=None):
    need, covers = _cover_sets(WD, wb, radius)
    if any(c.size == 0 for c in covers):
        return False, None
    if need.size == 0:
        return True, np.array([], dtype=int)
    m = cp_model.CpModel()
    useful = np.unique(np.concatenate(covers))
    x = {j: m.NewBoolVar(f"x{j}") for j in useful}
    for c in covers:
        m.AddBoolOr([x[j] for j in c])
    m.Add(sum(x.values()) <= k)
    if hint is not None:
        for j in hint:
            if j in x:
                m.AddHint(x[j], 1)
    s = _solver(time_limit)
    st = s.Solve(m)
    if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return True, np.array([j for j in useful if s.Value(x[j])], dtype=int)
    if st == cp_model.INFEASIBLE:
        return False, None
    raise TimeoutError(f"set-covering subproblem undecided at radius {radius:.2f} ({s.StatusName(st)})")


def p_center(obj: Objective, k: int, time_limit: float = 120.0, upper: Solution | None = None) -> ExactResult:
    """Exact weighted p-center (existing shelters fixed open)."""
    assert obj.model == "center"
    t0 = time.perf_counter()
    inst = obj.inst
    v = obj.w.astype(np.float64)
    WD, wb = v[:, None] * inst.D, v * inst.base
    if upper is None:
        upper = local_search(obj, greedy(obj, k).sites)
    ub, best = upper.value, upper.sites

    # The optimum equals some v_i * d_ij or v_i * base_i; search the sorted candidate radii <= UB.
    radii = np.unique(np.concatenate([WD[WD <= ub + 1e-9], wb[wb <= ub + 1e-9]]))
    lo, hi, steps = 0, radii.size - 1, 0
    while lo < hi:
        mid = (lo + hi) // 2
        remaining = time_limit - (time.perf_counter() - t0)
        if remaining <= 0:
            return ExactResult(best, ub, "TIME_LIMIT", time.perf_counter() - t0,
                               lower_bound=float(radii[lo]), info={"steps": steps})
        try:
            ok, sites = _cover_feasible(WD, wb, radii[mid], k, remaining, hint=best)
        except TimeoutError:
            return ExactResult(best, obj.value(_pad(best, k, inst.n)), "TIME_LIMIT", time.perf_counter() - t0,
                               lower_bound=float(radii[lo]), info={"steps": steps})
        steps += 1
        if ok:
            hi, best = mid, sites
        else:
            lo = mid + 1
    best = _pad(best, k, inst.n)
    val = obj.value(best)
    return ExactResult(best, val, "OPTIMAL", time.perf_counter() - t0, lower_bound=val,
                       info={"binary_search_steps": steps, "candidate_radii": int(radii.size)})


def _pad(sites: np.ndarray, k: int, n: int) -> np.ndarray:
    """A cover may need fewer than k sites; fill with arbitrary unused sites (objective unchanged)."""
    sites = np.unique(sites)
    if sites.size < k:
        rest = np.setdiff1d(np.arange(n), sites)[: k - sites.size]
        sites = np.sort(np.concatenate([sites, rest]))
    return sites


# --------------------------------------------------------------------------- maximal covering
def max_covering(inst: Instance, k: int, radius: float, time_limit: float = 60.0) -> ExactResult:
    """Maximise residents 70+ within ``radius`` metres of a shelter using ``k`` new sites."""
    t0 = time.perf_counter()
    demand = inst.pop70.astype(int)
    need, covers = _cover_sets(inst.D, inst.base, radius)
    m = cp_model.CpModel()
    x = [m.NewBoolVar(f"x{j}") for j in range(inst.n)]
    z = {}
    for i, c in zip(need, covers, strict=True):
        if demand[i] == 0 or c.size == 0:
            continue
        z[i] = m.NewBoolVar(f"z{i}")
        m.AddBoolOr([x[j] for j in c]).OnlyEnforceIf(z[i])
    m.Add(sum(x) == k)
    m.Maximize(sum(int(demand[i]) * zi for i, zi in z.items()))
    s = _solver(time_limit)
    st = s.Solve(m)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return ExactResult(None, None, s.StatusName(st), time.perf_counter() - t0)
    sites = np.array([j for j in range(inst.n) if s.Value(x[j])], dtype=int)
    total = demand.sum()
    already = demand[inst.base <= radius].sum()
    uncovered = 1 - (already + s.ObjectiveValue()) / total
    uncovered_lb = 1 - (already + s.BestObjectiveBound()) / total
    return ExactResult(sites, float(uncovered), s.StatusName(st), time.perf_counter() - t0,
                       lower_bound=float(uncovered_lb))


# --------------------------------------------------------------------------- p-median bound
def p_median_lagrangian(obj: Objective, k: int, iters: int = 400, upper: Solution | None = None,
                        delta0: float = 2.0, patience: int = 30) -> ExactResult:
    """Lagrangian relaxation of  sum_j y_ij + y_i0 = 1  (y_i0: served by the existing network).

    For multipliers lambda:  L(lambda) = sum_i lambda_i + sum_i min(0, c_i0 - lambda_i)
                                         + sum of the k most negative  V_j = sum_i min(0, c_ij - lambda_i)
    with c_ij = w_i d_ij.  Multipliers follow the subgradient rule of Fisher (1981); each relaxed
    solution's open set is evaluated as a feasible solution (Lagrangian heuristic).
    """
    assert obj.model == "median"
    t0 = time.perf_counter()
    inst = obj.inst
    w = obj.w.astype(np.float64)
    C = w[:, None] * inst.D                                   # (n, n) assignment costs
    c0 = w * inst.base
    if upper is None:
        upper = local_search(obj, greedy(obj, k).sites)
    best_sites, ub = upper.sites.copy(), upper.value
    lam = np.minimum(c0, np.sort(C, axis=1)[:, 1]) * 1.0
    lb, delta, stall = -np.inf, delta0, 0
    history = []
    for _ in range(iters):
        R = np.minimum(0.0, C - lam[:, None])
        V = R.sum(axis=0)
        open_ = np.argpartition(V, k)[:k]
        L = lam.sum() + np.minimum(0.0, c0 - lam).sum() + V[open_].sum()
        if L > lb + 1e-12:
            lb, stall = L, 0
        else:
            stall += 1
            if stall >= patience:
                delta, stall = delta / 2, 0
        val = obj.value(open_)
        if val < ub:
            ub, best_sites = val, np.sort(open_)
        history.append((lb, ub))
        g = 1.0 - (C[:, open_] - lam[:, None] < 0).sum(axis=1) - (c0 - lam < 0)
        norm = float((g ** 2).sum())
        if norm == 0 or delta < 1e-4 or (ub - lb) / ub < 1e-4:
            break
        lam = np.maximum(0.0, lam + delta * (ub - L) / norm * g)
    return ExactResult(best_sites, float(ub), "LAGRANGIAN", time.perf_counter() - t0,
                       lower_bound=float(lb), info={"iterations": len(history), "history": history})
