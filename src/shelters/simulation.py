"""Discrete-event simulation of a heat-wave day and simulation-based optimisation (SBO).

Deterministic location models assume every resident within reach is served. In a heat wave,
shelters have finite capacity and demand is uncertain, so a "good" deterministic layout can
still turn people away. This module:

1. samples stochastic demand scenarios (day temperature, Poisson arrivals per section,
   arrival times, walking tolerance, length of stay);
2. simulates each shelter as a loss system (M/G/c/c-like): people walk to the nearest open
   shelter within their tolerance; if it is full they try the next one, otherwise they go home;
3. optimises the location of ``k`` new shelters by a simulation-guided swap search whose
   objective is the *simulated* expected unmet demand, using common random numbers across
   candidates and a final re-evaluation on fresh scenarios (selection of the best).

All behavioural parameters are explicit assumptions exposed in the app for what-if analysis.
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field

import numpy as np

from .instance import Instance


@dataclass(frozen=True)
class HeatwaveParams:
    seek_rate: float = 0.04           # share of 70+ residents seeking a shelter on a reference heat-wave day
    heat_elasticity: float = 0.6      # extra demand in the hottest vs. average section (relative)
    temp_mean: float = 35.0           # daily max temperature (deg C) ~ Normal(temp_mean, temp_sd)
    temp_sd: float = 2.0
    temp_ref: float = 35.0
    temp_slope: float = 0.15          # relative demand increase per deg C above temp_ref
    tolerance_median_m: float = 700.0 # walking tolerance ~ LogNormal(log(median), sigma)
    tolerance_sigma: float = 0.5
    stay_mean_min: float = 150.0      # length of stay ~ Exponential(mean)
    capacity_existing: int = 20
    capacity_new: int = 20
    open_min: float = 10 * 60.0
    close_min: float = 21 * 60.0
    peak_min: float = 16 * 60.0
    retry: bool = True


@dataclass
class Scenario:
    temperature: float
    section: np.ndarray    # (N,) section index of each person, sorted by arrival time
    arrival: np.ndarray    # minutes
    tolerance: np.ndarray  # metres
    stay: np.ndarray       # minutes


@dataclass
class SimResult:
    demand: int
    served: int
    too_far: int
    rejected_full: int
    served_by_facility: np.ndarray
    peak_by_facility: np.ndarray
    unmet_by_section: np.ndarray

    @property
    def unmet_share(self) -> float:
        return (self.too_far + self.rejected_full) / max(self.demand, 1)


def sample_scenario(inst: Instance, p: HeatwaveParams, rng: np.random.Generator) -> Scenario:
    temp = rng.normal(p.temp_mean, p.temp_sd)
    mult = max(0.0, 1.0 + p.temp_slope * (temp - p.temp_ref))
    heat = inst.sections["heat"].to_numpy()
    heat_n = (heat - heat.mean()) / (heat.max() - heat.mean() + 1e-9)
    lam = inst.pop70 * p.seek_rate * mult * np.clip(1.0 + p.heat_elasticity * heat_n, 0.0, None)
    counts = rng.poisson(lam)
    section = np.repeat(np.arange(inst.n), counts)
    N = section.size
    arrival = rng.triangular(p.open_min, p.peak_min, p.close_min - 30, size=N)
    tolerance = rng.lognormal(np.log(p.tolerance_median_m), p.tolerance_sigma, size=N)
    stay = rng.exponential(p.stay_mean_min, size=N)
    order = np.argsort(arrival)
    return Scenario(temp, section[order], arrival[order], tolerance[order], stay[order])


def _facility_matrix(inst: Instance, sites) -> np.ndarray:
    sites = np.asarray(sites, dtype=int)
    return np.hstack([inst.D_exist, inst.D[:, sites]]) if sites.size else inst.D_exist


def simulate(inst: Instance, sites, scenario: Scenario, p: HeatwaveParams) -> SimResult:
    """Event-driven simulation of one day for a given set of new shelter sites."""
    Df = _facility_matrix(inst, sites)
    m = inst.D_exist.shape[1]
    nf = Df.shape[1]
    capacity = np.r_[np.full(m, p.capacity_existing), np.full(nf - m, p.capacity_new)].tolist()
    ranked = np.argsort(Df, axis=1)[:, :3]                     # 3 nearest shelters per section
    ranked_d = np.take_along_axis(Df, ranked, axis=1)

    occupancy = [0] * nf
    peak = [0] * nf
    served_f = [0] * nf
    unmet = np.zeros(inst.n, dtype=int)
    departures: list[tuple[float, int]] = []                   # (time, facility) min-heap
    served = too_far = rejected = 0
    tries = 2 if p.retry else 1

    sec, arr, tol, stay = (scenario.section.tolist(), scenario.arrival.tolist(),
                           scenario.tolerance.tolist(), scenario.stay.tolist())
    rk, rd = ranked.tolist(), ranked_d.tolist()
    for s, t, tau, dur in zip(sec, arr, tol, stay, strict=True):
        while departures and departures[0][0] <= t:            # release capacity
            _, f = heapq.heappop(departures)
            occupancy[f] -= 1
        if rd[s][0] > tau:
            too_far += 1
            unmet[s] += 1
            continue
        admitted = False
        for attempt in range(tries):
            if rd[s][attempt] > tau:
                break
            f = rk[s][attempt]
            if occupancy[f] < capacity[f]:
                occupancy[f] += 1
                peak[f] = max(peak[f], occupancy[f])
                served_f[f] += 1
                walk_min = rd[s][attempt] / 70.0               # ~4.2 km/h for older adults
                heapq.heappush(departures, (t + walk_min + dur, f))
                admitted = True
                break
        if admitted:
            served += 1
        else:
            rejected += 1
            unmet[s] += 1
    return SimResult(len(sec), served, too_far, rejected, np.array(served_f), np.array(peak), unmet)


@dataclass
class SimSummary:
    unmet_mean: float
    unmet_ci95: float
    unmet_cvar90: float
    too_far_share: float
    full_share: float
    demand_mean: float
    results: list[SimResult] = field(repr=False)


def evaluate(inst: Instance, sites, scenarios: list[Scenario], p: HeatwaveParams) -> SimSummary:
    res = [simulate(inst, sites, sc, p) for sc in scenarios]
    u = np.array([r.unmet_share for r in res])
    tail = np.sort(u)[-max(1, int(np.ceil(0.1 * u.size))):]
    dem = np.array([r.demand for r in res], dtype=float)
    return SimSummary(
        unmet_mean=float(u.mean()),
        unmet_ci95=float(1.96 * u.std(ddof=1) / np.sqrt(u.size)) if u.size > 1 else 0.0,
        unmet_cvar90=float(tail.mean()),
        too_far_share=float(np.mean([r.too_far / max(r.demand, 1) for r in res])),
        full_share=float(np.mean([r.rejected_full / max(r.demand, 1) for r in res])),
        demand_mean=float(dem.mean()),
        results=res,
    )


@dataclass
class SBOResult:
    sites: np.ndarray
    summary: SimSummary
    history: list[float]          # training-set unmet share of the incumbent per iteration
    runtime_s: float
    evaluations: int
    validated: dict[str, SimSummary]  # all compared layouts on independent validation scenarios
    best_name: str


def optimize(inst: Instance, k: int, p: HeatwaveParams, seeds: dict[str, np.ndarray],
             reps: int = 8, iterations: int = 25, proposals: int = 6, validation_reps: int = 30,
             seed: int = 0, progress=None) -> SBOResult:
    """Simulation-guided swap search with common random numbers + selection of the best.

    The simulation output is used to *propose* moves, not only to score them: the least-used
    new shelters are candidates to close, and the candidate sites whose walking catchment holds
    the most simulated unmet demand (too far or turned away) are candidates to open. Each
    proposed swap is evaluated on the same training scenarios (CRN) and the best improving
    one is accepted. Finally all layouts are re-evaluated on independent validation scenarios.
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    train = [sample_scenario(inst, p, np.random.default_rng(10_000 + r)) for r in range(reps)]
    cache: dict[tuple, SimSummary] = {}

    def sim(sites: np.ndarray) -> SimSummary:
        key = tuple(np.sort(sites))
        if key not in cache:
            cache[key] = evaluate(inst, np.array(key), train, p)
        return cache[key]

    seeds = {name: np.sort(np.asarray(s, dtype=int)) for name, s in seeds.items()}
    cur = min(seeds.values(), key=lambda s: sim(s).unmet_mean).copy()
    cur_sum = sim(cur)
    history = [cur_sum.unmet_mean]
    reach = (inst.D <= p.tolerance_median_m).astype(np.float32)
    m = inst.D_exist.shape[1]
    for it in range(iterations):
        unmet = np.sum([r.unmet_by_section for r in cur_sum.results], axis=0).astype(np.float32)
        used = np.sum([r.served_by_facility[m:] for r in cur_sum.results], axis=0)
        gain = unmet @ reach
        gain[cur] = -np.inf
        top = np.argsort(-gain)
        adds = np.r_[top[:proposals], rng.choice(top[:50], size=2, replace=False)]
        drops = np.r_[np.argsort(used)[:2], rng.integers(k)]
        best_move, best_val = None, cur_sum.unmet_mean
        for d in np.unique(drops):
            for a in np.unique(adds):
                if a in cur:
                    continue
                trial = cur.copy()
                trial[d] = a
                v = sim(trial).unmet_mean
                if v < best_val - 1e-9:
                    best_move, best_val = trial, v
        if best_move is not None:
            cur, cur_sum = np.sort(best_move), sim(best_move)
        history.append(cur_sum.unmet_mean)
        if progress:
            progress((it + 1) / iterations)

    val = validation_scenarios(inst, p, validation_reps)
    layouts = {**seeds, "Simulation-based optimisation": cur}
    validated = {name: evaluate(inst, s, val, p) for name, s in layouts.items()}
    best_name = min(validated, key=lambda nm: validated[nm].unmet_mean)
    return SBOResult(layouts[best_name], validated[best_name], history, time.perf_counter() - t0,
                     len(cache), validated, best_name)


def validation_scenarios(inst: Instance, p: HeatwaveParams, reps: int = 30) -> list[Scenario]:
    """The same independent scenarios used for selection of the best (fair comparisons)."""
    return [sample_scenario(inst, p, np.random.default_rng(90_000 + r)) for r in range(reps)]


def paired_difference(a: SimSummary, b: SimSummary) -> tuple[float, float]:
    """Mean and 95% CI half-width of unmet(a) - unmet(b) over the *same* scenarios (CRN)."""
    d = np.array([ra.unmet_share - rb.unmet_share for ra, rb in zip(a.results, b.results, strict=True)])
    return float(d.mean()), float(1.96 * d.std(ddof=1) / np.sqrt(d.size))
