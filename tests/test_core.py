"""Correctness tests. Exact and heuristic results are checked against brute-force enumeration
on a small sub-instance (40 sections of the Ciutat Vella district)."""

from __future__ import annotations

from dataclasses import replace
from itertools import combinations

import numpy as np
import pytest

from shelters import Instance, Objective, dmin_of, kpis
from shelters import exact as E
from shelters import heuristics as H
from shelters import multiobjective as MO
from shelters import simulation as S


@pytest.fixture(scope="session")
def full() -> Instance:
    return Instance.load()


@pytest.fixture(scope="session")
def small(full) -> Instance:
    return full.subset(np.arange(40))


def brute_force(obj: Objective, k: int) -> float:
    return min(obj.value(np.array(c)) for c in combinations(range(obj.inst.n), k))


# --------------------------------------------------------------------------- data & objectives
def test_instance_shapes(full):
    assert full.n == 1068
    assert full.D.shape == (1068, 1068) and full.D_exist.shape == (1068, len(full.shelters))
    assert np.allclose(np.diag(full.D), 0)
    assert (full.D >= 0).all() and np.isfinite(full.D).all()
    assert full.pop70.sum() == 233_545


def test_walking_distance_not_shorter_than_straight_line(full):
    xy = full.sections[["x", "y"]].to_numpy()
    euclid = np.linalg.norm(xy[:, None] - xy[None], axis=2)
    assert (full.D + 1e-3 >= euclid).mean() > 0.999


def test_adding_a_shelter_never_hurts(full):
    for model in ("center", "median", "coverage"):
        obj = Objective(full, model)
        assert obj.value([5]) <= obj.value([]) + 1e-9
        assert obj.value([5, 900]) <= obj.value([5]) + 1e-9


def test_kpis_baseline(full):
    k = kpis(full, [])
    assert 0 < k["coverage_70plus"] < 1
    assert k["max_walk_m"] >= k["p95_walk_70plus_m"] >= 0


# --------------------------------------------------------------------------- heuristics
def test_swap_scores_match_direct_evaluation(full):
    obj = Objective(full, "median")
    sites = np.array([10, 200, 500, 800])
    cand = np.array([1, 50, 999])
    sc = H.swap_scores(obj, sites, cand)
    for p in range(sites.size):
        for q, c in enumerate(cand):
            trial = sites.copy()
            trial[p] = c
            assert sc[p, q] == pytest.approx(obj.score(trial), rel=1e-5)


@pytest.mark.parametrize("name", list(H.ALGORITHMS))
@pytest.mark.parametrize("model", ["center", "median", "coverage"])
def test_algorithms_return_valid_solutions(small, name, model):
    obj = Objective(small, model, radius=300)
    sol = H.run(name, obj, 3, seed=1)
    assert sol.sites.size == 3 and np.unique(sol.sites).size == 3
    assert sol.value == pytest.approx(obj.value(sol.sites))
    assert sol.value >= brute_force(obj, 3) - 1e-9


@pytest.mark.parametrize("model", ["center", "median"])
def test_local_search_improves_greedy(small, model):
    obj = Objective(small, model)
    g = H.greedy(obj, 4)
    ls = H.local_search(obj, g.sites)
    assert ls.value <= g.value + 1e-9


@pytest.mark.parametrize("model", ["center", "median"])
def test_tabu_finds_optimum_on_small_instance(small, model):
    obj = Objective(small, model)
    assert H.run("Tabu search", obj, 2, sample=1.0).value == pytest.approx(brute_force(obj, 2), rel=1e-6)


# --------------------------------------------------------------------------- exact
@pytest.mark.parametrize("k", [1, 2, 3])
def test_p_center_exact_matches_brute_force(small, k):
    obj = Objective(small, "center")
    res = E.p_center(obj, k)
    assert res.status == "OPTIMAL"
    assert res.value == pytest.approx(brute_force(obj, k), rel=1e-6)


def test_p_center_exact_beats_heuristics_on_full_instance(full):
    obj = Objective(full, "center")
    res = E.p_center(obj, 15)
    assert res.status == "OPTIMAL"
    assert res.value <= H.run("VNS", obj, 15).value + 1e-6


@pytest.mark.parametrize("k", [1, 2, 3])
def test_max_covering_matches_brute_force(small, k):
    obj = Objective(small, "coverage", radius=300)
    res = E.max_covering(small, k, 300)
    assert res.value == pytest.approx(brute_force(obj, k), rel=1e-6)
    assert obj.value(res.sites) == pytest.approx(res.value, rel=1e-6)


def test_lagrangian_bound_is_valid(small):
    obj = Objective(small, "median")
    opt = brute_force(obj, 3)
    res = E.p_median_lagrangian(obj, 3)
    assert res.lower_bound <= opt + 1e-9 <= res.value + 1e-9


def test_set_covering_radius(full):
    res = E.min_sites_for_radius(full, 800)
    d = dmin_of(full, res.sites)
    assert (d <= 800 + 1e-6).all()
    assert res.status == "OPTIMAL"


# --------------------------------------------------------------------------- multi-objective
def test_non_dominated_sort_and_hypervolume():
    F = np.array([[1, 5], [2, 2], [5, 1], [3, 3], [6, 6]], dtype=float)
    fronts = MO.non_dominated_sort(F)
    assert set(fronts[0]) == {0, 1, 2}
    assert set(fronts[1]) == {3}
    assert MO.hypervolume_2d(np.array([[1.0, 1.0]]), np.array([2.0, 2.0])) == pytest.approx(1.0)
    assert MO.hypervolume_2d(np.array([[0, 1.0], [1.0, 0]]), np.array([2.0, 2.0])) == pytest.approx(3.0)


def test_nsga2_front_is_mutually_non_dominated(full):
    pf = MO.nsga2(full, 8, pop_size=24, generations=10)
    assert len(MO.non_dominated_sort(pf.F)[0]) == len(pf.F)
    assert pf.history[-1] >= pf.history[0]


# --------------------------------------------------------------------------- simulation
def test_simulation_conserves_people(full):
    p = S.HeatwaveParams()
    sc = S.sample_scenario(full, p, np.random.default_rng(0))
    r = S.simulate(full, [1, 2, 3], sc, p)
    assert r.demand == r.served + r.too_far + r.rejected_full == sc.section.size
    assert r.unmet_by_section.sum() == r.too_far + r.rejected_full
    assert (r.peak_by_facility[:len(full.shelters)] <= p.capacity_existing).all()


def test_unlimited_capacity_means_no_rejections(full):
    p = replace(S.HeatwaveParams(), capacity_existing=10**6, capacity_new=10**6)
    sc = S.sample_scenario(full, p, np.random.default_rng(1))
    assert S.simulate(full, [], sc, p).rejected_full == 0


def test_more_shelters_reduce_unmet_demand(full):
    p = S.HeatwaveParams()
    scen = S.validation_scenarios(full, p, 5)
    sites = E.max_covering(full, 20, p.tolerance_median_m).sites
    assert S.evaluate(full, sites, scen, p).unmet_mean < S.evaluate(full, [], scen, p).unmet_mean
