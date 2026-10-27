"""Benchmark every algorithm against exact references and precompute the marginal-benefit curve.

Outputs (read by the app):
  data/results/benchmark.csv  -- objective, optimality gap and runtime per model / k / algorithm / seed
  data/results/marginal.csv   -- exact optimum vs number of new shelters (p-center, MCLP)
Every heuristic run is also logged to the ``optimisation_run`` table of data/processed/bcn.sqlite.

Usage:  python scripts/benchmark.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shelters import Instance, Objective  # noqa: E402
from shelters import exact as E  # noqa: E402
from shelters import heuristics as H  # noqa: E402
from shelters.data import log_run  # noqa: E402

OUT = ROOT / "data" / "results"
DATA = ROOT / "data" / "processed"
RADIUS = 500.0
KS = (10, 30, 60)
SEEDS = (0, 1, 2)


def reference(inst: Instance, obj: Objective, k: int) -> tuple[float, str]:
    if obj.model == "center":
        r = E.p_center(obj, k)
        return r.value, "optimal (CP-SAT)"
    if obj.model == "coverage":
        r = E.max_covering(inst, k, RADIUS)
        return r.value, "optimal (CP-SAT)"
    r = E.p_median_lagrangian(obj, k, upper=H.run("Tabu search", obj, k))
    return r.lower_bound, "Lagrangian lower bound"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    inst = Instance.load()
    rows = []
    for model in ("center", "median", "coverage"):
        obj = Objective(inst, model, radius=RADIUS)
        for k in KS:
            ref, ref_kind = reference(inst, obj, k)
            for name in H.ALGORITHMS:
                for seed in SEEDS if name != "Greedy" else (0,):
                    sol = H.run(name, obj, k, seed=seed)
                    gap = (sol.value - ref) / ref if ref > 0 else 0.0
                    rows.append(dict(model=model, k=k, algorithm=name, seed=seed, value=sol.value,
                                     reference=ref, reference_kind=ref_kind, gap=gap, runtime_s=sol.runtime_s))
                    log_run(DATA, model, name, k, sol.value, sol.runtime_s, inst.codes[sol.sites].tolist())
                    print(f"{model:8s} k={k:3d} {name:18s} seed={seed} value={sol.value:.5g} "
                          f"gap={100 * gap:6.2f}% t={sol.runtime_s:.2f}s")
    pd.DataFrame(rows).to_csv(OUT / "benchmark.csv", index=False)

    center = Objective(inst, "center")
    marg = []
    for k in range(0, 81, 2):
        pc = E.p_center(center, k).value if k else center.value([])
        mc = E.max_covering(inst, k, RADIUS).value if k else Objective(inst, "coverage", radius=RADIUS).value([])
        marg.append(dict(k=k, p_center=pc, coverage_500m=1 - mc))
        print(f"k={k:3d} p-center={pc:8.2f} coverage={1 - mc:.3f}")
    pd.DataFrame(marg).to_csv(OUT / "marginal.csv", index=False)


if __name__ == "__main__":
    main()
