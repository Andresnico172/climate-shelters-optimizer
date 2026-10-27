"""Climate Shelters BCN -- interactive decision-support app.

Run locally:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from shelters import Instance, Objective, VulnerabilityWeights, dmin_of, kpis  # noqa: E402
from shelters import exact as E  # noqa: E402
from shelters import heuristics as H  # noqa: E402
from shelters import multiobjective as MO  # noqa: E402
from shelters import simulation as S  # noqa: E402
from shelters.data import RUNS_SQL, load_geojson, run_summary  # noqa: E402
from shelters.instance import DATA_DIR  # noqa: E402
from ui import ACCENT, COOL, INK, WARM, convergence, legend, polygons, section_map  # noqa: E402

st.set_page_config(page_title="Climate Shelters BCN · Optimiser", page_icon="🌡️", layout="wide")

EXACT = {
    "center": "Exact · OR-Tools CP-SAT (binary search + set covering)",
    "coverage": "Exact · OR-Tools CP-SAT (maximal covering MIP)",
    "median": "Lagrangian relaxation (lower bound + heuristic)",
}
MODEL_LABEL = {
    "center": "Equity · weighted p-center (minimise the worst vulnerability-weighted walk)",
    "median": "Efficiency · p-median (minimise the average walk of residents 70+)",
    "coverage": "Service level · maximal covering (maximise residents 70+ within R metres)",
}
RESULTS = ROOT / "data" / "results"


# --------------------------------------------------------------------------- cached data / solvers
@st.cache_resource(show_spinner="Loading Barcelona instance…")
def base_instance() -> Instance:
    return Instance.load()


@st.cache_resource
def map_polygons() -> pd.DataFrame:
    return polygons(load_geojson(DATA_DIR), base_instance().codes)


def instance(w: tuple[float, float, float]) -> Instance:
    return base_instance().with_weights(VulnerabilityWeights(*w))


@st.cache_data(show_spinner=False, max_entries=64)
def solve(model: str, algorithm: str, k: int, radius: float, w: tuple, seed: int) -> dict:
    inst = instance(w)
    obj = Objective(inst, model, radius=radius)
    if algorithm == EXACT[model]:
        if model == "center":
            r = E.p_center(obj, k)
        elif model == "coverage":
            r = E.max_covering(inst, k, radius)
        else:
            r = E.p_median_lagrangian(obj, k, upper=H.run("Tabu search", obj, k, seed=seed))
        hist = [h[1] for h in r.info.get("history", [])]
        return dict(sites=np.sort(r.sites), value=r.value, lb=r.lower_bound, status=r.status,
                    runtime=r.runtime_s, history=hist, info={k_: v for k_, v in r.info.items() if k_ != "history"})
    sol = H.run(algorithm, obj, k, seed=seed)
    return dict(sites=sol.sites, value=sol.value, lb=None, status="HEURISTIC", runtime=sol.runtime_s,
                history=sol.history, info=sol.info)


@st.cache_data(show_spinner=False, max_entries=16)
def pareto(k: int, w: tuple, generations: int) -> dict:
    inst = instance(w)
    pf = MO.nsga2(inst, k, generations=generations)
    c, m = Objective(inst, "center"), Objective(inst, "median")
    pc = E.p_center(c, k).sites
    pm = H.run("Tabu search", m, k).sites
    anchors = {"Exact p-center": (c.value(pc), m.value(pc)), "p-median (tabu)": (c.value(pm), m.value(pm))}
    return dict(sites=pf.sites, F=pf.F, hv=pf.history, runtime=pf.runtime_s, anchors=anchors)


def fmt_pct(x: float) -> str:
    return f"{100 * x:.1f}%"


# --------------------------------------------------------------------------- sidebar
st.sidebar.markdown("## 🌡️ Climate Shelters BCN")
st.sidebar.caption("Where should Barcelona open its next climate shelters?")
k = st.sidebar.slider("New shelters to open (k)", 1, 80, 20)
radius = st.sidebar.slider("Walking-distance target R (m)", 300, 1000, 500, 50,
                           help="Used by the covering model and the coverage KPIs.")
with st.sidebar.expander("Vulnerability index (p-center weights)"):
    st.caption("Composite index of min-max normalised criteria. The original study used age + park.")
    wa = st.slider("Share of residents aged 70+", 0.0, 1.0, 1.0, 0.1)
    wp = st.slider("Distance to nearest park", 0.0, 1.0, 1.0, 0.1)
    wh = st.slider("Summer heat exposure", 0.0, 1.0, 0.0, 0.1)
W = (wa, wp, wh)
st.sidebar.divider()
st.sidebar.caption("Data: Barcelona Open Data (Ajuntament de Barcelona, CC BY 4.0). "
                   "Walking distances on the city street graph (9,483 nodes).")

inst = instance(W)
poly = map_polygons()
base_k = kpis(inst, [], radius)

st.title("Optimal location of climate shelters in Barcelona")
st.markdown(
    "Climate shelters are cooled public spaces (libraries, civic centres, parks) where vulnerable people can "
    "escape extreme heat. This app decides **where to open new ones** using exact optimisation (OR-Tools), "
    "metaheuristics, multi-objective search and a **discrete-event simulation of a heat-wave day**."
)

tabs = st.tabs(["📍 Problem", "⚙️ Optimise", "📊 Benchmark", "⚖️ Equity vs efficiency",
                "🌡️ Heat-wave simulation", "📐 Methodology"])

# --------------------------------------------------------------------------- 1. problem
with tabs[0]:
    c = st.columns(4)
    c[0].metric("Residents aged 70+", f"{int(inst.pop70.sum()):,}")
    c[1].metric("Existing climate shelters", len(inst.shelters))
    c[2].metric(f"70+ within {radius} m walk", fmt_pct(base_k["coverage_70plus"]))
    c[3].metric("Average walk (70+)", f"{base_k['avg_walk_70plus_m']:.0f} m",
                help=f"Worst-served section: {base_k['max_walk_m']:,.0f} m")

    layer = st.segmented_control(
        "Map layer", ["Walk to nearest shelter", "Vulnerability index", "Share aged 70+",
                      "Heat exposure", "Distance to park"], default="Walk to nearest shelter")
    layer = layer or "Walk to nearest shelter"
    spec = {
        "Walk to nearest shelter": (inst.base, "walk", "{:.0f} m", WARM, 1500, "0 m", "≥1500 m"),
        "Vulnerability index": (inst.vulnerability, "index", "{:.2f}", COOL, None, "low", "high"),
        "Share aged 70+": (inst.sections["pct70"].to_numpy(), "70+", "{:.1%}", COOL, None, "low", "high"),
        "Heat exposure": (inst.sections["heat"].to_numpy(), "heat", "{:.1f} / 6", WARM, None, "cool", "hot"),
        "Distance to park": (inst.sections["dist_park"].to_numpy(), "park", "{:.0f} m", COOL, 1200, "0 m", "≥1200 m"),
    }[layer]
    vals, lab, f, pal, vmax, lo, hi = spec
    st.pydeck_chart(section_map(poly, inst.sections, vals, lab, f, pal, vmax, existing=inst.shelters))
    st.markdown(legend(layer, lo, hi, pal), unsafe_allow_html=True)
    st.info(
        f"**{base_k['residents_70plus_uncovered']:,} residents aged 70+** live more than {radius} m (walking) from "
        f"a climate shelter; {base_k['sections_uncovered']} of 1,068 census sections are uncovered. "
        "Each census section is a demand point and a candidate site; the 77 existing shelters stay open."
    )

# --------------------------------------------------------------------------- 2. optimise
with tabs[1]:
    c1, c2, c3 = st.columns([2.2, 2.2, 0.8])
    model = c1.selectbox("Model", list(MODEL_LABEL), format_func=MODEL_LABEL.get)
    algo = c2.selectbox("Algorithm", [EXACT[model], *H.ALGORITHMS])
    seed = c3.number_input("Seed", 0, 999, 0)
    if st.button("Optimise", type="primary", icon="▶️"):
        st.session_state["opt"] = (model, algo, k, float(radius), W, int(seed))

    if "opt" in st.session_state and st.session_state["opt"][2:5] == (k, float(radius), W):
        m_, a_, *_ = st.session_state["opt"]
        with st.spinner(f"Solving with {a_}…"):
            res = solve(*st.session_state["opt"])
        sites = res["sites"]
        after = kpis(inst, sites, radius)
        c = st.columns(5)
        c[0].metric(f"70+ within {radius} m", fmt_pct(after["coverage_70plus"]),
                    f"{100 * (after['coverage_70plus'] - base_k['coverage_70plus']):+.1f} pp")
        c[1].metric("Average walk (70+)", f"{after['avg_walk_70plus_m']:.0f} m",
                    f"{after['avg_walk_70plus_m'] - base_k['avg_walk_70plus_m']:+.0f} m", delta_color="inverse")
        c[2].metric("95th pct walk (70+)", f"{after['p95_walk_70plus_m']:.0f} m",
                    f"{after['p95_walk_70plus_m'] - base_k['p95_walk_70plus_m']:+.0f} m", delta_color="inverse")
        c[3].metric("Worst-served section", f"{after['max_walk_m']:,.0f} m",
                    f"{after['max_walk_m'] - base_k['max_walk_m']:+,.0f} m", delta_color="inverse")
        c[4].metric("Weighted max distance", f"{after['weighted_max_distance']:.0f}",
                    f"{after['weighted_max_distance'] - base_k['weighted_max_distance']:+.0f}", delta_color="inverse")

        left, right = st.columns([3, 2])
        with left:
            st.pydeck_chart(section_map(poly, inst.sections, dmin_of(inst, sites), "walk", "{:.0f} m",
                                        WARM, 1500, existing=inst.shelters, new_idx=sites))
            st.markdown(legend("Walk to nearest shelter", "0 m", "≥1500 m"), unsafe_allow_html=True)
        with right:
            obj = Objective(inst, m_, radius=radius)
            line = f"**Objective** ({obj.unit}): `{res['value']:.4g}` · **runtime** {res['runtime']:.2f} s · `{res['status']}`"
            if res["lb"] is not None and res["value"]:
                line += f"  \n**Lower bound** `{res['lb']:.4g}` → certified gap **{100 * (res['value'] - res['lb']) / res['value']:.2f}%**"
            st.markdown(line)
            if res["info"]:
                st.caption(", ".join(f"{a}: {b}" for a, b in res["info"].items()))
            if len(res["history"]) > 1:
                st.plotly_chart(convergence(res["history"], obj.unit))
            table = inst.sections.iloc[sites][["code", "district", "neighbourhood", "pop70", "pct70", "heat"]].copy()
            table["walk before (m)"] = inst.base[sites].round(0)
            st.dataframe(table, hide_index=True, height=240,
                         column_config={"pct70": st.column_config.NumberColumn("share 70+", format="percent"),
                                        "heat": st.column_config.NumberColumn("heat", format="%.1f")})
            st.download_button("Download selected sites (CSV)", table.to_csv(index=False).encode(),
                               f"new_shelters_{m_}_k{k}.csv", "text/csv")
    else:
        st.caption("Choose a model and an algorithm, then press **Optimise**. "
                   "The exact OR-Tools models solve the full instance (1,068 sites) in seconds.")

# --------------------------------------------------------------------------- 3. benchmark
with tabs[2]:
    bench_path, marg_path = RESULTS / "benchmark.csv", RESULTS / "marginal.csv"
    if bench_path.exists():
        b = pd.read_csv(bench_path)
        st.markdown("#### Metaheuristics vs. exact references (full instance, 3 seeds each)")
        st.caption("Gap = (heuristic − reference) / reference. References: proven optima from CP-SAT for p-center "
                   "and maximal covering (R = 500 m); Lagrangian lower bound for p-median. Default weights (age + park).")
        bm = st.segmented_control("Model", ["center", "median", "coverage"], default="center",
                                  format_func=lambda m: MODEL_LABEL[m].split(" (")[0], key="bench_model") or "center"
        sub = b[b.model == bm]
        agg = sub.groupby(["k", "algorithm"]).agg(gap=("gap", "mean"), gmin=("gap", "min"), gmax=("gap", "max"),
                                                  t=("runtime_s", "mean")).reset_index()
        order = list(H.ALGORITHMS)
        fig = go.Figure()
        palette = ["#9aa5b1", "#52606d", ACCENT, "#e8a33d", "#3e7cb1"]
        for alg, col in zip(order, palette, strict=True):
            a = agg[agg.algorithm == alg]
            fig.add_bar(x=a["k"].astype(str), y=100 * a["gap"], name=alg, marker_color=col,
                        error_y=dict(type="data", symmetric=False, array=100 * (a["gmax"] - a["gap"]),
                                     arrayminus=100 * (a["gap"] - a["gmin"])))
        fig.update_layout(barmode="group", xaxis_title="new shelters k", yaxis_title="gap to reference (%)",
                          xaxis_type="category",
                          height=360, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig)
        piv = agg.pivot(index="algorithm", columns="k", values=["gap", "t"]).loc[order]
        piv.columns = [f"{'gap' if a == 'gap' else 'time (s)'} k={kk}" for a, kk in piv.columns]
        st.dataframe(piv.style.format({c: "{:.2%}" for c in piv.columns if c.startswith("gap")}
                                      | {c: "{:.2f}" for c in piv.columns if c.startswith("time")}))
    if marg_path.exists():
        mg = pd.read_csv(marg_path)
        st.markdown("#### How many shelters are worth it? (exact optima, k = 0…80)")
        c1, c2 = st.columns(2)
        f1 = go.Figure(go.Scatter(x=mg.k, y=mg.p_center, mode="lines+markers", line=dict(color=ACCENT)))
        f1.update_layout(title="Weighted p-center optimum", xaxis_title="new shelters", yaxis_title="weighted m",
                         height=320, margin=dict(l=10, r=10, t=40, b=10))
        f2 = go.Figure(go.Scatter(x=mg.k, y=100 * mg.coverage_500m, mode="lines+markers", line=dict(color=INK)))
        f2.update_layout(title="Residents 70+ within 500 m (MCLP optimum)", xaxis_title="new shelters",
                         yaxis_title="%", height=320, margin=dict(l=10, r=10, t=40, b=10))
        c1.plotly_chart(f1)
        c2.plotly_chart(f2)
        st.caption("The first ~10 shelters cut the worst weighted walk by 83% (they serve isolated, vulnerable "
                   "sections); after that, returns diminish while coverage keeps growing almost linearly.")
    with st.expander("Experiment log — SQL query on the `optimisation_run` table"):
        st.code(RUNS_SQL.strip(), language="sql")
        st.dataframe(run_summary(DATA_DIR), hide_index=True)
    sc = st.columns(2)
    if sc[0].button("How many shelters guarantee R metres for everyone? (set covering)"):
        with st.spinner("Solving set covering with CP-SAT…"):
            r = E.min_sites_for_radius(inst, radius)
        sc[0].success(f"Every census section within **{radius} m** walking needs at least **{int(r.value)}** new "
                      f"shelters (proven optimal in {r.runtime_s:.1f} s).")

# --------------------------------------------------------------------------- 4. multi-objective
with tabs[3]:
    st.markdown("Equity (protect the worst-off, **p-center**) and efficiency (shorten the average walk, "
                "**p-median**) pull in different directions. NSGA-II approximates the Pareto front for the chosen k.")
    gens = st.select_slider("NSGA-II generations", [40, 80, 150], value=80)
    if st.button("Compute Pareto front", type="primary"):
        st.session_state["pf"] = (k, W, gens)
    if "pf" in st.session_state and st.session_state["pf"][:2] == (k, W):
        with st.spinner("Running NSGA-II…"):
            pf = pareto(*st.session_state["pf"])
        F = pf["F"]
        left, right = st.columns([3, 2])
        with left:
            pick = st.slider("Inspect front solution (left = most equitable)", 0, len(F) - 1, len(F) // 2)
            fig = go.Figure()
            fig.add_scatter(x=F[:, 1], y=F[:, 0], mode="lines+markers", name="NSGA-II front",
                            line=dict(color=INK, shape="hv"), marker=dict(size=7))
            for (name, (cv, mv)), sym in zip(pf["anchors"].items(), ["star", "diamond"], strict=True):
                fig.add_scatter(x=[mv], y=[cv], mode="markers", name=name, marker=dict(size=14, symbol=sym))
            fig.add_scatter(x=[F[pick, 1]], y=[F[pick, 0]], mode="markers", name="selected",
                            marker=dict(size=16, color=ACCENT, line=dict(color="white", width=2)))
            fig.update_layout(xaxis_title="average walk of residents 70+ (m)  → efficiency",
                              yaxis_title="worst weighted walk  → equity", height=420,
                              margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.08))
            st.plotly_chart(fig)
            st.caption(f"{len(F)} non-dominated layouts · {pf['runtime']:.1f} s · hypervolume grew "
                       f"{100 * (pf['hv'][-1] / pf['hv'][0] - 1):.1f}% during the search.")
        with right:
            s = pf["sites"][pick]
            st.pydeck_chart(section_map(poly, inst.sections, dmin_of(inst, s), "walk", "{:.0f} m", WARM, 1500,
                                        existing=inst.shelters, new_idx=s), height=420)
            kk = kpis(inst, s, radius)
            st.markdown(f"Worst weighted walk **{F[pick, 0]:.0f}** · average walk **{F[pick, 1]:.0f} m** · "
                        f"{fmt_pct(kk['coverage_70plus'])} of 70+ within {radius} m")

# --------------------------------------------------------------------------- 5. simulation
with tabs[4]:
    st.markdown(
        "Deterministic models assume everyone within reach is served. On a heat-wave day **demand is random and "
        "shelters fill up**. A discrete-event simulation (Poisson arrivals per section, random walking tolerance "
        "and length of stay, finite capacity, retry at the next shelter) evaluates each layout; a "
        "**simulation-based optimisation** then relocates shelters using the simulated unmet demand."
    )
    with st.expander("Simulation assumptions (illustrative, editable)", expanded=False):
        d = S.HeatwaveParams()
        a1, a2, a3 = st.columns(3)
        p = replace(
            d,
            seek_rate=a1.slider("Share of 70+ seeking a shelter per day", 0.01, 0.08, d.seek_rate, 0.005),
            temp_sd=a1.slider("Std. dev. of daily max temperature (°C)", 0.0, 4.0, d.temp_sd, 0.5),
            capacity_existing=a2.slider("Capacity per existing shelter", 5, 60, d.capacity_existing),
            capacity_new=a2.slider("Capacity per new shelter", 5, 60, d.capacity_new),
            tolerance_median_m=a3.slider("Median walking tolerance (m)", 300, 1200, int(d.tolerance_median_m), 50),
            stay_mean_min=a3.slider("Mean length of stay (min)", 30, 300, int(d.stay_mean_min), 10),
        )
    iters = st.select_slider("Optimisation iterations", [5, 10, 15, 25], value=10)
    if st.button("Simulate & optimise", type="primary"):
        bar = st.progress(0.0, "Deterministic seeds (exact MCLP, p-median tabu)…")
        mc = E.max_covering(inst, k, p.tolerance_median_m).sites
        md = H.run("Tabu search", Objective(inst, "median"), k).sites
        bar.progress(0.05, "Simulation-guided search…")
        res = S.optimize(inst, k, p, seeds={"Maximal covering (exact)": mc, "p-median (tabu)": md},
                         iterations=iters, progress=lambda f: bar.progress(0.05 + 0.9 * f, "Simulation-guided search…"))
        val = S.validation_scenarios(inst, p, 30)
        st.session_state["sbo"] = (k, p, res, S.evaluate(inst, [], val, p))
        bar.empty()
    if "sbo" in st.session_state and st.session_state["sbo"][:2] == (k, p):
        _, _, res, base_sim = st.session_state["sbo"]
        sbo = res.validated["Simulation-based optimisation"]
        rows = [("Today (no new shelters)", base_sim)] + list(res.validated.items())
        tbl = pd.DataFrame([{
            "layout": n_, "unmet demand": s_.unmet_mean, "± 95% CI": s_.unmet_ci95,
            "too far": s_.too_far_share, "turned away (full)": s_.full_share, "CVaR 90% (worst days)": s_.unmet_cvar90,
        } for n_, s_ in rows])
        st.markdown(f"**{base_sim.demand_mean:,.0f} people/day** seek a shelter on average · "
                    f"30 independent validation scenarios · {res.evaluations} simulated layouts in {res.runtime_s:.0f} s")
        st.dataframe(tbl.style.format({c: "{:.1%}" for c in tbl.columns if c != "layout"}), hide_index=True)
        msgs = []
        for name in res.validated:
            if name != "Simulation-based optimisation":
                m, ci = S.paired_difference(sbo, res.validated[name])
                sig = "significant" if abs(m) > ci else "not significant"
                msgs.append(f"vs **{name}**: {100 * m:+.2f} pp ± {100 * ci:.2f} ({sig}, paired CRN)")
        st.markdown("Simulation-based optimisation " + " · ".join(msgs))
        left, right = st.columns([3, 2])
        with left:
            served = np.mean([r.served_by_facility for r in sbo.results], axis=0)[len(inst.shelters):]
            unmet = np.mean([r.unmet_by_section for r in sbo.results], axis=0)
            st.pydeck_chart(section_map(poly, inst.sections, unmet, "unmet people/day", "{:.1f}", WARM,
                                        np.quantile(unmet, 0.98), existing=inst.shelters, new_idx=res.sites,
                                        new_sizes=60 + 12 * np.sqrt(served)))
            st.markdown(legend("Unmet people/day (SBO layout)", "0", "high"), unsafe_allow_html=True)
        with right:
            fig = go.Figure()
            for name_, s_ in rows:
                fig.add_bar(y=[name_.split(" (")[0]], x=[100 * s_.too_far_share], orientation="h",
                            marker_color="#e8a33d", name="too far", showlegend=name_ == rows[0][0])
                fig.add_bar(y=[name_.split(" (")[0]], x=[100 * s_.full_share], orientation="h",
                            marker_color=ACCENT, name="turned away", showlegend=name_ == rows[0][0])
            fig.update_layout(barmode="stack", xaxis_title="% of demand unmet", height=260,
                              margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.15))
            st.plotly_chart(fig)
            st.plotly_chart(convergence([100 * h for h in res.history], "unmet % (training scenarios)",
                                        "Simulation-guided search"))

# --------------------------------------------------------------------------- 6. methodology
with tabs[5]:
    st.markdown(r"""
### Data pipeline
* **1,068 census sections** (demand points and candidate sites), population and population aged 70+ (Padró),
  distance to the nearest park, summer **thermal-discomfort grid** (area-weighted per section), **77 existing climate shelters**.
* **Walking distances** on the municipal street graph (9,483 nodes, 14,831 links): every centroid and shelter is snapped to the
  largest connected component and distances are computed with Dijkstra (median detour factor 1.19 vs straight line).
* Tabular data live in **SQLite** and are loaded with SQL joins (portable to PostgreSQL / Azure SQL); the 1068×1068 matrix is a compressed NumPy array.

### Models  (existing shelters $E$ always open, choose a set $S$ of $k$ new sites)
Let $d_i(S)=\min\{\min_{e\in E} d_{ie},\ \min_{j\in S} d_{ij}\}$.

| Model | Objective | Solved with |
|---|---|---|
| Weighted **p-center** | $\min_S \max_i v_i\, d_i(S)$ | exact: binary search over the $\le n^2$ candidate radii, each step a **set-covering feasibility** model in CP-SAT |
| **p-median** | $\min_S \sum_i w_i\, d_i(S)$ | metaheuristics + **Lagrangian relaxation** (subgradient) giving a certified gap |
| **Maximal covering** | $\max_S \sum_i w_i\,[d_i(S)\le R]$ | exact MIP in CP-SAT |
| **Set covering** | $\min \lvert S\rvert$ s.t. $d_i(S)\le R\ \forall i$ | exact MIP in CP-SAT |

$v_i$ is the vulnerability index (age 70+, distance to park, heat), $w_i$ the share of residents 70+.

### Metaheuristics
Greedy-Add, **Teitz–Bart interchange** local search (fast evaluation with nearest / second-nearest bookkeeping, fully vectorised),
**tabu search** (short-term memory on entering/leaving sites, aspiration), **VNS** (Hansen & Mladenović) and an elitist
**genetic algorithm** (set-preserving crossover, spatial mutation, memetic polish). A $10^{-3}$ lexicographic tie-break lets local
search cross the plateaus of min-max objectives.

### Multi-objective
From-scratch **NSGA-II** (fast non-dominated sorting, crowding distance, $(\mu+\lambda)$ survival) seeded with the single-objective
optima; front quality tracked by 2-D **hypervolume**.

### Simulation-based optimisation
A **discrete-event simulation** of a heat-wave day: daily temperature $T\sim\mathcal N$, Poisson arrivals per section with rate
$\propto$ residents 70+ × heat exposure × $f(T)$, triangular arrival times peaking at 16:00, log-normal walking tolerance, exponential
stay, finite capacity (loss system) with one retry. Layouts are compared with **common random numbers**; the search proposes swaps
from simulation output (close the least-used new shelter, open where simulated unmet demand concentrates) and ends with a
**selection-of-the-best** on 30 independent scenarios, reporting paired confidence intervals and CVaR.

### Origin and differences from the original study
Started as a team project at the **Barcelona Modelling Week (July 2025)**: weighted p-center solved with Gurobi on straight-line
centroid distances. This version re-engineers it end-to-end: network walking distances, open-source solvers, an exact algorithm
that scales to the full instance, metaheuristics with a benchmark against proven optima, multi-objective analysis, and stochastic
simulation of capacity.

**Limitations.** Census-section centroids as demand points and candidate sites; simulation parameters are illustrative assumptions
(exposed for what-if analysis), not calibrated against shelter attendance data.
""")
