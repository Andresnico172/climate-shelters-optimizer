"""Data access layer.

Tabular data is read from a SQLite database with plain SQL (the same queries run unchanged
on PostgreSQL / Azure SQL); the dense distance matrix is stored as a compressed NumPy array.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

SECTIONS_SQL = """
SELECT  s.code, s.district, s.neighbourhood,
        p.pop, p.pop70,
        CAST(p.pop70 AS REAL) / MAX(p.pop, 1) AS pct70,
        e.dist_park, e.heat,
        s.area_km2, s.x, s.y, s.lon, s.lat, s.access_m
FROM census_section s
JOIN population  p USING (code)
JOIN environment e USING (code)
ORDER BY s.code
"""

SHELTERS_SQL = "SELECT shelter_id, section_code, x, y, lon, lat FROM climate_shelter ORDER BY shelter_id"

RUNS_SQL = """
SELECT model, algorithm, k, COUNT(*) AS runs,
       AVG(objective) AS mean_objective, MIN(objective) AS best_objective,
       AVG(runtime_s) AS mean_runtime_s
FROM optimisation_run
GROUP BY model, algorithm, k
ORDER BY model, k, mean_objective
"""


def _connect(data_dir: Path) -> sqlite3.Connection:
    return sqlite3.connect(data_dir / "bcn.sqlite")


def load_sections(data_dir: Path) -> pd.DataFrame:
    with _connect(data_dir) as con:
        return pd.read_sql_query(SECTIONS_SQL, con)


def load_shelters(data_dir: Path) -> pd.DataFrame:
    with _connect(data_dir) as con:
        return pd.read_sql_query(SHELTERS_SQL, con)


def load_distances(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(data_dir / "distances.npz") as z:
        return z["candidates"], z["existing"], z["codes"]


def load_geojson(data_dir: Path) -> dict:
    return json.loads((data_dir / "sections.geojson").read_text())


def log_run(data_dir: Path, model: str, algorithm: str, k: int, objective: float,
            runtime_s: float, selected_codes: list[int]) -> None:
    with _connect(data_dir) as con:
        con.execute(
            "INSERT INTO optimisation_run (model, algorithm, k, objective, runtime_s, selected_codes) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (model, algorithm, k, objective, runtime_s, json.dumps([int(c) for c in selected_codes])),
        )


def run_summary(data_dir: Path) -> pd.DataFrame:
    with _connect(data_dir) as con:
        return pd.read_sql_query(RUNS_SQL, con)
