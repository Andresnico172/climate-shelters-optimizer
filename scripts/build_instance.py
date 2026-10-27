"""Build the optimisation instance from Barcelona open data.

Pipeline (run once, offline; the app only reads the processed outputs):

1. Census sections (polygons) + population, population 70+, distance to nearest park.
2. Heat exposure: area-weighted mean of the summer thermal-discomfort grid (1 = cool, 6 = hot).
3. Street graph -> walking distances (Dijkstra) between every section centroid and
   every candidate / existing shelter, snapping points to the largest connected component.
4. Persist to CSV / NPZ / GeoJSON and to a SQLite database used by the app.

Usage:  python scripts/build_instance.py
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree
from shapely import wkt

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
CRS_UTM = "EPSG:25831"  # ETRS89 / UTM 31N, metres


def load_sections() -> gpd.GeoDataFrame:
    df = pd.read_csv(RAW / "census_sections.csv", dtype=str)
    df["code"] = df["codi_districte"].astype(int) * 1000 + df["codi_seccio_censal"].astype(int)
    gdf = gpd.GeoDataFrame(
        df[["code", "nom_districte", "nom_barri"]].rename(
            columns={"nom_districte": "district", "nom_barri": "neighbourhood"}
        ),
        geometry=df["geometria_etrs89"].apply(wkt.loads),
        crs=CRS_UTM,
    )

    pop = pd.read_csv(RAW / "population_total.csv").rename(
        columns={"Seccio_Censal": "code", "poblacion_total": "pop"}
    )
    pop70 = pd.read_csv(RAW / "population_70plus.csv").rename(
        columns={"Seccio_Censal": "code", "poblacion_total": "pop70"}
    )
    park = pd.read_csv(RAW / "nearest_park.csv").rename(
        columns={"codigo_seccion": "code", "distancia_metros": "dist_park"}
    )[["code", "dist_park"]]

    gdf = gdf.merge(pop, on="code").merge(pop70, on="code").merge(park, on="code")
    gdf["pct70"] = gdf["pop70"] / gdf["pop"].clip(lower=1)
    gdf["area_km2"] = gdf.area / 1e6
    gdf = gdf.sort_values("code").reset_index(drop=True)
    assert len(gdf) == 1068, f"expected 1068 census sections, got {len(gdf)}"
    return gdf


def add_heat_exposure(sections: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    comfort = gpd.read_file(RAW / "thermal_comfort.gpkg").to_crs(CRS_UTM)
    inter = gpd.overlay(sections[["code", "geometry"]], comfort[["gridcode", "geometry"]], how="intersection")
    inter["a"] = inter.area
    heat = inter.groupby("code").apply(lambda g: np.average(g["gridcode"], weights=g["a"]), include_groups=False)
    sections["heat"] = sections["code"].map(heat).fillna(heat.median())
    return sections


def load_shelters(sections: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    df = pd.read_csv(RAW / "climate_shelters.csv")
    gdf = gpd.GeoDataFrame(
        {"shelter_id": np.arange(len(df))},
        geometry=gpd.points_from_xy(df["longitud"], df["latitud"]),  # columns are UTM despite their names
        crs=CRS_UTM,
    )
    gdf = gpd.sjoin(gdf, sections[["code", "geometry"]], how="left", predicate="within")
    gdf = gdf.rename(columns={"code": "section_code"}).drop(columns="index_right")
    return gdf


def street_graph():
    nodes = pd.read_csv(RAW / "street_nodes.csv", sep=";", encoding="latin1")
    edges = pd.read_csv(RAW / "street_edges.csv", sep=";", encoding="latin1")
    idx = {c: i for i, c in enumerate(nodes["C_Nus"])}
    edges = edges[edges["C_Nus_I"].isin(idx) & edges["C_Nus_F"].isin(idx)]
    u = edges["C_Nus_I"].map(idx).to_numpy()
    v = edges["C_Nus_F"].map(idx).to_numpy()
    w = edges["LONGITUD"].astype(float).to_numpy()
    n = len(nodes)
    graph = coo_matrix((np.r_[w, w], (np.r_[u, v], np.r_[v, u])), shape=(n, n)).tocsr()

    _, labels = connected_components(graph, directed=False)
    main = labels == np.bincount(labels).argmax()
    xy = nodes[["Coord_X", "Coord_Y"]].to_numpy()
    print(f"street graph: {n} nodes, {len(edges)} edges, largest component {main.sum()} nodes")
    return graph, xy, main


def snap(points_xy: np.ndarray, node_xy: np.ndarray, mask: np.ndarray):
    """Nearest node of the main component and the straight-line access distance to it."""
    candidates = np.flatnonzero(mask)
    dist, k = cKDTree(node_xy[candidates]).query(points_xy)
    return candidates[k], dist


def walking_distances(sections, shelters):
    graph, node_xy, main = street_graph()
    sec_xy = np.c_[sections.geometry.centroid.x, sections.geometry.centroid.y]
    shl_xy = np.c_[shelters.geometry.x, shelters.geometry.y]
    sec_node, sec_off = snap(sec_xy, node_xy, main)
    shl_node, shl_off = snap(shl_xy, node_xy, main)

    net = dijkstra(graph, directed=False, indices=sec_node)  # (n_sections, n_nodes)
    d_cand = sec_off[:, None] + net[:, sec_node] + sec_off[None, :]
    np.fill_diagonal(d_cand, 0.0)
    d_exist = sec_off[:, None] + net[:, shl_node] + shl_off[None, :]
    assert np.isfinite(d_cand).all() and np.isfinite(d_exist).all()

    detour = d_cand[np.triu_indices(len(sec_xy), 1)] / np.maximum(
        np.linalg.norm(sec_xy[:, None] - sec_xy[None], axis=2)[np.triu_indices(len(sec_xy), 1)], 1
    )
    print(f"median network/euclidean detour factor: {np.median(detour):.2f}")
    return d_cand.astype(np.float32), d_exist.astype(np.float32), sec_xy, sec_off


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sections = add_heat_exposure(load_sections())
    shelters = load_shelters(sections)
    d_cand, d_exist, sec_xy, sec_off = walking_distances(sections, shelters)

    wgs = sections.to_crs("EPSG:4326")
    cent = sections.geometry.centroid.to_crs("EPSG:4326")
    sections["x"], sections["y"] = sec_xy[:, 0], sec_xy[:, 1]
    sections["lon"], sections["lat"] = cent.x.to_numpy(), cent.y.to_numpy()
    sections["access_m"] = sec_off
    shl_wgs = shelters.to_crs("EPSG:4326")
    shelters["x"], shelters["y"] = shelters.geometry.x, shelters.geometry.y
    shelters["lon"], shelters["lat"] = shl_wgs.geometry.x, shl_wgs.geometry.y

    sec_cols = ["code", "district", "neighbourhood", "pop", "pop70", "pct70", "dist_park", "heat",
                "area_km2", "x", "y", "lon", "lat", "access_m"]
    shl_cols = ["shelter_id", "section_code", "x", "y", "lon", "lat"]
    sec_df = pd.DataFrame(sections[sec_cols])
    shl_df = pd.DataFrame(shelters[shl_cols])
    sec_df.to_csv(OUT / "sections.csv", index=False)
    shl_df.to_csv(OUT / "shelters.csv", index=False)
    np.savez_compressed(OUT / "distances.npz", candidates=d_cand, existing=d_exist, codes=sec_df["code"].to_numpy())

    # Simplified polygons for the web map (~2 m tolerance keeps the file small).
    geo = wgs[["code", "geometry"]].copy()
    geo["geometry"] = geo.geometry.simplify(2e-5, preserve_topology=True)
    (OUT / "sections.geojson").write_text(json.dumps(json.loads(geo.to_json()), separators=(",", ":")))

    with sqlite3.connect(OUT / "bcn.sqlite") as con:
        sec_df[["code", "district", "neighbourhood", "x", "y", "lon", "lat", "area_km2", "access_m"]].to_sql(
            "census_section", con, if_exists="replace", index=False)
        sec_df[["code", "pop", "pop70"]].to_sql("population", con, if_exists="replace", index=False)
        sec_df[["code", "dist_park", "heat"]].to_sql("environment", con, if_exists="replace", index=False)
        shl_df.to_sql("climate_shelter", con, if_exists="replace", index=False)
        con.executescript("""
            CREATE TABLE IF NOT EXISTS optimisation_run (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                model TEXT, algorithm TEXT, k INTEGER, objective REAL,
                runtime_s REAL, selected_codes TEXT
            );
        """)

    print(f"sections={len(sec_df)} shelters={len(shl_df)} "
          f"pop={sec_df['pop'].sum():,} pop70={sec_df['pop70'].sum():,}")
    print(f"baseline: max walk to nearest shelter = {d_exist.min(axis=1).max():.0f} m, "
          f"median = {np.median(d_exist.min(axis=1)):.0f} m")


if __name__ == "__main__":
    main()
