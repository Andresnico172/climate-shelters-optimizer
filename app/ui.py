"""Map and chart helpers for the Streamlit app."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk

# Sequential ramps (low -> high), RGB.
WARM = [(255, 247, 214), (254, 217, 142), (254, 178, 76), (253, 141, 60), (240, 59, 32), (189, 0, 38)]
COOL = [(237, 248, 251), (191, 211, 230), (158, 188, 218), (140, 150, 198), (136, 86, 167), (110, 1, 107)]
EXISTING = [25, 85, 165]
NEW = [20, 20, 20]
ACCENT = "#c8202d"
INK = "#1f2933"


def polygons(geojson: dict, codes: np.ndarray) -> pd.DataFrame:
    """One row per polygon part, keyed by the section index used by the optimiser."""
    pos = {int(c): i for i, c in enumerate(codes)}
    rows = []
    for feat in geojson["features"]:
        geom = feat["geometry"]
        parts = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        for part in parts:
            rows.append({"idx": pos[int(feat["properties"]["code"])], "polygon": part})
    return pd.DataFrame(rows)


def ramp(values: np.ndarray, palette=WARM, vmin=None, vmax=None) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    lo = np.nanmin(v) if vmin is None else vmin
    hi = np.nanmax(v) if vmax is None else vmax
    t = np.clip((v - lo) / (hi - lo if hi > lo else 1.0), 0, 1) * (len(palette) - 1)
    i = np.minimum(t.astype(int), len(palette) - 2)
    f = (t - i)[:, None]
    pal = np.array(palette, dtype=float)
    return (pal[i] * (1 - f) + pal[i + 1] * f).astype(int)


def section_map(poly: pd.DataFrame, sections: pd.DataFrame, values: np.ndarray, value_label: str,
                fmt: str = "{:.0f}", palette=WARM, vmax=None, existing: pd.DataFrame | None = None,
                new_idx=None, new_sizes=None) -> pdk.Deck:
    colors = ramp(values, palette, vmax=vmax)
    names = (sections["neighbourhood"] + " · " + sections["district"]).to_numpy()
    texts = np.array([fmt.format(x) for x in values])
    df = poly.copy()
    idx = df["idx"].to_numpy()
    df["rgba"] = [[int(r), int(g), int(b), 185] for r, g, b in colors[idx]]  # plain ints: JSON-serialisable
    df["name"], df["code"], df["value"] = names[idx], sections["code"].to_numpy()[idx], texts[idx]
    df["label"] = value_label

    layers = [pdk.Layer("PolygonLayer", df, get_polygon="polygon", get_fill_color="rgba",
                        get_line_color=[255, 255, 255, 90], line_width_min_pixels=0.3,
                        pickable=True, auto_highlight=True)]
    if existing is not None:
        ex = existing.assign(name="Existing climate shelter", code=existing["section_code"].fillna(-1).astype(int),
                             value="", label="")
        layers.append(pdk.Layer("ScatterplotLayer", ex, get_position=["lon", "lat"], get_fill_color=EXISTING,
                                get_line_color=[255, 255, 255], line_width_min_pixels=1, stroked=True,
                                radius_min_pixels=4, get_radius=60, pickable=True))
    if new_idx is not None and len(new_idx):
        new_idx = np.asarray(new_idx)
        nw = sections.iloc[new_idx][["lon", "lat", "code", "neighbourhood", "district"]].copy()
        nw["name"] = "NEW shelter · " + nw["neighbourhood"]
        nw["value"], nw["label"] = "", ""
        nw["r"] = 70 if new_sizes is None else np.asarray(new_sizes, dtype=float)
        layers.append(pdk.Layer("ScatterplotLayer", nw, get_position=["lon", "lat"], get_fill_color=NEW,
                                get_line_color=[255, 220, 0], line_width_min_pixels=2, stroked=True,
                                radius_min_pixels=5, get_radius="r", pickable=True))
    return pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(latitude=41.400, longitude=2.160, zoom=11.2, pitch=0),
        tooltip={"html": "<b>{name}</b><br/>section {code}<br/>{label} {value}",
                 "style": {"fontSize": "12px"}},
        map_style=None,
    )


def legend(label: str, lo: str, hi: str, palette=WARM) -> str:
    stops = ", ".join(f"rgb{c}" for c in palette)
    return (f"<div style='display:flex;align-items:center;gap:8px;font-size:0.8rem;margin:-6px 0 6px'>"
            f"<span>{label}</span><span>{lo}</span>"
            f"<span style='flex:0 0 160px;height:10px;border-radius:2px;background:linear-gradient(90deg,{stops})'></span>"
            f"<span>{hi}</span>"
            f"<span style='margin-left:12px;color:rgb{tuple(EXISTING)}'>● existing shelter</span>"
            f"<span>■ new shelter</span></div>")


def convergence(history: list[float], ylabel: str, title: str = "Convergence") -> go.Figure:
    fig = go.Figure(go.Scatter(y=history, mode="lines", line=dict(color=ACCENT, width=2.5)))
    fig.update_layout(title=title, xaxis_title="iteration", yaxis_title=ylabel, height=300,
                      margin=dict(l=10, r=10, t=40, b=10))
    return fig
