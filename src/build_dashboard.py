"""
Builds the interactive UAE experience map -- capability #1 of the brief, "Interactive UAE
experience map" (switchable Experience / Priority / Trend / Confidence layers, national view
+ zone drill-down) -- as one self-contained HTML file. No copilot/chat here on purpose (that's
capability #7, a separate piece); this is the map + evidence panel it sits inside of.

Design: plain SVG + vanilla JS, no external libraries. Every hexagon's shape is precomputed in
Python (real H3 geometry, real equirectangular-ish projection -- see `_project`) and written
straight into the page as an SVG path; the browser only has to recolor paths on layer/zone/
quarter/emirate changes, not compute any geometry. That keeps the file self-contained (works
offline, no CDN dependency).

Run: `python -m src.build_dashboard` from the repo root. Reads `zone_priority.parquet` (the
output of the scores -> trend -> anomaly -> priority pipeline, now carrying the `emirate` field
added by `scripts/build_emirate_field.py`) and `population_zones_uae.parquet` (for the
population-represented KPI, which needs the *full* population universe, not just measured
zones, as its denominator). Writes `data/processed/uae_dashboard.html`.
"""
import json
import math
from pathlib import Path

import geopandas as gpd
import h3
import pandas as pd


# Single source of truth for the pipeline's H3 resolution, same file every other notebook reads
# (notebooks/04_h3_resolution_choice.ipynb writes it) -- reading it here rather than hardcoding
# a second copy is what keeps this file's hex geometry from silently drifting out of sync with
# the `h3_cell` IDs in zone_priority.parquet if the chosen resolution ever changes again.
MAP_H3_RESOLUTION = json.loads(Path("data/processed/h3_resolution.json").read_text())["chosen_resolution"]
QUARTER_ORDER = ["2024Q3", "2024Q4", "2025Q1", "2025Q2", "2025Q3", "2025Q4", "2026Q1", "2026Q2"]
MAP_WIDTH = 900

# Purely cosmetic map-text pins (city names for visual orientation) -- unrelated to the real
# `emirate` field used for filtering/attribution. Kept separate on purpose: a zone's emirate
# comes from the authoritative boundary join (scripts/build_emirate_field.py), never from
# "nearest of these 9 dots," which is what this dashboard used before that field existed.
PLACE_LABELS = {
    "ABU DHABI": (24.4667, 54.3667), "DUBAI": (25.2048, 55.2708), "SHARJAH": (25.3573, 55.4033),
    "AJMAN": (25.4111, 55.4354), "RAK": (25.7895, 55.9432), "FUJAIRAH": (25.1288, 56.3265),
    "AL AIN": (24.2075, 55.7447), "RUWAIS": (24.1102, 52.7306), "MADINAT ZAYED": (23.6588, 53.7081),
    "UMM AL QUWAIN": (25.5647, 55.5552),
}

EMIRATES = ["Abu Dhabi", "Dubai", "Sharjah", "Ajman", "Umm Al Quwain", "Ras Al Khaimah", "Fujairah"]


def build_hex_grid(boundary_path: Path, extra_cells=None):
    """Every H3 hexagon (at MAP_H3_RESOLUTION) covering the UAE, as an SVG path string in the
    shared pixel projection. Returns (list of {id, path}, projection function, viewBox width/height).

    `extra_cells` (optional): H3 cell IDs to guarantee a polygon for even if `polygon_to_cells`
    didn't produce them. `polygon_to_cells` includes a cell only if its own CENTROID falls
    inside the boundary polygon -- a different test than `h3.latlng_to_cell(tile_lat, tile_lon)`,
    which is what actually assigns a real measurement to a cell in `05_zone_aggregation.ipynb`.
    At resolution 6's larger (~36 km^2) hexagons, a real classified zone can hold a genuine,
    boundary-clipped measurement while its centroid sits just outside the exact coastline/border
    polygon -- 49 of 671 classified 2026Q2 zones, including the #1 nationally-ranked priority
    zone, were missing a polygon for exactly this reason before this parameter existed. Pass every
    zone ID that ever appears in `zone_priority.parquet` here so no classified zone is ever
    unselectable on the map."""
    uae = gpd.read_file(boundary_path).to_crs("EPSG:4326")
    shape = uae.geometry.iloc[0]
    min_lon, min_lat, max_lon, max_lat = shape.bounds
    lat_mid_rad = math.radians((min_lat + max_lat) / 2)

    lon_span_km = (max_lon - min_lon) * 111.32 * math.cos(lat_mid_rad)
    lat_span_km = (max_lat - min_lat) * 111.32
    scale = MAP_WIDTH / lon_span_km
    height = lat_span_km * scale

    def project(lon, lat):
        x = (lon - min_lon) * 111.32 * math.cos(lat_mid_rad) * scale
        y = (max_lat - lat) * 111.32 * scale
        return round(x, 1), round(y, 1)

    polys = []
    for part in shape.geoms:
        ring = [(lat, lon) for lon, lat in part.exterior.coords]
        polys.append(h3.LatLngPoly(ring))
    all_cells = set(h3.polygon_to_cells(h3.LatLngMultiPoly(*polys), MAP_H3_RESOLUTION))
    if extra_cells:
        all_cells |= set(extra_cells)

    hexagons = []
    for cell in all_cells:
        corners = h3.cell_to_boundary(cell)
        points = [project(lon, lat) for lat, lon in corners]
        d = "M" + " L".join(f"{x},{y}" for x, y in points) + " Z"
        hexagons.append({"id": cell, "d": d})

    labels = [
        {"name": name, **dict(zip(("x", "y"), project(lon, lat)))}
        for name, (lat, lon) in PLACE_LABELS.items()
    ]

    return hexagons, project, round(MAP_WIDTH, 1), round(height, 1), labels


def _num(v):
    return None if pd.isna(v) else round(float(v), 1)


def build_zone_data(priority_path: Path, population_path: Path):
    """Per-classified-zone data for *every* quarter (not just the latest -- the quarter
    selector needs all 8), plus per-quarter KPIs and the 8-quarter Experience Index history
    (for the trend sparkline, which is quarter-independent so computed once)."""
    df = pd.read_parquet(priority_path)
    classified_all = df.loc[~df["insufficient_evidence"]].copy()

    cell_centers = {c: h3.cell_to_latlng(c) for c in classified_all["h3_cell"].unique()}

    history = (
        df[df["h3_cell"].isin(classified_all["h3_cell"])]
        .sort_values("quarter")
        .groupby("h3_cell")["experience_index"]
        .apply(lambda s: [None if pd.isna(v) else round(float(v), 1) for v in s])
        .to_dict()
    )

    def zone_record(row) -> dict:
        lat, lon = cell_centers[row["h3_cell"]]
        return {
            "emirate": row["emirate"],
            "peer_group": row["peer_group"],
            "experience_index": _num(row["experience_index"]),
            "confidence_score": _num(row["confidence_score"]),
            "priority_score": _num(row["priority_score"]),
            "priority_zone": bool(row["priority_zone"]),
            "download_mbps": _num(row["download_mbps"]),
            "upload_mbps": _num(row["upload_mbps"]),
            "latency_ms": _num(row["latency_effective_ms"]),
            "tests": int(row["tests"]),
            "devices": int(row["devices"]),
            "quarters_observed": int(row["quarters_observed"]),
            "population": int(round(row["population"])),
            "trend_pts_per_qtr": _num(row["trend_pts_per_qtr"]),
            "deteriorating": bool(row["deteriorating"]),
            "peer_gap": _num(row["peer_gap"]),
            "peer_gap_pct": _num(row["peer_gap_pct"]),
            "peer_median": _num(row["peer_group_median_experience"]),
            "peer_gap_ml_anomaly": bool(row["peer_gap_ml_anomaly"]),
            "temporal_anomaly_ml_flag": bool(row["temporal_anomaly_ml_flag"]),
            "bands": {
                "peer_gap": str(row["peer_gap_band"]),
                "ml_anomaly": str(row["ml_anomaly_band"]),
                "deterioration": str(row["deterioration_band"]),
                "population": str(row["population_band"]),
            },
            "lat": round(lat, 4),
            "lon": round(lon, 4),
            "sparkline": history.get(row["h3_cell"], []),
        }

    zones_by_quarter, top5_by_quarter, kpis_by_quarter = {}, {}, {}
    population_all = pd.read_parquet(population_path)["population"].sum()

    for i, quarter in enumerate(QUARTER_ORDER):
        classified = classified_all[classified_all["quarter"] == quarter]
        if classified.empty:
            continue
        records = {row["h3_cell"]: zone_record(row) for _, row in classified.iterrows()}
        zones_by_quarter[quarter] = records
        top5_by_quarter[quarter] = (
            classified.sort_values("priority_score", ascending=False).head(5)["h3_cell"].tolist()
        )
        prev_classified = classified_all[classified_all["quarter"] == QUARTER_ORDER[i - 1]] if i > 0 else None
        kpis_by_quarter[quarter] = {
            "scored_zones": len(classified),
            "population_pct": round(float(100 * classified["population"].sum() / population_all), 1),
            "priority_zones": int(classified["priority_zone"].sum()),
            "median_download": round(float(classified["download_mbps"].median()), 0),
            "experience_qoq": (
                round(float(classified["experience_index"].median() - prev_classified["experience_index"].median()), 1)
                if prev_classified is not None and not prev_classified.empty else None
            ),
        }

    # Population denominators for the "population represented" KPI under an emirate filter --
    # each emirate's own full population universe, not just its measured/classified zones.
    # Reads the authoritative emirate_zones_uae.parquet (covers every populated cell, not
    # just Ookla-measured ones -- see scripts/build_emirate_field.py) rather than deriving a
    # partial mapping from zone_priority.parquet, which would silently under-count any
    # populated cell Ookla never sampled at all.
    pop_df = pd.read_parquet(population_path)
    emirate_zones = pd.read_parquet("data/processed/emirate_zones_uae.parquet")
    pop_by_emirate = pop_df.merge(emirate_zones, on="h3_cell", how="left")
    population_by_emirate = {"All UAE": int(population_all)}
    for emirate in EMIRATES:
        population_by_emirate[emirate] = int(pop_by_emirate.loc[pop_by_emirate["emirate"] == emirate, "population"].sum())

    trend_abs = classified_all["trend_pts_per_qtr"].dropna().abs()
    domains = {
        "experience": [0, 100],
        "confidence": [0, 100],
        "priority": [0, round(float(classified_all["priority_score"].max()), 1)],
        "trend": [round(float(-trend_abs.quantile(0.95)), 1), round(float(trend_abs.quantile(0.95)), 1)],
    }

    return zones_by_quarter, top5_by_quarter, kpis_by_quarter, domains, population_by_emirate


def render_html(hexagons, width, height, labels, zones_by_quarter, top5_by_quarter,
                 kpis_by_quarter, domains, population_by_emirate) -> str:
    """Assembles the full self-contained HTML page: CSS for layout/theme, embedded JSON for
    the hex grid + per-quarter zone data, and vanilla JS for layer/quarter/emirate switching
    and zone drill-down. No chat/copilot UI -- that's a separate capability, not part of this
    map."""
    hex_paths = "\n".join(
        f'<path id="hex-{h["id"]}" class="hex" d="{h["d"]}" onclick="selectZone(\'{h["id"]}\')"></path>'
        for h in hexagons
    )
    label_svg = "\n".join(
        f'<text class="place-label" x="{l["x"]}" y="{l["y"]}">{l["name"]}</text>' for l in labels
    )
    quarter_options = "\n".join(
        f'<option value="{q}"{" selected" if q == QUARTER_ORDER[-1] else ""}>{q}</option>' for q in QUARTER_ORDER
    )
    emirate_options = "\n".join(f'<option value="{e}">{e}</option>' for e in EMIRATES)
    latest_scored = kpis_by_quarter[QUARTER_ORDER[-1]]["scored_zones"]

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>UAE Mobile Network Experience Intelligence</title>
<style>
{_CSS}
</style>
</head>
<body>

<header>
  <div class="title-block">
    <h1>UAE Mobile Network Experience Intelligence</h1>
    <div class="subtitle">Outside-in public mobile-experience intelligence &middot; Mobile only &middot; Not e&amp; network data</div>
  </div>
  <div class="header-controls">
    <div class="select-group">
      <label for="quarter-select">Quarter</label>
      <select id="quarter-select" onchange="setQuarter(this.value)">
      {quarter_options}
      </select>
    </div>
    <div class="select-group">
      <label for="emirate-select">Emirate</label>
      <select id="emirate-select" onchange="setEmirate(this.value)">
        <option value="All UAE">All UAE</option>
        {emirate_options}
      </select>
    </div>
    <div class="layer-buttons" id="layer-buttons">
      <button class="layer-btn active" data-layer="experience" onclick="setLayer('experience')">Experience</button>
      <button class="layer-btn" data-layer="priority" onclick="setLayer('priority')">Priority</button>
      <button class="layer-btn" data-layer="trend" onclick="setLayer('trend')">Trend</button>
      <button class="layer-btn" data-layer="confidence" onclick="setLayer('confidence')">Confidence</button>
    </div>
  </div>
</header>

<div class="disclaimer">
  REAL PUBLIC OOKLA / WORLDPOP / OSM DATA &middot; {latest_scored} OF ~15,700 ZONES MEET THE EVIDENCE THRESHOLD (LATEST QUARTER) &middot;
  NOT E&amp; NETWORK DATA &middot; PROTOTYPE, PRE-VALIDATION
</div>

<div class="kpi-strip" id="kpi-strip"></div>

<main>
  <div class="map-pane">
    <svg id="map-svg" viewBox="0 0 {width} {height}" preserveAspectRatio="xMidYMid meet">
      <g id="hex-layer">
      {hex_paths}
      </g>
      <g id="label-layer">
      {label_svg}
      </g>
    </svg>
    <div class="legend" id="legend"></div>
  </div>

  <aside class="side-panel">
    <div class="panel-block">
      <div class="panel-title" id="priority-list-title">This quarter &middot; top priority zones</div>
      <ol class="priority-list" id="priority-list"></ol>
    </div>
    <div class="panel-block zone-detail" id="zone-detail"></div>
  </aside>
</main>

<footer>
  &copy; OpenStreetMap contributors (ODbL) &middot; WorldPop (CC BY 4.0) &middot;
  Ookla Speedtest Open Data, aggregated from S3 parquet/performance/type=mobile,
  2024 Q3 &ndash; 2026 Q2 (CC BY-NC-SA 4.0, non-commercial research/education use only) &middot;
  H3 resolution {MAP_H3_RESOLUTION}
</footer>

<script>
const QUARTER_ORDER = {json.dumps(QUARTER_ORDER)};
const ZONES_BY_QUARTER = {json.dumps(zones_by_quarter)};
const TOP5_BY_QUARTER = {json.dumps(top5_by_quarter)};
const KPIS_BY_QUARTER = {json.dumps(kpis_by_quarter)};
const DOMAINS = {json.dumps(domains)};
const POPULATION_BY_EMIRATE = {json.dumps(population_by_emirate)};
{_JS}
</script>

</body>
</html>
"""


_CSS = """
:root {
  --bg: #f7f7f8; --panel: #ffffff; --border: #e2e2e6; --text: #1a1a1e; --text-dim: #6b6b74;
  --accent: #d1293d; --accent-dark: #9c1f2f; --grey-hex: #e3e3e6; --stroke: #ffffff;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text); font-family: -apple-system, "Segoe UI", Roboto, Arial, sans-serif; font-size: 14px; }
header { display: flex; justify-content: space-between; align-items: center; padding: 14px 20px; background: var(--panel); border-bottom: 1px solid var(--border); flex-wrap: wrap; gap: 10px; }
h1 { font-size: 17px; margin: 0; letter-spacing: 0.2px; }
.subtitle { font-size: 10.5px; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.5px; margin-top: 2px; }
.header-controls { display: flex; align-items: center; gap: 14px; flex-wrap: wrap; }
.select-group { display: flex; flex-direction: column; gap: 2px; }
.select-group label { font-size: 9.5px; text-transform: uppercase; letter-spacing: 0.4px; color: var(--text-dim); }
.select-group select { font-size: 12px; padding: 4px 6px; border-radius: 5px; border: 1px solid var(--border); background: var(--panel); color: var(--text); }
.layer-buttons { display: flex; gap: 4px; background: var(--bg); padding: 3px; border-radius: 7px; }
.layer-btn { border: none; background: transparent; padding: 6px 14px; font-size: 12px; font-weight: 600; border-radius: 5px; cursor: pointer; color: var(--text-dim); }
.layer-btn.active { background: var(--accent); color: #fff; }
.disclaimer { background: #fdeeee; color: var(--accent-dark); text-align: center; font-size: 10.5px; letter-spacing: 0.4px; padding: 5px; text-transform: uppercase; border-bottom: 1px solid #f6cfd4; }
.kpi-strip { display: grid; grid-template-columns: repeat(5, 1fr); gap: 1px; background: var(--border); border-bottom: 1px solid var(--border); }
.kpi { background: var(--panel); padding: 12px 18px; }
.kpi-value { font-size: 22px; font-weight: 700; }
.kpi-unit { font-size: 12px; font-weight: 500; color: var(--text-dim); margin-left: 2px; }
.kpi-label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.4px; color: var(--text-dim); margin-top: 2px; }
.kpi-sub { font-size: 10.5px; color: var(--text-dim); opacity: 0.8; }
.stat-up { color: #1a9850; } .stat-down { color: var(--accent); }
main { display: flex; height: calc(100vh - 190px); min-height: 520px; }
.map-pane { flex: 1; position: relative; background: #fbfbfc; overflow: hidden; }
#map-svg { width: 100%; height: 100%; display: block; }
.hex { fill: var(--grey-hex); stroke: var(--stroke); stroke-width: 0.4; cursor: pointer; transition: fill 0.15s; }
.hex:hover { stroke: #999; stroke-width: 1; }
/* Selection is purely visual: a CSS transform scales the rendered SVG path around its own
   fill-box center (transform-box: fill-box) -- it never touches the path's `d` coordinates,
   so the real H3 geometry/area is untouched. 1.04 sits in the requested 103-105% band: the
   smallest bump that reads as "selected" next to same-color neighbors without looking like a
   size change in the underlying data. No transition on `transform` anywhere on `.hex` --
   selecting and deselecting both snap instantly, on purpose (no grow/shrink/fade animation in
   either direction). Re-appended to the end of its parent in JS on selection so the enlarged
   hex draws on top of (not clipped under) its neighbors. */
.hex.selected { stroke: var(--accent); stroke-width: 2.2; transform-box: fill-box; transform-origin: center; transform: scale(1.04); }
.place-label { font-size: 7px; fill: #9a9aa2; letter-spacing: 0.5px; text-transform: uppercase; pointer-events: none; }
.legend { position: absolute; left: 14px; bottom: 14px; background: rgba(255,255,255,0.92); border: 1px solid var(--border); border-radius: 6px; padding: 8px 10px; font-size: 10.5px; width: 190px; }
.legend-title { font-weight: 600; text-transform: uppercase; font-size: 9.5px; color: var(--text-dim); margin-bottom: 5px; }
.legend-gradient { height: 8px; border-radius: 4px; margin-bottom: 3px; }
.legend-scale { display: flex; justify-content: space-between; color: var(--text-dim); font-size: 9.5px; }
.legend-grey { display: flex; align-items: center; gap: 5px; margin-top: 6px; color: var(--text-dim); font-size: 9.5px; }
.legend-grey-swatch { width: 10px; height: 10px; border-radius: 2px; background: var(--grey-hex); border: 1px solid var(--border); }
.side-panel { width: 360px; border-left: 1px solid var(--border); background: var(--panel); overflow-y: auto; }
.panel-block { padding: 14px 16px; border-bottom: 1px solid var(--border); }
.panel-title { font-size: 10.5px; text-transform: uppercase; letter-spacing: 0.4px; color: var(--text-dim); margin-bottom: 8px; font-weight: 600; }
.priority-list { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 6px; }
.priority-item { display: flex; align-items: center; gap: 10px; padding: 8px; border-radius: 6px; cursor: pointer; border: 1px solid var(--border); }
.priority-item:hover { background: var(--bg); }
.priority-item.selected { border-color: var(--accent); background: #fdeeee; }
.priority-rank { width: 18px; height: 18px; border-radius: 50%; background: var(--bg); font-size: 10px; font-weight: 700; display: flex; align-items: center; justify-content: center; flex-shrink: 0; }
.priority-info { flex: 1; min-width: 0; }
.priority-name { font-size: 12px; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.priority-meta { font-size: 10.5px; color: var(--text-dim); }
.priority-score { background: var(--accent); color: #fff; font-size: 11px; font-weight: 700; padding: 2px 7px; border-radius: 10px; flex-shrink: 0; }
.empty-note { font-size: 11.5px; color: var(--text-dim); padding: 6px 2px; }
.zone-detail { flex: 1; }
.zd-region { font-size: 10.5px; color: var(--text-dim); text-transform: uppercase; }
.zd-name { font-size: 15px; font-weight: 700; margin: 3px 0 6px; }
.zd-tags { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 10px; }
.tag { font-size: 9.5px; border: 1px solid var(--border); border-radius: 10px; padding: 2px 8px; color: var(--text-dim); }
.tag.priority-tag { background: var(--accent); color: #fff; border: none; }
.zd-big-numbers { display: flex; gap: 22px; margin: 10px 0; }
.zd-big { text-align: left; }
.zd-big .num { font-size: 26px; font-weight: 700; }
.zd-big .lbl { font-size: 9.5px; color: var(--text-dim); text-transform: uppercase; }
.confidence-bar-track { height: 6px; background: var(--bg); border-radius: 3px; margin: 4px 0; overflow: hidden; }
.confidence-bar-fill { height: 100%; background: var(--accent); }
.zd-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; margin: 12px 0; }
.zd-metric { background: var(--bg); border-radius: 6px; padding: 7px 8px; }
.zd-metric .lbl { font-size: 9px; text-transform: uppercase; color: var(--text-dim); }
.zd-metric .val { font-size: 14px; font-weight: 700; margin-top: 2px; }
.sparkline-wrap { margin: 10px 0; }
.factor-row { display: flex; justify-content: space-between; align-items: center; padding: 5px 0; border-bottom: 1px solid var(--border); font-size: 12px; }
.band { font-size: 10px; font-weight: 700; padding: 2px 8px; border-radius: 9px; }
.band.High { background: #fdeeee; color: var(--accent-dark); }
.band.Medium { background: #fff4e0; color: #a86b00; }
.band.Low { background: #eef7ee; color: #2e7d32; }
.insufficient-msg { background: var(--bg); border-radius: 8px; padding: 14px; font-size: 12.5px; color: var(--text-dim); line-height: 1.5; }
footer { padding: 8px 20px; font-size: 10px; color: var(--text-dim); background: var(--panel); border-top: 1px solid var(--border); }
"""


_JS = """
let currentLayer = 'experience';
let currentQuarter = QUARTER_ORDER[QUARTER_ORDER.length - 1];
let currentEmirate = 'All UAE';
let selectedZone = null;

function lerp(a, b, t) { return a + (b - a) * t; }
function lerpColor(c1, c2, t) {
  const [r1,g1,b1] = c1, [r2,g2,b2] = c2;
  return `rgb(${Math.round(lerp(r1,r2,t))},${Math.round(lerp(g1,g2,t))},${Math.round(lerp(b1,b2,t))})`;
}
const RED = [209,41,61], YELLOW = [255,205,90], GREEN = [26,152,80], GREY = [180,180,186];
function rdylgn(t) {
  t = Math.max(0, Math.min(1, t));
  return t < 0.5 ? lerpColor(RED, YELLOW, t / 0.5) : lerpColor(YELLOW, GREEN, (t - 0.5) / 0.5);
}
function sequential(t, light, dark) {
  t = Math.max(0, Math.min(1, t));
  return lerpColor(light, dark, t);
}
function diverging(t) {
  t = Math.max(-1, Math.min(1, t));
  return t < 0 ? lerpColor(GREY, RED, -t) : lerpColor(GREY, GREEN, t);
}

const LAYER_CONFIG = {
  experience: { field: 'experience_index', label: 'Experience Index (0-100)', kind: 'rdylgn', hint: '< 50 potential concern &middot; 80+ strong' },
  priority:   { field: 'priority_score',   label: 'Priority Score (evidence-weighted)', kind: 'sequential', light: [253,229,229], dark: [155,20,20], hint: 'low &rarr; investigate first' },
  trend:      { field: 'trend_pts_per_qtr',label: 'Trend (index pts / quarter)', kind: 'diverging', hint: 'red = declining vs peers &middot; green = improving' },
  confidence: { field: 'confidence_score', label: 'Confidence (0-100)', kind: 'sequential', light: [232,240,253], dark: [30,64,150], hint: 'higher = more test evidence behind the score' },
};

function currentZones() {
  return ZONES_BY_QUARTER[currentQuarter] || {};
}

function visibleZoneEntries() {
  const zones = currentZones();
  return Object.entries(zones).filter(([id, z]) => currentEmirate === 'All UAE' || z.emirate === currentEmirate);
}

function colorFor(z, layer) {
  const cfg = LAYER_CONFIG[layer];
  const v = z[cfg.field];
  if (v === null || v === undefined) return null;
  const [lo, hi] = DOMAINS[layer];
  const t = (v - lo) / (hi - lo || 1);
  if (cfg.kind === 'rdylgn') return rdylgn(t);
  if (cfg.kind === 'sequential') return sequential(t, cfg.light, cfg.dark);
  if (cfg.kind === 'diverging') return diverging(t);
}

function renderLayer() {
  const cfg = LAYER_CONFIG[currentLayer];
  const zones = currentZones();
  const filterActive = currentEmirate !== 'All UAE';
  // Must visit every hexagon physically drawn on the map, not just the ones with a record in
  // `zones` (the current quarter) -- one row is one (h3_cell, quarter) pair, so a hex with no
  // entry here has no valid analytical value THIS quarter, full stop, even if it was colored a
  // moment ago under a different quarter. Only touching `zones`' own keys left every other hex's
  // last-painted `style.fill` sitting there untouched, which is exactly how a previous quarter's
  // (or a previous layer's) color kept "surviving" a quarter change.
  document.querySelectorAll('.hex').forEach(el => {
    const id = el.id.slice(4); // strip the 'hex-' prefix
    const z = zones[id];
    if (!z) {
      el.style.fill = '#e3e3e6'; // no record for this h3_cell in this quarter -- never inherit an old one
      return;
    }
    const inFilter = !filterActive || z.emirate === currentEmirate;
    // Out-of-filter zones fall back to the same neutral grey as a genuinely unclassified
    // hex (never a faded version of their real color) -- an emirate filter must mean "no
    // other emirate's analytical value is shown," not "shown a little less."
    el.style.fill = inFilter ? (colorFor(z, currentLayer) || '#e3e3e6') : '#e3e3e6';
  });
  renderLegend(cfg);
}

function renderLegend(cfg) {
  const [lo, hi] = DOMAINS[currentLayer];
  let gradientCss;
  if (cfg.kind === 'rdylgn') gradientCss = 'linear-gradient(90deg, rgb(209,41,61), rgb(255,205,90), rgb(26,152,80))';
  else if (cfg.kind === 'diverging') gradientCss = 'linear-gradient(90deg, rgb(209,41,61), rgb(180,180,186), rgb(26,152,80))';
  else gradientCss = `linear-gradient(90deg, rgb(${cfg.light.join(',')}), rgb(${cfg.dark.join(',')}))`;
  document.getElementById('legend').innerHTML = `
    <div class="legend-title">${cfg.label}</div>
    <div class="legend-gradient" style="background:${gradientCss}"></div>
    <div class="legend-scale"><span>${lo}</span><span>${hi}</span></div>
    <div class="legend-scale" style="margin-top:4px">${cfg.hint}</div>
    <div class="legend-grey"><div class="legend-grey-swatch"></div>insufficient public evidence &mdash; not classified</div>
  `;
}

function renderKpis() {
  const entries = visibleZoneEntries();
  const filterActive = currentEmirate !== 'All UAE';
  const popDenom = POPULATION_BY_EMIRATE[currentEmirate];
  const popSum = entries.reduce((s, [, z]) => s + z.population, 0);
  const priorityCount = entries.filter(([, z]) => z.priority_zone).length;
  const downloads = entries.map(([, z]) => z.download_mbps).sort((a, b) => a - b);
  const medianDownload = downloads.length ? downloads[Math.floor(downloads.length / 2)] : 0;

  const idx = QUARTER_ORDER.indexOf(currentQuarter);
  let qoq = null;
  if (idx > 0) {
    const prevEntries = Object.entries(ZONES_BY_QUARTER[QUARTER_ORDER[idx - 1]] || {})
      .filter(([id, z]) => !filterActive || z.emirate === currentEmirate);
    if (prevEntries.length && entries.length) {
      const med = arr => { const s = arr.slice().sort((a,b)=>a-b); return s[Math.floor(s.length/2)]; };
      qoq = Math.round((med(entries.map(([,z])=>z.experience_index)) - med(prevEntries.map(([,z])=>z.experience_index))) * 10) / 10;
    }
  }
  const qoqClass = qoq === null ? '' : (qoq >= 0 ? 'stat-up' : 'stat-down');
  const qoqSign = qoq !== null && qoq >= 0 ? '+' : '';
  const qoqDisplay = qoq === null ? '&ndash;' : `${qoqSign}${qoq}<span class="kpi-unit">pts</span>`;
  const prevLabel = idx > 0 ? QUARTER_ORDER[idx - 1] : '–';

  document.getElementById('kpi-strip').innerHTML = `
    <div class="kpi"><div class="kpi-value">${entries.length}</div><div class="kpi-label">Scored zones</div><div class="kpi-sub">meeting the evidence threshold</div></div>
    <div class="kpi"><div class="kpi-value">${popDenom ? Math.round(1000*popSum/popDenom)/10 : 0}%</div><div class="kpi-label">Population represented</div><div class="kpi-sub">est. residents in scored zones</div></div>
    <div class="kpi"><div class="kpi-value" style="color:var(--accent)">${priorityCount}</div><div class="kpi-label">Priority zones</div><div class="kpi-sub">flagged nationally, shown in view</div></div>
    <div class="kpi"><div class="kpi-value">${medianDownload.toFixed(0)}<span class="kpi-unit">Mbps</span></div><div class="kpi-label">Median download</div><div class="kpi-sub">across scored zones in view</div></div>
    <div class="kpi"><div class="kpi-value ${qoqClass}">${qoqDisplay}</div><div class="kpi-label">Experience Index QoQ</div><div class="kpi-sub">vs ${prevLabel}</div></div>
  `;
}

function setLayer(layer) {
  currentLayer = layer;
  document.querySelectorAll('.layer-btn').forEach(b => b.classList.toggle('active', b.dataset.layer === layer));
  renderLayer();
}

// One shared selection state (`selectedZone`) drives the map hex, the priority-list row
// highlight, and the details panel alike -- a quarter/emirate change never invents a second
// selection concept, it only re-validates the same one against the new view.
function clearSelection() {
  if (selectedZone) {
    const prevEl = document.getElementById('hex-' + selectedZone);
    if (prevEl) prevEl.classList.remove('selected');
  }
  selectedZone = null;
  document.querySelectorAll('.priority-item').forEach(li => li.classList.remove('selected'));
  document.getElementById('zone-detail').innerHTML =
    '<div class="empty-note">Click a hexagon, or a zone in the priority list, to see its details.</div>';
}

function refreshSelection() {
  if (!selectedZone) return;
  const filterActive = currentEmirate !== 'All UAE';
  const raw = currentZones()[selectedZone];
  // A zone with no data THIS quarter still physically exists (every hexagon is a fixed
  // geographic cell, drawn at every quarter) -- selectZone already renders an honest "no
  // classification this quarter" panel for that, so it stays selected. A zone that belongs to
  // a DIFFERENT emirate than the active filter is different: it's now drawn as plain
  // background grey (never colored), so keeping it "selected" would leave an enlarged,
  // highlighted hexagon with a details panel for a zone the view no longer shows -- exactly
  // the stale-selection state that must not happen. Clear it instead.
  if (filterActive && raw && raw.emirate !== currentEmirate) {
    clearSelection();
  } else {
    selectZone(selectedZone);
  }
}

function setQuarter(q) {
  currentQuarter = q;
  renderLayer();
  renderKpis();
  renderPriorityList();
  refreshSelection();
}

function setEmirate(e) {
  currentEmirate = e;
  renderLayer();
  renderKpis();
  renderPriorityList();
  refreshSelection();
}

function sparklineSvg(values) {
  const pts = values.map((v, i) => ({ i, v })).filter(p => p.v !== null);
  if (pts.length < 2) return '<div style="font-size:11px;color:var(--text-dim)">Not enough classified quarters for a trend line.</div>';
  const w = 300, h = 46, pad = 4;
  const vals = pts.map(p => p.v);
  const lo = Math.min(...vals), hi = Math.max(...vals);
  const x = i => pad + (i / 7) * (w - 2 * pad);
  const y = v => h - pad - ((v - lo) / (hi - lo || 1)) * (h - 2 * pad);
  const line = pts.map(p => `${x(p.i)},${y(p.v)}`).join(' ');
  const dots = pts.map(p => `<circle cx="${x(p.i)}" cy="${y(p.v)}" r="2.4" fill="#d1293d"></circle>`).join('');
  return `<svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}">
    <polyline points="${line}" fill="none" stroke="#d1293d" stroke-width="1.8"></polyline>${dots}
  </svg>`;
}

function shortId(id) {
  const trimmed = id.replace(/f+$/, '');
  return trimmed.slice(-6);
}

function factorRow(label, band) {
  return `<div class="factor-row"><span>${label}</span><span class="band ${band}">${band}</span></div>`;
}

function selectZone(id) {
  if (selectedZone) {
    const prev = document.getElementById('hex-' + selectedZone);
    if (prev) prev.classList.remove('selected');
  }
  selectedZone = id;
  const el = document.getElementById('hex-' + id);
  if (el) {
    el.classList.add('selected');
    // SVG stacking order is DOM order, not z-index -- re-appending moves this <path> to the
    // end of its parent so the enlarged hex draws over its neighbors instead of being
    // partly hidden under them. A DOM reorder only, never touches the path's `d` coordinates.
    el.parentNode.appendChild(el);
  }

  document.querySelectorAll('.priority-item').forEach(li => li.classList.toggle('selected', li.dataset.id === id));

  const filterActive = currentEmirate !== 'All UAE';
  const raw = currentZones()[id];
  // A zone that exists but belongs to a different emirate than the active filter must not
  // open its real data panel either -- the map already shows it as plain background grey,
  // and the panel has to agree, not leak another emirate's numbers through a click.
  const outOfFilter = !!raw && filterActive && raw.emirate !== currentEmirate;
  const z = outOfFilter ? null : raw;
  const panel = document.getElementById('zone-detail');
  if (!z) {
    panel.innerHTML = outOfFilter ? `
      <div class="zd-region">H3 zone &middot; ${id}</div>
      <div class="zd-name">Outside the current emirate filter</div>
      <div class="insufficient-msg">
        This hexagon belongs to ${raw.emirate}, not ${currentEmirate}. Switch the Emirate filter
        to ${raw.emirate} (or "All UAE") to see its data.
      </div>` : `
      <div class="zd-region">H3 zone &middot; ${id}</div>
      <div class="zd-name">No classification this quarter</div>
      <div class="insufficient-msg">
        This hexagon has too few public Ookla measurements this quarter to compute a reliable
        Experience Index -- shown as "insufficient public evidence," not treated as poor
        performance. It may still have data in other quarters.
      </div>`;
    return;
  }
  panel.innerHTML = `
    <div class="zd-region">${z.emirate} &middot; H3 ${id} &middot; ${currentQuarter}</div>
    <div class="zd-name">Zone ${shortId(id)}</div>
    <div class="zd-tags">
      <span class="tag">Peer: ${z.peer_group}</span>
      ${z.priority_zone ? '<span class="tag priority-tag">Priority zone</span>' : ''}
      ${z.deteriorating ? '<span class="tag">Deteriorating</span>' : ''}
    </div>
    <div class="zd-big-numbers">
      <div class="zd-big"><div class="num">${z.priority_score}</div><div class="lbl">Priority score</div></div>
      <div class="zd-big"><div class="num">${z.experience_index}</div><div class="lbl">Experience index</div></div>
    </div>
    <div class="lbl" style="font-size:10px;color:var(--text-dim)">
      Peer-group median ${z.peer_median} &middot; ${z.peer_gap >= 0 ? '+' : ''}${z.peer_gap} pts (${z.peer_gap_pct}%)
    </div>
    <div style="margin-top:10px">
      <div class="lbl" style="font-size:10px;color:var(--text-dim)">CONFIDENCE &middot; ${z.confidence_score}/100</div>
      <div class="confidence-bar-track"><div class="confidence-bar-fill" style="width:${z.confidence_score}%"></div></div>
      <div style="font-size:10.5px;color:var(--text-dim)">${z.tests} tests &middot; ${z.devices} devices &middot; ${z.quarters_observed}/8 quarters observed</div>
    </div>
    <div class="zd-grid">
      <div class="zd-metric"><div class="lbl">Download</div><div class="val">${z.download_mbps} Mbps</div></div>
      <div class="zd-metric"><div class="lbl">Upload</div><div class="val">${z.upload_mbps} Mbps</div></div>
      <div class="zd-metric"><div class="lbl">Latency</div><div class="val">${z.latency_ms} ms</div></div>
      <div class="zd-metric"><div class="lbl">Population</div><div class="val">${z.population.toLocaleString()}</div></div>
      <div class="zd-metric"><div class="lbl">Trend</div><div class="val">${z.trend_pts_per_qtr === null ? '&ndash;' : (z.trend_pts_per_qtr >= 0 ? '+' : '') + z.trend_pts_per_qtr + '/qtr'}</div></div>
      <div class="zd-metric"><div class="lbl">ML anomaly</div><div class="val" style="font-size:11px">${z.peer_gap_ml_anomaly || z.temporal_anomaly_ml_flag ? 'Flagged' : 'None'}</div></div>
    </div>
    <div class="sparkline-wrap">
      <div class="panel-title" style="margin-bottom:2px">8-quarter Experience Index trend</div>
      ${sparklineSvg(z.sparkline)}
    </div>
    <div class="panel-title">Why this priority &mdash; factor contributions</div>
    ${factorRow('Experience gap vs. peers', z.bands.peer_gap)}
    ${factorRow('ML anomaly strength', z.bands.ml_anomaly)}
    ${factorRow('Deterioration', z.bands.deterioration)}
    ${factorRow('Population exposure', z.bands.population)}
  `;
}

function renderPriorityList() {
  const filterActive = currentEmirate !== 'All UAE';
  const title = document.getElementById('priority-list-title');
  const list = document.getElementById('priority-list');

  let ranked;
  if (!filterActive) {
    title.textContent = `${currentQuarter} · top priority zones (national)`;
    ranked = TOP5_BY_QUARTER[currentQuarter] || [];
  } else {
    title.textContent = `${currentQuarter} · top priority zones in ${currentEmirate}`;
    ranked = visibleZoneEntries()
      .sort((a, b) => b[1].priority_score - a[1].priority_score)
      .slice(0, 5)
      .map(([id]) => id);
  }

  if (!ranked.length) {
    list.innerHTML = '<div class="empty-note">No classified zones for this emirate/quarter combination.</div>';
    return;
  }

  list.innerHTML = ranked.map((id, i) => {
    const z = currentZones()[id];
    if (!z) return '';
    return `<li class="priority-item" data-id="${id}" onclick="selectZone('${id}')">
      <div class="priority-rank">${i + 1}</div>
      <div class="priority-info">
        <div class="priority-name">${z.emirate} &middot; ${shortId(id)}</div>
        <div class="priority-meta">${z.peer_group} &middot; Exp ${z.experience_index} &middot; Conf ${z.confidence_score}%</div>
      </div>
      <div class="priority-score">P${Math.round(z.priority_score)}</div>
    </li>`;
  }).join('');
}

renderKpis();
renderPriorityList();
renderLayer();
const initialTop5 = TOP5_BY_QUARTER[currentQuarter] || [];
if (initialTop5.length) selectZone(initialTop5[0]);
"""


def main():
    all_data_cells = pd.read_parquet(
        Path("data/processed/zone_priority.parquet"), columns=["h3_cell"]
    )["h3_cell"].unique().tolist()
    hexagons, project, width, height, labels = build_hex_grid(
        Path("data/raw/boundary/uae_boundary.geojson"), extra_cells=all_data_cells
    )
    zones_by_quarter, top5_by_quarter, kpis_by_quarter, domains, population_by_emirate = build_zone_data(
        Path("data/processed/zone_priority.parquet"), Path("data/processed/population_zones_uae.parquet")
    )
    html = render_html(hexagons, width, height, labels, zones_by_quarter, top5_by_quarter,
                        kpis_by_quarter, domains, population_by_emirate)

    out_path = Path("data/processed/uae_dashboard.html")
    out_path.write_text(html, encoding="utf-8")
    print(f"Saved: {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")
    print(f"Hexagons drawn: {len(hexagons)} | quarters with data: {len(zones_by_quarter)}")
    print(f"Latest quarter classified zones: {kpis_by_quarter[list(kpis_by_quarter)[-1]]['scored_zones']}")


if __name__ == "__main__":
    main()
