# UAE Mobile Network Experience Intelligence 


## Project Overview

This project aims to develop an AI-powered geographic intelligence
platform for analyzing publicly measured mobile network experience
across the UAE.

## Setup

```
pip install -r requirements.txt
```

Tested with Python 3.13. Then open any notebook in `notebooks/` (VS Code's Jupyter extension or
JupyterLab both work) and run all cells top to bottom — each collection notebook downloads its own
raw data on first run, so no manual download step is needed.

## Data Sources

| Dataset | Status | Notebook | Notes |
|---|---|---|---|
| Ookla Speedtest Open Data (mobile) | Acquired, UAE-clipped, aggregated to H3 zones | [`01_ookla_collection.ipynb`](notebooks/01_ookla_collection.ipynb) → [`05_zone_aggregation.ipynb`](notebooks/05_zone_aggregation.ipynb) | 8 quarters, 2024 Q3 → 2026 Q2. CC BY-NC-SA 4.0 (non-commercial). |
| WorldPop UAE population | Acquired, validated, aggregated to H3 zones | [`02_worldpop_collection.ipynb`](notebooks/02_worldpop_collection.ipynb) → [`05_zone_aggregation.ipynb`](notebooks/05_zone_aggregation.ipynb) | R2025A constrained, 2026, 100m. National total 11,476,873 matches brief; conserved exactly through H3 aggregation. CC BY 4.0. |
| OpenStreetMap | Acquired, UAE-clipped, aggregated to H3 zones | [`03_osm_collection.ipynb`](notebooks/03_osm_collection.ipynb) → [`07_osm_feature_extraction.ipynb`](notebooks/07_osm_feature_extraction.ipynb) | Geofabrik GCC States PBF, buildings/roads/POIs extracted with pyosmium, land-use checked but not used as a classifier (too thin — see notebook). ODbL. |

All three are joined into one analysis-ready table by
[`08_zone_quarter_table.ipynb`](notebooks/08_zone_quarter_table.ipynb) →
`data/processed/zone_quarter_table.parquet` (one row per `(H3 cell, quarter)`, 13,597 rows), then
[`09_peer_group_classifier.ipynb`](notebooks/09_peer_group_classifier.ipynb) adds a `peer_group`
column to that same table (see below).

Raw files live under `data/raw/<dataset>/` and are **gitignored on purpose**: the 8 Ookla quarters
alone total ~1.5GB with individual files over GitHub's 100MB push limit, so they can't be committed
at all, and the OSM extract (~250MB) isn't far behind. Every collection notebook downloads its own
raw inputs (including the UAE boundary polygon) if they're not already present locally, so running
a notebook end to end reproduces `data/raw/` from nothing — you never need to source these files any
other way. Processed, UAE-clipped outputs (small) live under `data/processed/` and **are** committed,
so you don't have to re-run the raw collection notebooks just to get a working dataset.

**Geographic unit:** H3 resolution 7, chosen empirically in
[`04_h3_resolution_choice.ipynb`](notebooks/04_h3_resolution_choice.ipynb) — it reproduces the
brief's own cited density figures (median 4 tests/zone, ~83% of zones below 30 tests) on
independently rebuilt data, yields ~300 well-measured zones nationally as the brief predicts, and
has quarter-over-quarter stability close to resolution 6 while resolving far more zones for
drill-down. The decision (with supporting numbers) is persisted at
`data/processed/h3_resolution.json` so downstream notebooks can't silently drift out of sync with it.

## Code Layout

- `notebooks/` — data collection, validation, and exploration (Ookla, WorldPop, OSM, H3
  zone aggregation, mapping, peer-group classification). Each collection notebook is
  self-contained and reproduces its own raw inputs on first run.
- `src/` — deterministic Python analytics code, not notebooks. Currently
  [`compute_scores.py`](src/compute_scores.py): pure, side-effect-free functions for the
  Experience Index, Confidence Score, and Peer Gap. One entry point,
  `score_zone_quarters(df)`, runs the full pipeline; notebooks and the eventual copilot import
  from here rather than duplicating the formulas. Kept as plain Python rather than a notebook
  since it's library code meant to be imported, not read top-to-bottom.

## Current Phase

Phase 2 (Scores), building on a cleared Phase 1 gate. Real public UAE mobile measurements render
on a map, per-zone, in [`06_first_uae_map.ipynb`](notebooks/06_first_uae_map.ipynb) (saved
standalone at `data/processed/uae_map_2026Q2.html`) — raw metrics only, no scores yet.

Since then: [`08_zone_quarter_table.ipynb`](notebooks/08_zone_quarter_table.ipynb) joins Ookla
(per-quarter), WorldPop, and OSM density into one master table,
`data/processed/zone_quarter_table.parquet` (one row per `(H3 cell, quarter)`, 13,597 rows) —
everything downstream reads from this file instead of re-joining three parquet files each time.
[`09_peer_group_classifier.ipynb`](notebooks/09_peer_group_classifier.ipynb) then builds the
brief's mandatory peer-group classifier: KMeans (`k=4`) on log-transformed, standardized
population density, building-footprint %, POI density and road density (never OSM land-use tag,
confirmed too thin on this data), with human-readable labels assigned from cluster centroids via
an explainable rule (density-intensity rank, then POI-to-built ratio for the industrial/
residential split). All four groups clear a 30-zone minimum and show real within-group spread in
raw download speed (proxy pending the real Experience Index) — both risks the brief calls out
explicitly. Result:

| Peer group | Zones | Pop. density (per km²) | Building footprint % | POI density (per km²) | Road density (km/km²) |
|---|---|---|---|---|---|
| commercial/urban-core | 303 | 2,823 | 9.7% | 12.4 | 20.4 |
| low-density residential | 854 | 860 | 0.7% | 0.6 | 8.8 |
| industrial | 1,413 | 138 | 0.07% | 0.09 | 2.6 |
| rural/edge | 830 | 4 | 0.01% | 0.03 | 0.6 |

Saved to `data/processed/peer_groups_uae.parquet` (standalone) and merged as a `peer_group`
column onto `zone_quarter_table.parquet`. One honest caveat for the report: at H3 res-7, the
"industrial" label is the largest group (41.6% of zones) and reads more like "sparse/light
development" than strictly industrial land — a proxy from density signature, not a direct
land-use tag (which the brief already ruled out as too thin). Worth restating in the limitations
section, not hidden.

[`10_scores_and_peer_gap.ipynb`](notebooks/10_scores_and_peer_gap.ipynb) then runs the real
Experience Index, Confidence Score, and Peer Gap for the first time — output at
`data/processed/zone_scores.parquet` (13,597 rows, all `zone_quarter_table` columns plus
`experience_index`, `confidence_score`, `insufficient_evidence`, `peer_gap`,
`peer_gap_pct`). Headline numbers:

- **54.0% of zone-quarters nationally are flagged insufficient evidence** (below 5 tests/quarter)
  and get no Experience Index at all — matches the brief's own cited median of ~4 tests/zone
  exactly. Latest quarter (2026Q2) alone: 1,815 zones measured, 868 classified.
- **Confidence correctly separates real cases**: a real zone-quarter with 5,064 tests/156 devices
  across all 8 quarters scores 70.9 confidence; a real 1-test/1-device zone scores 43.1 and gets
  `NaN` Experience Index — "insufficient public evidence," not a bad score.
- **Weight sensitivity (T5 preview, on 868 real classified zones, not 8 synthetic ones)**:
  perturbing Experience weights ±15% barely moves the ranking — Spearman ρ = 0.998–0.999,
  top-20 overlap 19–20 out of 20 across all three perturbations tested. The current 50/20/30
  split is not fragile on this data.
- **Peer Gap is live**: peer-group medians differ meaningfully (39.0 industrial vs. 44.2
  commercial/urban-core, 2026Q2), so the ten worst-gap zones per quarter can now be listed
  directly — the first of the brief's ten canonical questions this system can answer end to end.

## Phase 3 (Intelligence) — trend, ML anomaly detection, priority

Built directly as `src/*.py` (no exploratory notebook for these three — same reusable-logic
pattern as `compute_scores.py`, just skipping the narrated write-up for now to get to the
product UI faster):

- [`src/trends.py`](src/trends.py) — deterioration relative to peer-group trend (tracks how
  `peer_gap` moves quarter to quarter, not raw Mbps), flagged only after **3 consecutive**
  declining quarters. Tried at 2 first (the brief's lower bound): that flagged 413 zones
  nationally at some point across the window, which is sampling noise on data this sparse, not
  genuine deterioration (the brief's own comparable figures are 71–99 zones). 3 consecutive
  declines brings that to 107 — much closer to the brief's cited scale, and still within its
  "two or three quarters" allowance.
- [`src/anomaly_detection.py`](src/anomaly_detection.py) — the mandatory ML capability: two
  Isolation Forest models (Peer Gap: vs. peer-group z-scores this quarter; Temporal Anomaly: vs.
  the zone's own history), each compared against a deterministic bottom-decile-download
  baseline. Real, honest divergence: Peer Gap ML flags 313 zone-quarters vs. the baseline's 381,
  with only 65 in common — the ML is finding something different from the simple rule, not just
  rediscovering it. **This is a first-pass detector, not the formal T2 benchmark** (synthetic
  anomaly injection with precision/recall/F1 on a blind hold-out set) — that's still open, see
  below.
- [`src/priority.py`](src/priority.py) — `Priority = 0.35·PeerGap + 0.20·MLAnomaly +
  0.20·Deterioration + 0.25·Population`, then **multiplied** by `confidence/100` — not added as
  a fifth factor — so the brief's fixed rule (low confidence can never auto-become high
  priority) is structural, not a judgment call that could leak through. Top 10% per quarter
  flagged as priority zones (89 of 868 in the latest quarter). Weights are a first pass, not yet
  sensitivity-tested (T5) — see below.
- [`src/run_pipeline.py`](src/run_pipeline.py) — chains all of the above (`compute_scores` →
  `trends` → `anomaly_detection` → `priority`) into one script,
  `data/processed/zone_priority.parquet`. Re-run this whenever an upstream input changes.

## Phase 4 (Product) — the interactive map, minus the copilot

[`src/build_dashboard.py`](src/build_dashboard.py) generates capability #1 (Interactive UAE
experience map: Experience/Priority/Trend/Confidence layers, national view, zone drill-down) as
one self-contained HTML file — `data/processed/uae_dashboard.html` — matching the brief's design
mockup, minus the copilot chat (a separate capability, not built yet). Plain SVG + vanilla JS,
no external libraries or CDN dependency: every hexagon's geometry is precomputed in Python (real
H3 boundaries, a simple equirectangular projection) and written straight into the page, so the
browser only recolors paths on click — no client-side geometry math, and it works fully offline.
15,695 background hexagons at H3 res 7 (matching every other notebook — resolution 8 covering
the whole country was the ~110,000-hexagon, 50MB-file mistake `06_first_uae_map.ipynb` originally
made). Clicking an unclassified (grey) hexagon correctly shows "insufficient public evidence,"
never a fabricated score. Verified with a headless-browser screenshot pass (all four layers, zone
drill-down, and the insufficient-evidence state) — no console errors, `src/run_pipeline.py` →
`src/build_dashboard.py` reproduces cleanly end to end.

Run: `python -m src.run_pipeline && python -m src.build_dashboard` from the repo root.

**Known limitations, stated plainly rather than hidden:**
- No real place names — OSM place/locality data isn't joined in yet, so zones show as
  `H3 <id>` plus a "nearest emirate" approximation (nearest-nominal-center distance, not a real
  boundary lookup). Inventing names like the mockup's "Al Warsan / International City" would
  violate the brief's "never invented" rule, so this is left honest rather than pretty.
- Only the latest quarter (2026Q2) is interactive; the quarter dropdown is a label, not a
  working selector — full time-travel across all 8 quarters is future work.
- The ML anomaly detector, trend threshold, and priority weights are working first passes, not
  yet formally validated (see Next Steps).

## Next Steps

1. **Data dictionary** — document every field, transformation and source across all datasets.
2. **T0 coverage/representativeness audit** — scored vs. eligible zones by emirate, share of
   population in sufficiently-sampled cells, measurement availability across urban/suburban/rural.
3. **Formal T2 synthetic anomaly benchmark** — inject known anomalies (download −40%, latency
   +80%, gradual 3-quarter deterioration, combined), precision/recall/F1/FPR on a blind hold-out
   set, for both ML detectors against the baseline.
4. **T5 priority sensitivity test** — perturb the 0.35/0.20/0.20/0.25 weights ±10–20%, report
   whether the top-ranked zones reshuffle.
5. **Grounded copilot + AI zone briefs** — capability #7, the one piece of the dashboard
   deliberately left out so far.
6. **Testing & evaluation pack** — T0–T7 formally run and reported, business case, architecture
   diagram, limitations statement, presentation.

**Note:** `src/experience_confidence_scores.py` (an earlier synthetic-data draft of the
Experience/Confidence formulas) is now superseded by `src/compute_scores.py` and can likely be
removed — left in place for now since it was Saif's contribution, worth a quick team check
before deleting.