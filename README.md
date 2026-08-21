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
  zone aggregation, mapping). Each collection notebook is self-contained and reproduces its
  own raw inputs on first run.
- `src/` — deterministic Python analytics code, not notebooks. Currently
  [`compute_scores.py`](src/compute_scores.py): the Experience Index and Confidence Score
  formulas (Phase 2). Runs on synthetic sample data shaped like the real zone-quarter table
  until `05_zone_aggregation.ipynb`'s output is reshaped to match (see the file's schema
  docstring and section 6 for the one-line swap). Kept as plain Python rather than a notebook
  since it's library code other notebooks and the eventual copilot will import, not an
  exploratory analysis.

## Current Phase

Dataset acquisition and validation (Phase 1: Foundation). **Gate cleared:** real public UAE mobile
measurements now render on a map, per-zone, in
[`06_first_uae_map.ipynb`](notebooks/06_first_uae_map.ipynb) (saved standalone at
`data/processed/uae_map_2026Q2.html`). This map shows raw metrics only (download, upload, latency,
test count, population) — the Experience/Confidence/Priority scores are Phase 2/3 work.

## Next Steps

1. **Peer-group composite classifier** — join `osm_density_zones_uae.parquet` (building-footprint,
   POI and road density, from [`07_osm_feature_extraction.ipynb`](notebooks/07_osm_feature_extraction.ipynb))
   with `population_zones_uae.parquet` and classify zones into the brief's four stable groups
   (commercial/urban-core, low-density residential, industrial, rural/edge) — never by OSM
   land-use tag alone, confirmed too thin for commercial/retail on this data.
2. **Data dictionary** — document every field, transformation and source across all three datasets.
3. **T0 coverage/representativeness audit** — scored vs. eligible zones by emirate, share of
   population in sufficiently-sampled cells, measurement availability across urban/suburban/rural.

After that: Experience Index and Confidence Score (Phase 2), then anomaly detection, trend
intelligence and the priority engine (Phase 3), the product UI and copilot (Phase 4), and finally
the testing/evaluation pack (Phase 5).