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
| Ookla Speedtest Open Data (mobile) | Acquired, UAE-clipped | [`01_ookla_collection.ipynb`](notebooks/01_ookla_collection.ipynb) | 8 quarters, 2024 Q3 → 2026 Q2. CC BY-NC-SA 4.0 (non-commercial). |
| WorldPop UAE population | Acquired, validated | [`02_worldpop_collection.ipynb`](notebooks/02_worldpop_collection.ipynb) | R2025A constrained, 2026, 100m. National total 11,476,873 matches brief. CC BY 4.0. |
| OpenStreetMap | Raw extract downloaded; UAE feature extraction pending | [`03_osm_collection.ipynb`](notebooks/03_osm_collection.ipynb) | Geofabrik GCC States PBF (no UAE-only subregion). ODbL. |

Raw files live under `data/raw/<dataset>/` and are **gitignored on purpose**: the 8 Ookla quarters
alone total ~1.5GB with individual files over GitHub's 100MB push limit, so they can't be committed
at all, and the OSM extract (~250MB) isn't far behind. Every collection notebook downloads its own
raw inputs (including the UAE boundary polygon) if they're not already present locally, so running
a notebook end to end reproduces `data/raw/` from nothing — you never need to source these files any
other way. Processed, UAE-clipped outputs (small) live under `data/processed/` and **are** committed,
so you don't have to re-run the raw collection notebooks just to get a working dataset.

## Current Phase

Dataset acquisition and validation (Phase 1: Foundation). Gate to clear before moving on: display
real public UAE mobile measurements on a map.

## Next Steps

1. **Pick and justify the H3 resolution** for zone aggregation. Raw Ookla tiles are too sparse to
   score individually (median 2 tests/tile) — need to empirically compare res 6/7/8 on measurement
   density and score stability rather than just taking the brief's suggested res 7 on faith.
2. **OSM UAE feature extraction** — use pyosmium (not pyrosm, too slow on this file size) to pull
   UAE buildings, roads, POIs and land-use polygons out of the GCC States PBF, clipped to
   `data/raw/boundary/uae_boundary.geojson`. Feeds the peer-group density composite (population +
   building-footprint + POI + road density) — do not classify zones by OSM land-use tag alone.
3. **Data dictionary** — document every field, transformation and source across all three datasets.
4. **T0 coverage/representativeness audit** — scored vs. eligible zones by emirate, share of
   population in sufficiently-sampled cells, measurement availability across urban/suburban/rural.
5. **First UAE map** with real measurements — clears the Phase 1 gate above.

After that: Experience Index and Confidence Score (Phase 2), then anomaly detection, trend
intelligence and the priority engine (Phase 3), the product UI and copilot (Phase 4), and finally
the testing/evaluation pack (Phase 5).