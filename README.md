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

Dataset acquisition and validation.