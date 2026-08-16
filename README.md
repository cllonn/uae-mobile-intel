# UAE Mobile Network Experience Intelligence 


## Project Overview

This project aims to develop an AI-powered geographic intelligence
platform for analyzing publicly measured mobile network experience
across the UAE.

## Data Sources

| Dataset | Status | Notebook | Notes |
|---|---|---|---|
| Ookla Speedtest Open Data (mobile) | Acquired, UAE-clipped | [`01_ookla_collection.ipynb`](notebooks/01_ookla_collection.ipynb) | 8 quarters, 2024 Q3 → 2026 Q2. CC BY-NC-SA 4.0 (non-commercial). |
| WorldPop UAE population | Acquired, validated | [`02_worldpop_collection.ipynb`](notebooks/02_worldpop_collection.ipynb) | R2025A constrained, 2026, 100m. National total 11,476,873 matches brief. CC BY 4.0. |
| OpenStreetMap | Raw extract downloaded; UAE feature extraction pending | [`03_osm_collection.ipynb`](notebooks/03_osm_collection.ipynb) | Geofabrik GCC States PBF (no UAE-only subregion). ODbL. |

Raw files live under `data/raw/<dataset>/` (gitignored); processed, UAE-clipped outputs live under `data/processed/`.

## Current Phase

Dataset acquisition and validation.