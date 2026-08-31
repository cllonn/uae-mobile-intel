# Data dictionary

Every field, transformation, and source across the pipeline, from raw acquisition to the final
table (`zone_priority.parquet`) that the dashboard and copilot both read. Field names match the
actual parquet columns — verified against the live files, not written from memory.

## Raw sources

| Dataset | Version / access | Licence | What it provides |
|---|---|---|---|
| Ookla Speedtest Open Data (Mobile) | 8 quarters, 2024 Q3 – 2026 Q2, downloaded from `ookla-open-data` S3 bucket (`parquet/performance/type=mobile/year=<Y>/quarter=<Q>/`) | CC BY-NC-SA 4.0 (non-commercial) | Per-tile (~610m) quarterly aggregates: avg download/upload (kbps), latency, loaded latency, test count, device count, quadkey, tile centroid |
| WorldPop UAE population | R2025A constrained, 2026, 100m resolution | CC BY 4.0 | Estimated population per 100m raster pixel |
| OpenStreetMap | Geofabrik GCC States extract, downloaded via `download.geofabrik.de` | ODbL (attribution + share-alike) | Building footprints, roads, POIs, and the 7 emirate `admin_level=4` administrative boundaries |

## Processed tables

### `zone_quarter_table.parquet` — one row per (H3 res-6 cell, quarter), 5,011 rows

The master pre-scoring join: Ookla (per-quarter) + WorldPop (static) + OSM density (static) +
peer group (static) + emirate (static).

| Field | Meaning | Source / transformation |
|---|---|---|
| `h3_cell` | H3 resolution-6 cell ID | Ookla tile lat/lon → `h3.latlng_to_cell` |
| `quarter` | e.g. `"2026Q2"` | Ookla file's year/quarter |
| `n_tiles` | Number of raw ~610m Ookla tiles aggregated into this cell-quarter | Count from `01_ookla_collection.ipynb` |
| `tests`, `devices` | Test count / device count, summed across tiles in the cell | Raw Ookla, test-weighted sum (`05_zone_aggregation.ipynb`) — `devices` is an upper-bound proxy (no device ID in the public dataset, so a device active in two tiles is counted twice) |
| `download_mbps`, `upload_mbps` | Test-weighted average speed | Ookla kbps → Mbps, weighted by `tests` per source tile |
| `latency_ms` | Unloaded latency, test-weighted average | Always populated |
| `latency_loaded_ms` | Loaded latency (brief's preferred metric) | ~99% populated for UAE; falls back to `latency_ms` downstream |
| `population` | Estimated residents in this H3 cell | WorldPop pixel population summed by cell (`05_zone_aggregation.ipynb`), national total conserved exactly |
| `building_count`, `building_area_m2`, `road_count`, `road_length_m`, `poi_count`, `zone_area_km2` | Raw OSM density counts | `07_osm_feature_extraction.ipynb`, pyosmium extraction clipped to the UAE boundary |
| `building_footprint_pct`, `building_count_per_km2`, `poi_count_per_km2`, `road_density_km_per_km2` | Density ratios | Derived from the raw OSM counts above ÷ `zone_area_km2` |
| `peer_group` | One of 4 labels: `commercial/urban-core`, `low-density residential`, `industrial`, `rural/edge` | KMeans (k=4) on log-transformed, standardized population/building/POI/road density (`09_peer_group_classifier.ipynb`) — never OSM land-use tag |
| `emirate` | One of the 7 UAE emirates | H3 cell centroid vs. real OSM administrative boundary polygons, `scripts/build_emirate_field.py` |

### `zone_priority.parquet` — the final table; dashboard and copilot both read only this file

Everything in `zone_quarter_table.parquet`, plus every computed score, in the order the
pipeline (`src/run_pipeline.py`) adds them:

| Field | Meaning | Computed by |
|---|---|---|
| `latency_effective_ms` | `latency_loaded_ms`, falling back to `latency_ms` | `compute_scores.add_effective_latency` |
| `quarters_observed` | How many of the 8 quarters this zone has any measurement in | `compute_scores.add_quarters_observed` |
| `experience_index` | 0–100 deterministic score: 50% download + 20% upload + 30% inverse latency, each min-max normalized (1st/99th percentile clipped) | `compute_scores.experience_index`. **NaN if `insufficient_evidence`.** |
| `confidence_score` | 0–100: 50% log-scaled test volume + 30% devices/tests ratio + 20% quarters-observed fraction | `compute_scores.confidence_score` |
| `insufficient_evidence` | `True` if `tests < MIN_TESTS_FOR_RELIABLE_EVIDENCE` in `compute_scores.py` (currently `1`, not the `30` used when these docs were first written — see `docs/validation_summary.md`) | The single authoritative eligibility gate — every score/flag below is only computed for `insufficient_evidence == False` rows |
| `peer_group_median_experience` | Median Experience Index of this zone's `(peer_group, quarter)`, computed from classified zones only | `compute_scores.peer_gap` |
| `peer_gap`, `peer_gap_pct` | `experience_index` − peer median, in points and % | `compute_scores.peer_gap` |
| `deteriorating` | `True` if this quarter closes 3 consecutive quarter-over-quarter declines in `peer_gap` | `trends.add_trend`. **Scoped per quarter** — not "has this zone ever declined," see `docs/validation_summary.md` |
| `trend_pts_per_qtr` | Least-squares slope of `experience_index` over the last 4 quarters | `trends.add_trend` |
| `peer_gap_ml_score`, `peer_gap_ml_anomaly` | Isolation Forest #1: how unusual vs. peer-group z-scores of download/upload/latency, this quarter | `anomaly_detection.add_peer_gap_ml` |
| `temporal_anomaly_ml_score`, `temporal_anomaly_ml_flag` | Isolation Forest #2: how unusual vs. this zone's own history (QoQ change + volatility) | `anomaly_detection.add_temporal_anomaly_ml` |
| `baseline_bottom_decile_flag` | Deterministic baseline: bottom-decile `download_mbps` within `(peer_group, quarter)` | `anomaly_detection.add_baseline_bottom_decile` — compared against the two ML flags in T2, see `docs/validation_summary.md` |
| `priority_score` | 0–100: weighted sum of 4 factors below, **multiplied** by `confidence_score/100` | `priority.add_priority` |
| `priority_zone` | `True` if in the top 10% of `priority_score` for that quarter | `priority.add_priority` |
| `factor_peer_gap`, `factor_ml_anomaly`, `factor_deterioration`, `factor_population` | The 4 inputs to `priority_score`, each rescaled 0–100 within their own quarter | `priority.add_priority`. Weights: 35% / 20% / 20% / 25% |
| `peer_gap_band`, `ml_anomaly_band`, `deterioration_band`, `population_band` | Low/Medium/High tercile bucket of each factor above, for the "why this priority" display | `priority._factor_band` |

### `zone_scores.parquet` — Phase 2 checkpoint, same schema as `zone_quarter_table.parquet` plus `experience_index`/`confidence_score`/`insufficient_evidence`/`peer_gap`/`peer_gap_pct`

Written by `notebooks/10_scores_and_peer_gap.ipynb`, which runs the exact same
`compute_scores.score_zone_quarters()` call `run_pipeline.py` does, then adds three checks not
duplicated elsewhere: a classification-rate sanity check, a real-data Confidence sanity check
(finds an actual well-measured vs. sparse zone-quarter), and an **Experience Index** weight-
sensitivity check (Spearman + top-20 overlap under ±15% perturbation — distinct from
`tests/test_t5_priority_sensitivity.py`, which perturbs the *Priority* weights, not Experience's).
**Not read by any downstream code** — `run_pipeline.py` recomputes this step itself directly
from `zone_quarter_table.parquet` rather than reading this file. Verified consistent with
`zone_priority.parquet` on every shared column (0 mismatches across a 150-value spot check). Re-run
the notebook if `zone_quarter_table.parquet` or the scoring formulas change and you want this
checkpoint refreshed too — it won't happen automatically the way `zone_priority.parquet` does
via `run_pipeline.py`.

### `emirate_zones_uae.parquet` — one row per H3 cell (measured + populated union), 1,869 rows

`h3_cell` → `emirate`. Covers every H3 cell Ookla ever measured *and* every cell WorldPop
records population in (not just measured ones) — needed so per-emirate population
denominators are complete, not just the measured subset. See `scripts/build_emirate_field.py`.

### `peer_groups_uae.parquet`, `population_zones_uae.parquet`, `osm_density_zones_uae.parquet`

Standalone versions of the same static (non-quarterly) columns already folded into
`zone_quarter_table.parquet` above — kept separately so each notebook's own output can be
inspected without re-deriving it from the joined table.
