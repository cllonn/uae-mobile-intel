# UAE Mobile Network Experience Intelligence

An AI-powered geographic intelligence platform for analyzing publicly measured mobile network
experience across the UAE, built from public Ookla Speedtest, WorldPop, and OpenStreetMap data.
**This is not e& network data** — the public dataset carries no operator attribution, and the
system says so explicitly wherever it's relevant.

## Status

- **Phase 1 — Foundation: Complete.** Data acquisition, UAE boundary clipping, H3 resolution
  choice, real emirate field.
- **Phase 2 — Scores: Complete.** Experience Index, Confidence Score, peer groups, Peer Gap.
- **Phase 3 — Intelligence: Complete, validated through T0–T5.** Trend/deterioration, ML
  anomaly detection with an honest baseline comparison, Priority Engine.
- **Phase 4 — Dashboard: Working real-data prototype.** All 8 quarters, all 7 emirates, 4
  layers, drill-down — every number traced to `zone_priority.parquet` (T1).
- **GenAI / Grounded Copilot: Next phase, pending mentor confirmation.** `src/copilot_tools.py`
  and `src/copilot.py` exist as **preparatory/prototype routing code** — a deterministic tool
  layer and a keyword-based question router with a template-only narrator (no LLM connected
  yet). This is scaffolding for the next phase, not a completed GenAI system; see "Grounded
  copilot (prototype)" below for exactly what does and doesn't exist yet.

## Setup

```
pip install -r requirements.txt
```

Tested with Python 3.13. Open any notebook in `notebooks/` (VS Code's Jupyter extension or
JupyterLab both work) and run all cells top to bottom — each collection notebook downloads its
own raw data on first run, so no manual download step is needed.

## Data sources

| Dataset | Notebook | Notes |
|---|---|---|
| Ookla Speedtest Open Data (mobile) | [`01_ookla_collection.ipynb`](notebooks/01_ookla_collection.ipynb) → [`05_zone_aggregation.ipynb`](notebooks/05_zone_aggregation.ipynb) | 8 quarters, 2024 Q3 → 2026 Q2. CC BY-NC-SA 4.0 (non-commercial). |
| WorldPop UAE population | [`02_worldpop_collection.ipynb`](notebooks/02_worldpop_collection.ipynb) → [`05_zone_aggregation.ipynb`](notebooks/05_zone_aggregation.ipynb) | R2025A constrained, 2026, 100m. National total 11,476,873, conserved exactly through H3 aggregation. CC BY 4.0. |
| OpenStreetMap | [`03_osm_collection.ipynb`](notebooks/03_osm_collection.ipynb) → [`07_osm_feature_extraction.ipynb`](notebooks/07_osm_feature_extraction.ipynb) | Geofabrik GCC States PBF: buildings/roads/POIs for peer grouping, and the 7 emirate administrative boundaries. ODbL. |

Full field-by-field detail is in [`docs/data_dictionary.md`](docs/data_dictionary.md).

Raw files live under `data/raw/<dataset>/` and are **gitignored on purpose** (the 8 Ookla
quarters alone total ~1.5GB, over GitHub's push limit). Every collection notebook downloads its
own raw inputs if they're not already present, so running a notebook end to end reproduces
`data/raw/` from nothing. Processed outputs (small) live under `data/processed/` and **are**
committed.

**Geographic unit:** H3 resolution 6 (project decision — see
[`04_h3_resolution_choice.ipynb`](notebooks/04_h3_resolution_choice.ipynb) for the resolution
6/7/8 comparison and the tradeoffs this decision accepts). Persisted at
`data/processed/h3_resolution.json`.

## Pipeline architecture

```
Ookla + WorldPop + OSM
          |
       H3 Zones  ---- + emirate, peer_group (static, joined once)
          |
Experience + Confidence   (src/compute_scores.py, deterministic)
          |
Peer Gap + Trend + ML anomaly   (src/trends.py, src/anomaly_detection.py)
          |
   Priority Engine   (src/priority.py, deterministic + confidence guardrail)
          |
 zone_priority.parquet   <- the one file the dashboard and copilot both read
       /        \
 Dashboard    src/copilot_tools.py   (10 deterministic query functions, no scoring)
 (build_          |
  dashboard.py) src/copilot.py  --  question -> tool routing -> [LLM narrates | template fallback]
                   |
            Grounded answer / zone brief
```

- `notebooks/` — data collection, exploration, and one-time classification work (Ookla,
  WorldPop, OSM, H3 zone aggregation, peer-group classification). Self-contained; each
  reproduces its own raw inputs on first run.
- `src/` — deterministic Python analytics + the copilot, meant to be imported, not read
  top-to-bottom as a notebook.
- `scripts/` — one-off/reusable utilities (emirate boundary extraction, the canonical-questions
  demo).
- `tests/` — reproducible validation scripts (T1/T3/T4/T5).
- `docs/` — data dictionary, validation summary, T7 usability protocol.

## The evidence threshold (read this first)

**A zone-quarter needs `tests >= MIN_TESTS_FOR_RELIABLE_EVIDENCE` to be classified at all** — the
single authoritative gate in `src/compute_scores.py` every other module reads
(`insufficient_evidence`), never re-derives. Below that bar, a zone gets **no Experience Index,
no Priority Score, nothing** — the honest output is "insufficient public evidence," never a low
score. Confidence Score is a separate concept: it still differentiates strength of evidence
*among* zones that clear this bar — it doesn't decide whether a zone is classified.

> **Current code state:** `MIN_TESTS_FOR_RELIABLE_EVIDENCE = 1` in `src/compute_scores.py` as of
> this revision (not the `30` this section previously described). That value was already changed,
> uncommitted, in the working tree before the H3 resolution 6 migration below — it was
> deliberately left as-is during that migration (out of scope for a resolution change) rather
> than reverted. **If `30` was the intended value, this needs a separate, explicit fix** — see
> `docs/validation_summary.md` for the numbers at both thresholds.

2026Q2 headline numbers at H3 resolution 6, under the *current* `tests >= 1` rule: 671 zones
measured, **671 classified (100%)**, 93.5% of national population represented, 68 zones flagged
for investigation. Across all 8 quarters: 5,011 of 5,011 zone-quarters classified (0%
insufficient evidence).

For reference, the *same resolution-6 data* re-scored at the historical `tests >= 30` bar gives
155 classified (23.1%), 67.1% population represented, 16 priority zones — the number that's
directly comparable to the old resolution-7 baseline below, since it isolates the resolution
change from the threshold change.

**Resolution 7 → 6, holding the evidence threshold fixed at `>= 30`** (isolates the effect of
the H3 resolution change alone): 1,815 → 671 zones measured, 314 → 155 classified (17.3% → 23.1%
of measured zones), 37.6% → 67.1% of national population represented, 32 → 16 priority zones
flagged. Fewer, larger zones pool more tests each, so a bigger *share* clears the bar even though
the *count* of zones nationally drops.

## Deterministic scores

**Experience Index** (`src/compute_scores.py`) — 0–100: 50% download + 20% upload + 30% inverse
latency, each min-max normalized. **Confidence Score** — 0–100: 50% log-scaled test volume +
30% devices/tests ratio + 20% quarters-observed fraction. **Peer Gap** — a zone's Experience
Index minus its peer group's median, in the *same quarter*, computed from classified zones only.
Peer groups (`09_peer_group_classifier.ipynb`, KMeans k=4 on density features, never OSM
land-use tag): commercial/urban-core, low-density residential, industrial, rural/edge. 2026Q2
peer medians (H3 resolution 6): commercial/urban-core 47.8, low-density residential 45.0,
industrial 40.4, rural/edge 34.4 — real, meaningful spread.

## Trend, ML anomaly detection, and Priority

- **Trend** (`src/trends.py`) — deterioration relative to peer-group trend (tracks `peer_gap`
  quarter to quarter, never raw Mbps), flagged only after 3 consecutive declining quarters. 84
  unique zones hit this pattern at some point across the 8-quarter window; 22 are currently
  deteriorating as of 2026Q2 (H3 resolution 6, current `tests >= 1` threshold — see "The evidence
  threshold" above).
- **ML anomaly detection** (`src/anomaly_detection.py`) — two Isolation Forest models (Peer Gap:
  vs. peer-group z-scores this quarter; Temporal Anomaly: vs. the zone's own history), each
  compared against a deterministic bottom-decile-download baseline. **T2 result, reported
  honestly**: on a blind synthetic-anomaly hold-out (`tests/test_t2_anomaly_benchmark.py`,
  reproducible), the simple baseline outperformed both ML detectors on every metric
  (precision/recall/F1/FPR) — see `docs/validation_summary.md` for the full numbers, including
  a per-fault-type breakdown showing the ML almost never catches a pure download-only fault.
  Kept as supplementary evidence, not presented as superior.
- **Priority** (`src/priority.py`) — `Priority = 0.35·PeerGap + 0.20·MLAnomaly +
  0.20·Deterioration + 0.25·Population`, **multiplied** by `confidence/100` (not added as a
  fifth factor), so a low-confidence zone structurally cannot reach a high Priority Score. Top
  10% per quarter flagged — 68 of 671 in 2026Q2. Sensitivity-tested (T5): ±15% weight
  perturbation, worst-case Spearman ρ=0.998 — not fragile.
- `src/run_pipeline.py` chains all of the above into `data/processed/zone_priority.parquet`.
  Re-run whenever an upstream input changes: `python -m src.run_pipeline`.

## Emirate field

`scripts/build_emirate_field.py` extracts the 7 emirates' real administrative boundaries
(`admin_level=4`, matched by OSM `ISO3166-2` tag) from the GCC PBF already in the repo, and
assigns every H3 zone to one emirate via centroid-in-polygon join. One `emirate` column, added
once to `zone_quarter_table.parquet`, flows through `run_pipeline.py` into
`zone_priority.parquet` — the dashboard, T0 reporting, and the copilot all read the same field,
never separate logic. Re-run with `python -m scripts.build_emirate_field` if the boundary or
zone set ever changes (caches the extracted boundary GeoJSON; only re-extracts from the 252MB
PBF if that cache is deleted).

## Dashboard

`src/build_dashboard.py` generates the interactive UAE map as one self-contained HTML file —
`data/processed/uae_dashboard.html` — plain SVG + vanilla JS, no external libraries, works
offline. All 8 quarters and all 7 emirates are browsable via dropdowns; switching either
re-filters the map, the top-priority list, and the KPI strip live. Four switchable layers
(Experience / Priority / Trend / Confidence). Clicking an unclassified (grey) hexagon shows
"insufficient public evidence," never a fabricated score.

Run: `python -m src.build_dashboard` (after `run_pipeline.py`).

**Known limitations, stated plainly:**
- No invented place names — zones show as `H3 <id>` plus the real `emirate` field.
- Priority Score is a national ranking; filtering to one emirate shows which nationally-ranked
  zones fall there, not a separate local re-ranking.
- The ML anomaly detector currently underperforms the deterministic baseline (T2, reported
  honestly, not hidden).

## Grounded copilot (prototype — pending mentor confirmation before Phase 5 is called "done")

Preparatory work for the GenAI phase, not a finished GenAI system: no LLM is connected yet.
`src/copilot_tools.py` — 10 deterministic functions (`get_top_priority_zones`,
`get_zone_trend`, `get_coverage_summary`, etc.) that read only `zone_priority.parquet` and
return plain dicts. No score is ever calculated in this layer. `src/copilot.py` routes a
question to one of these functions by keyword matching (not an LLM), then narrates the result —
today that narration is a deterministic string template only; wiring in a real LLM (via
`ANTHROPIC_API_KEY`, `pip install anthropic`) is the next-phase work, not yet done. A
keyword-based guardrail (`copilot.check_refusal`) intercepts operator-attribution questions
("which e& site...") **before** any tool is chosen. `copilot.generate_zone_brief(zone_id)`
assembles the same evidence a real AI zone brief would need; every number in it traces back to
`copilot_tools.get_zone_details`.

Run all 14 test questions end-to-end (tool selected, tool output, final template answer):
`python -m scripts.demo_canonical_questions`

## Testing & validation

`tests/` holds reproducible T1/T2/T3/T4/T5 scripts — all passing on the current
(30-test-threshold) pipeline output, including the honest T2 ML-vs-baseline result.
`docs/validation_summary.md` has the full T0–T7 status and how to run them.
`docs/t7_usability_protocol.md` is a real, un-run live-tester protocol, ready to execute.
`docs/data_dictionary.md` documents every field across every processed table.
`docs/mentor_methodology_summary.md` is a short, plain-language walkthrough for the mentor demo.

## Next steps

1. Run the T7 live usability protocol with a real outside tester (`docs/t7_usability_protocol.md`).
2. Get mentor sign-off on the analytical methodology (Section "Status" above) before starting
   GenAI in earnest.
3. Connect a real LLM (`ANTHROPIC_API_KEY` + `pip install anthropic`) and score the T6
   51-question copilot bank.
4. Business case, architecture diagram for future internal-data integration, final presentation.
