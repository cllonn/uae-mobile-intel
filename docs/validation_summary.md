# T0–T7 validation summary

**Authoritative state**: `MIN_TESTS_FOR_RELIABLE_EVIDENCE = 30` in `src/compute_scores.py`. All
numbers on this page are from the pipeline as it stands today. Every T1/T2/T3/T4/T5 script is
in `tests/` and reproduces its own numbers on demand — see "Running the tests" below.

## Evidence threshold correction (context for every number below)

The dashboard was found to disagree visually with an earlier reference map for the same
quarter. Root cause: the reference map used `tests >= 30` as its evidence bar; the scoring
pipeline used `tests >= 5`. Fixed by correcting the one authoritative constant in
`src/compute_scores.py`; every downstream module already read the `insufficient_evidence` flag
this constant produces rather than re-deriving its own threshold, so nothing else needed to
change. One real bug surfaced by the stricter threshold (4 zones landing alone in their peer
group, blocking their Priority Score entirely) was fixed in `src/priority.py` — a missing ML
signal now defaults to "no anomaly evidence" rather than nulling the other three factors.

**2026Q2, before vs. after:**

| | tests≥5 (superseded) | tests≥30 (current) |
|---|---|---|
| Classified zones | 868 | **314** |
| Population represented | 65.5% | **37.6%** |
| Priority zones flagged | 89 | **32** |

## Status by test

| Test | What it checks | Status | Result |
|---|---|---|---|
| T0 | Coverage & representativeness | **Passed, reproducible** | 2026Q2: 1,815 measured zones, **314 classified (17.3%)**, 37.6% of national population represented. By emirate (real polygon join): Dubai 58.8% pop. represented, Ajman 63.2%, Sharjah 30.8%, Abu Dhabi 25.3%, Umm Al Quwain 18.3%, Fujairah 9.7%, Ras Al Khaimah 9.0%. By peer group: commercial/urban-core 76.4% coverage vs. industrial 1.3% and rural/edge 1.6% — dense, well-tested areas clear the bar; sparse ones honestly don't. All from `copilot_tools.get_coverage_summary()`, live against the real `emirate` field. |
| T1 | Pipeline correctness | **Passed, reproducible** | `tests/test_t1_pipeline.py`: 22 zones, **220 checks, 0 mismatches** — raw metrics survive scoring unchanged, dashboard JSON matches `zone_priority.parquet` exactly. |
| T2 | Synthetic anomaly benchmark | **Passed (script now exists), honest negative result** | `tests/test_t2_anomaly_benchmark.py`: 314 classified zones split 80/20 (251 train / 63 hold-out), synthetic faults injected into the hold-out only, reference statistics computed from train only, Isolation Forest unchanged from production. **Peer Gap ML**: precision 0.639, recall 0.263, F1 0.372, FPR 0.079. **Baseline (bottom-decile download)**: precision 0.697, recall 0.354, F1 0.470, FPR 0.082 — baseline wins on every metric, again. **New this run — per-fault-type recall**: download-drop-only 0.016 (essentially never caught), latency-spike-only 0.232, combined 0.582 — confirms the earlier hypothesis that the ML's right-skewed, un-transformed download z-score buries a pure download-only signal. **Temporal Anomaly ML** (gradual 3-quarter decline, no baseline exists for this shape): precision 0.25, recall 0.059, F1 0.095. Not tuned to these or any other numbers — single fixed seed, run once. |
| T3 | Trend correctness | **Passed, reproducible** | `tests/test_t3_trends.py`: 20 zones, independently reimplemented (never imports `src/trends.py`), **20/20 deteriorating-flag agreement, 20/20 trend-slope agreement**. |
| T4 | Confidence behaviour | **Passed, reproducible** | `tests/test_t4_confidence.py`: constructed 500-tests/100-devices vs. 2-tests/1-device case (identical network conditions) — busy scored (58.5), sparse labeled insufficient evidence (26.3, not classified). All **575/575** real 2-test/1-device zones nationally confirmed insufficient — unaffected by the threshold correction, since 2 < 30 either way. |
| T5 | Priority sensitivity | **Passed, reproducible** | `tests/test_t5_priority_sensitivity.py`, on the corrected 314-zone population: ±15% perturbation on all 4 weights, **worst-case Spearman ρ=0.996, worst-case top-20 overlap 0.80**. Ranking is not fragile. |
| T6 | Copilot evaluation | Prep done, scoring pending | 51-question bank prepared, unaffected by the threshold change. `src/copilot.py` exists; all 14 test questions route and answer correctly (`python -m scripts.demo_canonical_questions`). Full 51-question scoring needs a connected LLM. |
| T7 | Zone-brief audit & usability | Protocol ready, live run pending | See `docs/t7_usability_protocol.md` — a real, un-run live-tester protocol. The dashboard's emirate filter is the structural fix an earlier review called for. |

## ML decision (T2) — unchanged conclusion, now reproducible

Isolation Forest kept exactly as-is; the deterministic bottom-decile baseline still outperforms
it, on two independent runs (the original externally-reported one, and this repo's own
reproducible script) — reported honestly both times, not hidden. `src/priority.py` already
weights ML anomaly as supplementary (20%) rather than primary. The per-fault-type breakdown
newly available from `test_t2_anomaly_benchmark.py` explains *why*: the ML's feature space
(raw, un-transformed peer-group z-scores of a right-skewed metric) is much weaker on a pure
download-only fault than a combined one. No model changes were made to fix this — that's a
future, separately-evaluated experiment, not tonight's scope.

## What "deteriorating" means (T3 disambiguation)

`trends.py`'s `deteriorating` column is scoped **per quarter**: True means *that specific
quarter* closed a 3-consecutive-quarter decline vs. peers, not "has this zone ever declined."
`copilot_tools.get_deteriorating_zones()` makes this explicit with a `scope` parameter
(`"current"` vs. `"ever_in_window"`).

## Running the tests

```
python -m tests.test_t1_pipeline
python -m tests.test_t2_anomaly_benchmark
python -m tests.test_t3_trends
python -m tests.test_t4_confidence
python -m tests.test_t5_priority_sensitivity
```
No pytest required — each runs standalone. Also `pytest`-discoverable if you have it installed.

| Test file | Independent of production code? |
|---|---|
| `test_t1_pipeline.py` | Compares files directly (raw vs. scored vs. dashboard JSON) — no production logic re-run |
| `test_t2_anomaly_benchmark.py` | Reuses `src/anomaly_detection.py`'s constants (contamination, random state) for a fair comparison, but the train/hold-out split, injection, and evaluation are all written independently here |
| `test_t3_trends.py` | Yes — reimplements the deteriorating-flag and slope logic from scratch, never imports `src/trends.py` |
| `test_t4_confidence.py` | No — calls the real `src/compute_scores.py::confidence_score`; a behavior check, not an independent-recomputation check |
| `test_t5_priority_sensitivity.py` | No — calls the real `src/priority.py::add_priority` with perturbed weights |

T0 has no script (see its row above for the live tool call that produces it). T6 and T7 need
the copilot/live tester and can't run as unattended scripts.
