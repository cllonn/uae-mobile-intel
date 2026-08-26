"""
Deterministic tool layer between the LLM copilot (src/copilot.py) and the computed analytics.

Every function here is a pure read against `zone_priority.parquet` (or the hardcoded scoring
CONFIG already defined in compute_scores.py / trends.py / anomaly_detection.py / priority.py,
for `get_methodology`) -- nothing is recalculated, nothing is invented. This is the boundary
the brief requires: the LLM may only call these functions and narrate what they return; it
must never compute an Experience Index, Confidence Score, Peer Gap, trend, anomaly score,
population exposure, or Priority Score itself.

Every function returns a plain dict/list-of-dicts (JSON-serializable, no numpy/pandas types),
never prose -- so a test can assert on the actual numbers, and an LLM prompt can drop the
result straight into context.
"""
from pathlib import Path

import pandas as pd

from src.compute_scores import CONFIDENCE_WEIGHTS, EXPERIENCE_WEIGHTS, MIN_TESTS_FOR_RELIABLE_EVIDENCE, TESTS_SATURATION
from src.priority import PRIORITY_WEIGHTS, PRIORITY_ZONE_FRACTION
from src.trends import CONSECUTIVE_DECLINES_REQUIRED

ZONE_PRIORITY_PATH = Path("data/processed/zone_priority.parquet")

_df_cache: pd.DataFrame | None = None


def _load() -> pd.DataFrame:
    global _df_cache
    if _df_cache is None:
        _df_cache = pd.read_parquet(ZONE_PRIORITY_PATH)
    return _df_cache


def _latest_quarter() -> str:
    return sorted(_load()["quarter"].unique())[-1]


def _num(v):
    return None if pd.isna(v) else (bool(v) if isinstance(v, (bool,)) else round(float(v), 2))


def _classified(quarter: str | None = None, emirate: str | None = None):
    """Shared filter every listing tool uses: one quarter (defaults to latest), classified
    zones only (insufficient-evidence zones are never eligible for a ranking), optionally
    one emirate. Returns (dataframe, resolved_quarter)."""
    df = _load()
    quarter = quarter or _latest_quarter()
    sub = df[(df["quarter"] == quarter) & (~df["insufficient_evidence"])]
    if emirate:
        sub = sub[sub["emirate"] == emirate]
    return sub, quarter


def _summary_row(row: pd.Series) -> dict:
    """The compact per-zone shape used by every listing tool."""
    return {
        "zone_id": row["h3_cell"],
        "quarter": row["quarter"],
        "emirate": row["emirate"],
        "experience_index": _num(row["experience_index"]),
        "confidence_score": _num(row["confidence_score"]),
        "peer_group": row["peer_group"],
        "peer_gap": _num(row["peer_gap"]),
        "population": int(row["population"]),
        "priority_score": _num(row["priority_score"]),
        "priority_zone": bool(row["priority_zone"]),
        "evidence_status": "scored",
    }


# ---------------------------------------------------------------------------
# Zone-scoped tools
# ---------------------------------------------------------------------------

def get_zone_details(zone_id: str, quarter: str | None = None) -> dict:
    """Everything known about one zone in one quarter. If the zone has no row for that
    quarter, or has too few tests to classify, says so explicitly rather than guessing."""
    df = _load()
    quarter = quarter or _latest_quarter()
    match = df[(df["h3_cell"] == zone_id) & (df["quarter"] == quarter)]
    if match.empty:
        return {"zone_id": zone_id, "quarter": quarter, "error": "no_data_for_zone_and_quarter"}
    row = match.iloc[0]
    if bool(row["insufficient_evidence"]):
        return {
            "zone_id": zone_id, "quarter": quarter, "evidence_status": "insufficient_evidence",
            "tests": int(row["tests"]), "devices": int(row["devices"]),
            "message": "Too few public Ookla measurements this quarter to compute a reliable "
                       "score. This is not the same as poor performance -- it means there is "
                       "not enough public evidence to assess this zone.",
        }
    return {
        "zone_id": zone_id, "quarter": quarter, "emirate": row["emirate"],
        "evidence_status": "scored",
        "experience_index": _num(row["experience_index"]),
        "confidence_score": _num(row["confidence_score"]),
        "peer_group": row["peer_group"],
        "download_mbps": _num(row["download_mbps"]),
        "upload_mbps": _num(row["upload_mbps"]),
        "latency_ms": _num(row["latency_effective_ms"]),
        "tests": int(row["tests"]), "devices": int(row["devices"]),
        "quarters_observed": int(row["quarters_observed"]),
        "population": int(row["population"]),
        "peer_gap": _num(row["peer_gap"]), "peer_gap_pct": _num(row["peer_gap_pct"]),
        "peer_group_median_experience": _num(row["peer_group_median_experience"]),
        "deteriorating": bool(row["deteriorating"]),
        "trend_pts_per_qtr": _num(row["trend_pts_per_qtr"]),
        "peer_gap_ml_anomaly": bool(row["peer_gap_ml_anomaly"]),
        "temporal_anomaly_ml_flag": bool(row["temporal_anomaly_ml_flag"]),
        "priority_score": _num(row["priority_score"]),
        "priority_zone": bool(row["priority_zone"]),
        "priority_factors": {
            "peer_gap": str(row["peer_gap_band"]),
            "ml_anomaly": str(row["ml_anomaly_band"]),
            "deterioration": str(row["deterioration_band"]),
            "population": str(row["population_band"]),
        },
    }


def get_zone_peer_comparison(zone_id: str, quarter: str | None = None) -> dict:
    """How this zone compares to the peer group it was actually compared against -- never
    the UAE-wide distribution, per the brief's mandatory peer-group rule."""
    detail = get_zone_details(zone_id, quarter)
    if detail.get("evidence_status") != "scored":
        return detail
    df = _load()
    peer_rows = df[
        (df["quarter"] == detail["quarter"]) & (df["peer_group"] == detail["peer_group"])
        & (~df["insufficient_evidence"])
    ]
    return {
        "zone_id": zone_id, "quarter": detail["quarter"],
        "peer_group": detail["peer_group"], "peer_group_size": int(len(peer_rows)),
        "experience_index": detail["experience_index"],
        "peer_group_median_experience": detail["peer_group_median_experience"],
        "peer_gap": detail["peer_gap"], "peer_gap_pct": detail["peer_gap_pct"],
    }


def get_zone_trend(zone_id: str) -> dict:
    """The zone's full 8-quarter history -- what changed, and when. `deteriorating` is
    scoped per quarter (True means *that* quarter closed a 3+ consecutive-quarter decline
    vs. peers, not "this zone has ever declined") -- callers asking "is it deteriorating
    now" should read the last entry, not `any()` across the list."""
    df = _load()
    rows = df[df["h3_cell"] == zone_id].sort_values("quarter")
    if rows.empty:
        return {"zone_id": zone_id, "error": "unknown_zone_id"}
    history = []
    for _, row in rows.iterrows():
        if bool(row["insufficient_evidence"]):
            history.append({"quarter": row["quarter"], "evidence_status": "insufficient_evidence"})
        else:
            history.append({
                "quarter": row["quarter"], "evidence_status": "scored",
                "experience_index": _num(row["experience_index"]),
                "peer_gap": _num(row["peer_gap"]),
                "deteriorating": bool(row["deteriorating"]),
            })
    latest = rows.iloc[-1]
    return {
        "zone_id": zone_id,
        "quarters_observed": int(latest["quarters_observed"]),
        "trend_pts_per_qtr": _num(latest["trend_pts_per_qtr"]),
        "currently_deteriorating": bool(latest["deteriorating"]) if not latest["insufficient_evidence"] else None,
        "ever_deteriorated_in_window": bool(rows["deteriorating"].any()),
        "history": history,
    }


# ---------------------------------------------------------------------------
# Listing tools
# ---------------------------------------------------------------------------

def get_top_priority_zones(n: int = 5, quarter: str | None = None, emirate: str | None = None) -> list[dict]:
    """Which N zones should be investigated first -- ranked by Priority Score, which already
    bakes in the confidence guardrail (low-confidence zones can't score high here)."""
    sub, quarter = _classified(quarter, emirate)
    ranked = sub.sort_values("priority_score", ascending=False).head(n)
    return [_summary_row(r) for _, r in ranked.iterrows()]


def get_weakest_zones(n: int = 10, quarter: str | None = None, emirate: str | None = None) -> list[dict]:
    """Lowest Experience Index among classified zones -- "where does experience appear
    weakest," independent of population or priority ranking."""
    sub, quarter = _classified(quarter, emirate)
    ranked = sub.sort_values("experience_index", ascending=True).head(n)
    return [_summary_row(r) for _, r in ranked.iterrows()]


def get_deteriorating_zones(quarter: str | None = None, emirate: str | None = None,
                             scope: str = "current") -> list[dict]:
    """`scope="current"` (default): zones whose *this-quarter* flag closed a 3+ quarter
    decline -- "which areas are deteriorating right now." `scope="ever_in_window"`: any zone
    that hit this pattern at least once across all 8 quarters -- a broader, historical count.
    These are genuinely different questions; keeping them as separate, named parameters
    (rather than one ambiguous "deteriorating zones" list) is deliberate."""
    if scope == "ever_in_window":
        df = _load()
        sub = df[~df["insufficient_evidence"]]
        if emirate:
            sub = sub[sub["emirate"] == emirate]
        ever = sub.loc[sub["deteriorating"], "h3_cell"].unique().tolist()
        # Only zones still classified in the latest quarter get a summary row (priority_score
        # etc. are NA for a zone that has since dropped below the evidence threshold) -- a
        # zone that deteriorated earlier but is now insufficient-evidence is real, but isn't
        # something this listing can show a current score for.
        latest, _ = _classified(quarter=_latest_quarter(), emirate=emirate)
        latest_rows = latest[latest["h3_cell"].isin(ever)]
        return [_summary_row(r) for _, r in latest_rows.iterrows()]
    sub, quarter = _classified(quarter, emirate)
    flagged = sub[sub["deteriorating"]]
    return [_summary_row(r) for _, r in flagged.iterrows()]


def get_anomalous_zones(kind: str, quarter: str | None = None, emirate: str | None = None) -> list[dict]:
    """`kind="peer_gap"`: unusual vs. comparable peers, right now (Isolation Forest #1).
    `kind="temporal"`: unusual vs. its own history (Isolation Forest #2). These are the two
    analytically distinct ML outputs the brief requires -- never conflate them."""
    col = {"peer_gap": "peer_gap_ml_anomaly", "temporal": "temporal_anomaly_ml_flag"}.get(kind)
    if col is None:
        return [{"error": f"unknown kind '{kind}', expected 'peer_gap' or 'temporal'"}]
    sub, quarter = _classified(quarter, emirate)
    flagged = sub[sub[col]]
    return [_summary_row(r) for _, r in flagged.iterrows()]


def get_high_population_weak_zones(n: int = 10, quarter: str | None = None, emirate: str | None = None) -> list[dict]:
    """Where weak experience and high population exposure occur together: below the median
    Experience Index *and* above the median population, within the same quarter/emirate
    scope, ranked by population -- a simple, explainable intersection rather than a fifth
    composite score."""
    sub, quarter = _classified(quarter, emirate)
    if sub.empty:
        return []
    exp_median = sub["experience_index"].median()
    pop_median = sub["population"].median()
    candidates = sub[(sub["experience_index"] <= exp_median) & (sub["population"] >= pop_median)]
    ranked = candidates.sort_values("population", ascending=False).head(n)
    return [_summary_row(r) for _, r in ranked.iterrows()]


def get_coverage_summary(quarter: str | None = None, emirate: str | None = None) -> dict:
    """The T0-style honesty check: how much of the country/population is actually
    represented by zones with enough evidence to score, at the requested scope."""
    df = _load()
    quarter = quarter or _latest_quarter()
    q_df = df[df["quarter"] == quarter]
    if emirate:
        q_df = q_df[q_df["emirate"] == emirate]
    classified = q_df[~q_df["insufficient_evidence"]]

    pop_all_path = Path("data/processed/population_zones_uae.parquet")
    emirate_zones_path = Path("data/processed/emirate_zones_uae.parquet")
    pop_all = pd.read_parquet(pop_all_path)
    if emirate:
        emirate_zones = pd.read_parquet(emirate_zones_path)
        pop_all = pop_all.merge(emirate_zones, on="h3_cell", how="left")
        pop_all = pop_all[pop_all["emirate"] == emirate]
    population_universe = int(pop_all["population"].sum())
    population_represented = int(classified["population"].sum())

    return {
        "quarter": quarter, "emirate": emirate or "All UAE",
        "measured_zones": int(len(q_df)),
        "classified_zones": int(len(classified)),
        "population_universe": population_universe,
        "population_represented": population_represented,
        "population_represented_pct": round(100 * population_represented / population_universe, 1) if population_universe else None,
        "priority_zones": int(classified["priority_zone"].sum()),
    }


# ---------------------------------------------------------------------------
# Methodology
# ---------------------------------------------------------------------------

_METHODOLOGY = {
    "experience_index": {
        "formula": "Experience = w_download*D + w_upload*U + w_latency*L (each 0-1 normalized, "
                   "latency inverted so higher is always better), rescaled to 0-100.",
        "weights": EXPERIENCE_WEIGHTS,
        "note": "Deterministic. Computed once per (zone, quarter); not classified at all "
               "below the minimum-tests threshold.",
    },
    "confidence_score": {
        "formula": "Confidence = w_tests*log-scaled test volume + w_devices*(devices/tests) "
                   "+ w_quarters*(quarters observed / 8 total quarters).",
        "weights": CONFIDENCE_WEIGHTS,
        "min_tests_for_reliable_evidence": MIN_TESTS_FOR_RELIABLE_EVIDENCE,
        "tests_saturation_point": TESTS_SATURATION,
        "note": "Below the minimum test count, a zone gets no Experience Index at all -- "
               "'insufficient public evidence', not a low score.",
    },
    "peer_gap": {
        "formula": "Peer Gap = this zone's Experience Index - its peer group's median "
                   "Experience Index, in the same quarter. Peer group is a composite of "
                   "population/building/POI/road density (never OSM land-use tag, and "
                   "never the UAE-wide distribution).",
    },
    "trend": {
        "formula": f"A zone is flagged 'deteriorating' once its Peer Gap has declined for "
                   f"{CONSECUTIVE_DECLINES_REQUIRED} consecutive quarters -- relative to its "
                   f"peer group, never raw Mbps, and never on a single bad quarter.",
        "consecutive_declines_required": CONSECUTIVE_DECLINES_REQUIRED,
    },
    "priority_score": {
        "formula": "Priority = (w_peer_gap*PeerGapFactor + w_ml_anomaly*MLAnomalyFactor + "
                   "w_deterioration*DeteriorationFactor + w_population*PopulationFactor) "
                   "* (Confidence / 100).",
        "weights": PRIORITY_WEIGHTS,
        "priority_zone_fraction": PRIORITY_ZONE_FRACTION,
        "note": "Confidence is multiplied in, not added as a fifth factor -- a low-confidence "
               "zone cannot reach a high Priority Score however bad its other factors look. "
               "Sensitivity-tested (T5): +-15% weight perturbation, mean Spearman rho=0.998.",
    },
}


def get_methodology(metric: str) -> dict:
    """Returns the formula/weights/thresholds actually used for one named metric -- the
    same CONFIG values compute_scores.py/trends.py/priority.py import, not a paraphrase."""
    result = _METHODOLOGY.get(metric)
    if result is None:
        return {"error": f"unknown metric '{metric}'", "known_metrics": list(_METHODOLOGY)}
    return {"metric": metric, **result}
