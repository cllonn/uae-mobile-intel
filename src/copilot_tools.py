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

from src.compute_scores import (
    CONFIDENCE_WEIGHTS, EXPERIENCE_WEIGHTS, MIN_TESTS_FOR_RELIABLE_EVIDENCE,
    CONFIDENCE_TESTS_CAP, CONFIDENCE_DEVICES_CAP,
)
from src.priority import PRIORITY_WEIGHTS, PRIORITY_ZONE_FRACTION, CONFIDENCE_GATE_THRESHOLD
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


def _band(v):
    """Same null-safety as `_num` but for the Low/Medium/High band strings -- `str(nan)`
    produces the literal text `'nan'`, which is not a valid band and must never reach the
    narrator (LLM or template) as if it were one. Used for `priority_factors`, which is real
    for 'full' evidence-tier zones and null for 'low'/'insufficient' tier zones (no Priority
    Score was ever computed for them)."""
    return None if pd.isna(v) else str(v)


def _classified(quarter: str | None = None, emirate: str | None = None):
    """Shared filter every listing tool uses: one quarter (defaults to latest), any zone with
    an Experience Index -- 'low' or 'full' evidence tier (insufficient-evidence zones are
    never eligible for a listing at all), optionally one emirate. Returns (dataframe,
    resolved_quarter). NOT the right filter for anything Priority-related -- 'low' tier zones
    have no Priority Score at all; see `_full_tier` below for that."""
    df = _load()
    quarter = quarter or _latest_quarter()
    sub = df[(df["quarter"] == quarter) & (~df["insufficient_evidence"])]
    if emirate:
        sub = sub[sub["emirate"] == emirate]
    return sub, quarter


def _full_tier(quarter: str | None = None, emirate: str | None = None):
    """Shared filter for anything Priority-related: 'full' evidence tier only (tests>=30),
    the same population `src/priority.py` computes a Priority Score for. A 'low' tier zone
    (10-29 tests) has a real Experience Index but is explicitly excluded from the Priority
    shortlist per the brief -- 'unsafe as a recommendation' -- so it must never be returned by
    a tool whose job is ranking/flagging priority zones. Returns (dataframe, resolved_quarter)."""
    df = _load()
    quarter = quarter or _latest_quarter()
    sub = df[(df["quarter"] == quarter) & (df["evidence_tier"] == "full")]
    if emirate:
        sub = sub[sub["emirate"] == emirate]
    return sub, quarter


def _summary_row(row: pd.Series) -> dict:
    """The compact per-zone shape used by every listing tool. `evidence_tier` is always
    included -- 'low' or 'full' here, since these rows are always drawn from `_classified`/
    `_full_tier` which already exclude 'insufficient' -- so a narrator can tell a real-but-
    ineligible zone (priority_score null, priority_zone False) apart from a real-and-scored one
    without guessing from the null alone."""
    return {
        "zone_id": row["h3_cell"],
        "quarter": row["quarter"],
        "emirate": row["emirate"],
        "area_name": row["area_name"],
        "evidence_tier": row["evidence_tier"],
        "experience_index": _num(row["experience_index"]),
        "confidence_score": _num(row["confidence_score"]),
        "peer_group": row["peer_group"],
        "peer_gap": _num(row["peer_gap"]),
        "population": int(row["population"]),
        "priority_score": _num(row["priority_score"]),
        "priority_zone": bool(row["priority_zone"]),
        # Severity bands (2026-09-17) -- null for 'low'-tier rows (no Priority Score/factors
        # computed at all for that tier), never guessed. Exposed here, not just buried in
        # `get_zone_details`, so any listing tool's caller (narration or a test) can verify
        # *which* severity a returned zone actually has, rather than assuming a flag and a
        # band always coincide.
        "deteriorating": bool(row["deteriorating"]),
        "deterioration_band": _band(row["deterioration_band"]),
        "peer_gap_band": _band(row["peer_gap_band"]),
        "temporal_anomaly_band": _band(row["temporal_anomaly_band"]),
        "evidence_status": "scored",
    }


# ---------------------------------------------------------------------------
# Zone-scoped tools
# ---------------------------------------------------------------------------

def get_zone_details(zone_id: str, quarter: str | None = None) -> dict:
    """Everything known about one zone in one quarter. If the zone has no row for that
    quarter, or has too few tests to classify, says so explicitly rather than guessing.

    `evidence_tier` is always present when the zone has any evidence at all ('low' or 'full')
    -- 'low' tier zones (10-29 tests) get a real Experience Index/Confidence Score here but
    `priority_score`/`priority_zone`/`priority_factors` are `None`/`False`/all-`None`, never a
    fabricated value: `src/priority.py` never computes a Priority Score for that tier at all.
    A narrator must read `evidence_tier` before trying to explain a priority ranking."""
    df = _load()
    quarter = quarter or _latest_quarter()
    match = df[(df["h3_cell"] == zone_id) & (df["quarter"] == quarter)]
    if match.empty:
        return {"zone_id": zone_id, "quarter": quarter, "error": "no_data_for_zone_and_quarter"}
    row = match.iloc[0]
    if bool(row["insufficient_evidence"]):
        return {
            "zone_id": zone_id, "quarter": quarter, "evidence_status": "insufficient_evidence",
            "emirate": row["emirate"], "area_name": row["area_name"],
            "evidence_tier": row["evidence_tier"],
            "tests": int(row["tests"]), "devices": int(row["devices"]),
            "message": "Too few public Ookla measurements this quarter to compute a reliable "
                       "score. This is not the same as poor performance -- it means there is "
                       "not enough public evidence to assess this zone.",
        }
    return {
        "zone_id": zone_id, "quarter": quarter, "emirate": row["emirate"],
        "area_name": row["area_name"],
        "evidence_status": "scored",
        "evidence_tier": row["evidence_tier"],
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
        "priority_shortlist_eligible": row["evidence_tier"] == "full",
        "priority_factors": {
            "peer_gap": _band(row["peer_gap_band"]),
            "temporal_anomaly": _band(row["temporal_anomaly_band"]),
            "deterioration": _band(row["deterioration_band"]),
            "population": _band(row["population_band"]),
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
        "evidence_tier": detail["evidence_tier"],
        "peer_group": detail["peer_group"], "peer_group_size": int(len(peer_rows)),
        "experience_index": detail["experience_index"],
        "peer_group_median_experience": detail["peer_group_median_experience"],
        "peer_gap": detail["peer_gap"], "peer_gap_pct": detail["peer_gap_pct"],
    }


def get_zone_trend(zone_id: str) -> dict:
    """The zone's full 8-quarter history -- what changed, and when. `deteriorating` is
    scoped per quarter (True means *that* quarter closed a 3+ consecutive-quarter decline
    vs. peers, not "this zone has ever declined") -- callers asking "is it deteriorating
    now" should read the last entry, not `any()` across the list.

    Each scored history entry also carries the raw `download_mbps`/`upload_mbps`/`latency_ms`
    for that quarter (already on `zone_priority.parquet`, not recomputed) -- needed to answer
    "did download improve?" / "did latency deteriorate?" style questions, which are about one
    raw measurement's own quarter-over-quarter change, not the Experience Index trend. The
    top-level `*_change_qoq` fields are a plain subtraction between the two most recent SCORED
    quarters' raw values (None if fewer than 2 scored quarters exist) -- simple arithmetic on
    two already-computed numbers, not a new score."""
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
                "download_mbps": _num(row["download_mbps"]),
                "upload_mbps": _num(row["upload_mbps"]),
                "latency_ms": _num(row["latency_effective_ms"]),
            })
    latest = rows.iloc[-1]

    scored_history = [h for h in history if h["evidence_status"] == "scored"]
    changes = {"download_mbps_change_qoq": None, "upload_mbps_change_qoq": None, "latency_ms_change_qoq": None}
    if len(scored_history) >= 2:
        prev, cur = scored_history[-2], scored_history[-1]
        for field, key in (("download_mbps", "download_mbps_change_qoq"),
                           ("upload_mbps", "upload_mbps_change_qoq"),
                           ("latency_ms", "latency_ms_change_qoq")):
            if prev[field] is not None and cur[field] is not None:
                changes[key] = round(cur[field] - prev[field], 2)

    return {
        "zone_id": zone_id,
        "quarters_observed": int(latest["quarters_observed"]),
        "trend_pts_per_qtr": _num(latest["trend_pts_per_qtr"]),
        "currently_deteriorating": bool(latest["deteriorating"]) if not latest["insufficient_evidence"] else None,
        "ever_deteriorated_in_window": bool(rows["deteriorating"].any()),
        "history": history,
        **changes,
    }


# ---------------------------------------------------------------------------
# Listing tools
# ---------------------------------------------------------------------------

def get_top_priority_zones(n: int = 5, quarter: str | None = None, emirate: str | None = None) -> list[dict]:
    """Which N zones should be investigated first -- ranked by Priority Score, which already
    bakes in the confidence guardrail (low-confidence zones can't score high here). Explicitly
    'full' evidence tier only (`_full_tier`, not `_classified`) -- 'low' tier zones have no
    Priority Score at all (NaN, not a real number), so they must never be candidates for a
    ranking, however the sort happens to order NaNs."""
    sub, quarter = _full_tier(quarter, emirate)
    ranked = sub.sort_values("priority_score", ascending=False).head(n)
    return [_summary_row(r) for _, r in ranked.iterrows()]


def get_priority_zones(quarter: str | None = None, emirate: str | None = None) -> list[dict]:
    """Every zone actually flagged `priority_zone == True` this quarter -- the real,
    data-determined shortlist (currently the top ~10% of 'full' evidence-tier zones by
    Priority Score; see `src/priority.py::PRIORITY_ZONE_FRACTION`), not a caller-chosen top N.
    This is the set the mandatory "AI zone brief for every Priority zone" capability must
    iterate over -- `get_top_priority_zones(n=5)` is a different, narrower question ("which
    five first") and is the wrong tool for "brief every priority zone"."""
    sub, quarter = _full_tier(quarter, emirate)
    flagged = sub[sub["priority_zone"]].sort_values("priority_score", ascending=False)
    return [_summary_row(r) for _, r in flagged.iterrows()]


def _experience_ranked(ascending: bool, n: int, quarter: str | None, emirate: str | None) -> list[dict]:
    """Shared ranking behind get_weakest_zones/get_strongest_zones -- same classified-zone
    scope, same summary shape, only the sort direction differs. Kept as one function so the two
    public, differently-named tools (needed so the router can keep dispatching 'lowest' and
    'highest' Experience questions to stable, distinct tool names -- see copilot.py's
    _resolve_experience_query) can never drift apart in scope or filtering logic."""
    sub, quarter = _classified(quarter, emirate)
    ranked = sub.sort_values("experience_index", ascending=ascending).head(n)
    return [_summary_row(r) for _, r in ranked.iterrows()]


def get_weakest_zones(n: int = 10, quarter: str | None = None, emirate: str | None = None) -> list[dict]:
    """Lowest Experience Index among classified zones -- "where does experience appear
    weakest," independent of population or priority ranking."""
    return _experience_ranked(True, n, quarter, emirate)


def get_strongest_zones(n: int = 10, quarter: str | None = None, emirate: str | None = None) -> list[dict]:
    """Highest Experience Index among classified zones -- "where does experience appear
    strongest/highest/best," the mirror-image counterpart of get_weakest_zones. Independent of
    population or priority ranking, same as get_weakest_zones."""
    return _experience_ranked(False, n, quarter, emirate)


_VALID_SEVERITIES = ("low", "medium", "high")


def get_deteriorating_zones(quarter: str | None = None, emirate: str | None = None,
                             scope: str = "current", n: int | None = None,
                             severity: str | None = None) -> list[dict]:
    """`scope="current"` (default): zones whose *this-quarter* flag closed a persistent
    national-relative decline -- "which areas are deteriorating right now." `scope=
    "ever_in_window"`: any zone that hit this pattern at least once across all 8 quarters -- a
    broader, historical count. These are genuinely different questions; keeping them as
    separate, named parameters (rather than one ambiguous "deteriorating zones" list) is
    deliberate.

    `severity` ('low'/'medium'/'high', 2026-09-17), when given, narrows the result to zones
    whose `deterioration_band` matches exactly -- "areas with HIGH deterioration," a different
    question from "areas that ARE deteriorating" (this tool's own default, severity-blind
    behavior). Always applied on top of the `deteriorating` gate above, never instead of it --
    `deterioration_band` alone is NOT a safe filter by itself: a non-deteriorating zone (zero
    magnitude) also bands 'Low' by construction (see src/priority.py), so filtering on the band
    without the gate would silently pull in zones that were never flagged as deteriorating at
    all. Only meaningful with `scope="current"` (deterioration_band is a current-quarter
    Priority factor, not a historical-window concept).

    `n`, when given, narrows the (already-flagged, and already severity-filtered if `severity`
    was given) result to the N strongest current-quarter cases, ranked by `trend_pts_per_qtr`
    ascending (most negative = fastest ABSOLUTE decline). Kept as the existing, unchanged
    ranking field here -- `severity`/`deterioration_band` above is the new, correct way to ask
    for "how severe," so `n`'s own ranking basis is intentionally left alone rather than
    re-derived as part of this change. `n=None` (the default) keeps every matching zone,
    unranked, exactly as before."""
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
    if severity is not None:
        if severity not in _VALID_SEVERITIES:
            return [{"error": f"unknown severity '{severity}', expected one of {_VALID_SEVERITIES}"}]
        flagged = flagged[flagged["deterioration_band"].astype(str).str.lower() == severity]
    if n is not None:
        flagged = flagged.sort_values("trend_pts_per_qtr", ascending=True).head(n)
    return [_summary_row(r) for _, r in flagged.iterrows()]


_ANOMALY_KIND_COLUMNS = {
    "peer_gap": {"flag": "peer_gap_ml_anomaly", "band": "peer_gap_band", "factor": "factor_peer_gap"},
    "temporal": {"flag": "temporal_anomaly_ml_flag", "band": "temporal_anomaly_band", "factor": "factor_temporal_anomaly"},
}


def get_anomalous_zones(kind: str, quarter: str | None = None, emirate: str | None = None,
                         severity: str | None = None, n: int | None = None) -> list[dict]:
    """`kind="peer_gap"`: unusual vs. comparable peers, right now (Isolation Forest #1).
    `kind="temporal"`: unusual vs. its own history (Isolation Forest #2). These are the two
    analytically distinct ML outputs the brief requires -- never conflate them.

    Three independent questions this tool answers, never conflated (2026-09-17):
      - Neither `severity` nor `n` given (the original, unchanged default): "areas WITH a
        temporal/peer-gap anomaly" -- the ML detector's own flagged set
        (`temporal_anomaly_ml_flag` / `peer_gap_ml_anomaly`), severity-blind.
      - `severity` ('low'/'medium'/'high') given: "areas with HIGH temporal anomaly" -- a
        DIFFERENT, band-based question. Filters on the zone's own `temporal_anomaly_band` /
        `peer_gap_band` (the continuous Priority-factor severity, percentile-ranked among all
        classified zones every quarter -- not the ML flag's top-~8%-contamination cutoff, and
        not guaranteed to be the same set). This is the fix for a real bug: the ML-flagged set
        and the "High"-banded set usually overlap heavily but are not defined to be identical,
        so a caller asking specifically for severity must filter on the band, never quietly
        substitute the flag for it.
      - `n` given (with or without `severity`): "highest/strongest N [temporal/peer-gap]
        anomalies" -- ranks by the zone's own continuous factor score
        (`factor_temporal_anomaly` / `factor_peer_gap`) descending and caps at N. Without
        `severity`, ranks the WHOLE classified population by score (not just the ML-flagged
        subset) -- the flagged set and the true top-N by score are usually the same zones, but
        this asks for the actual top N by magnitude, not an approximation of it."""
    cols = _ANOMALY_KIND_COLUMNS.get(kind)
    if cols is None:
        return [{"error": f"unknown kind '{kind}', expected 'peer_gap' or 'temporal'"}]
    sub, quarter = _classified(quarter, emirate)
    if severity is not None:
        if severity not in _VALID_SEVERITIES:
            return [{"error": f"unknown severity '{severity}', expected one of {_VALID_SEVERITIES}"}]
        candidates = sub[sub[cols["band"]].astype(str).str.lower() == severity]
    elif n is not None:
        candidates = sub
    else:
        candidates = sub[sub[cols["flag"]]]
    if n is not None:
        candidates = candidates.sort_values(cols["factor"], ascending=False, na_position="last").head(n)
    return [_summary_row(r) for _, r in candidates.iterrows()]


# The composite land-use classifier's 4 stable groups (notebooks/09_peer_group_classifier.ipynb
# -- population/building/POI/road density, never a single OSM land-use tag; see
# project_gt_challenge memory for why). The exact strings stored in `peer_group` -- kept here as
# the one place that spells them, so `src/copilot.py`'s alias table can't drift from what's
# actually in the data.
PEER_GROUPS = ("industrial", "commercial/urban-core", "low-density residential", "rural/edge")


def get_zones_by_peer_group(peer_group: str, quarter: str | None = None, emirate: str | None = None,
                             n: int | None = None) -> list[dict]:
    """Every classified zone in the given composite peer group (2026-09-17) -- "show me
    industrial areas" / "which zones are commercial" -- a plain categorical filter, not a
    ranking. `peer_group` must be one of `PEER_GROUPS` exactly (resolving a user's free-text
    phrasing, e.g. "industry zones", down to one of these four canonical values is
    src/copilot.py's `route()`/semantic classifier's job, before this tool is ever called --
    same contract every other tool here already has for its own arguments). `n`, when given,
    narrows to the N weakest zones in that peer group by Experience Index (the one ranking
    every other listing tool in this file already defaults to) -- omitted (the default) returns
    every zone in that peer group, unranked."""
    if peer_group not in PEER_GROUPS:
        return [{"error": f"unknown peer_group '{peer_group}', expected one of {PEER_GROUPS}"}]
    sub, quarter = _classified(quarter, emirate)
    candidates = sub[sub["peer_group"] == peer_group]
    if n is not None:
        candidates = candidates.sort_values("experience_index", ascending=True).head(n)
    return [_summary_row(r) for _, r in candidates.iterrows()]


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


# Raw supporting measurements get_metric_extreme/get_metric_threshold_zones can report -- both
# explicitly NOT the Experience Index (a computed composite; that's get_weakest_zones/
# get_top_priority_zones etc.). `download_mbps`/`upload_mbps` map to their own raw columns;
# `latency_ms` maps to `latency_effective_ms` -- the same "loaded latency, falling back to
# unloaded" column every other latency figure in this app already shows the user (see
# src/compute_scores.py::add_effective_latency) -- never the raw, less-representative column.
_METRIC_COLUMNS = {"download_mbps": "download_mbps", "upload_mbps": "upload_mbps", "latency_ms": "latency_effective_ms"}
_METRIC_UNITS = {"download_mbps": "Mbps", "upload_mbps": "Mbps", "latency_ms": "ms"}


def get_metric_threshold_zones(metric: str, comparison: str, quarter: str | None = None,
                                emirate: str | None = None) -> list[dict]:
    """Zones whose raw measurement (download_mbps, upload_mbps, or latency_ms -- never the
    Experience Index, a different question; see get_weakest_zones/get_strongest_zones for that)
    is above or below the median of that same measurement among classified zones in the exact
    quarter/emirate scope the caller asked about -- the median is computed here, once,
    deterministically (pandas), over that same scope, never a fixed/national number and never
    estimated by the narrator or an LLM. Generalizes the download-only get_above_median_download_
    zones this replaces to all three raw metrics and both comparison directions, so download/
    upload/latency share one implementation instead of three near-duplicate ones.

    `comparison='above'`: metric > median, ranked furthest-above-the-median first.
    `comparison='below'`: metric < median, ranked furthest-below-the-median first. This is a
    literal numeric comparison on the raw value, never a "better/worse" judgment -- unlike
    get_metric_extreme's 'worst'/'best' valence flip (latency: lower is better), a zone with
    latency ABOVE the median latency really does have a higher latency number, which is exactly
    what 'above' means here regardless of whether that's good or bad for that metric."""
    col = _METRIC_COLUMNS.get(metric)
    if col is None:
        return [{"error": f"unknown metric '{metric}', expected one of {list(_METRIC_COLUMNS)}"}]
    if comparison not in ("above", "below"):
        return [{"error": f"unknown comparison '{comparison}', expected 'above' or 'below'"}]

    sub, quarter = _classified(quarter, emirate)
    if sub.empty:
        return []
    median_value = round(float(sub[col].median()), 2)
    if comparison == "above":
        matched = sub[sub[col] > median_value].sort_values(col, ascending=False)
    else:
        matched = sub[sub[col] < median_value].sort_values(col, ascending=True)

    return [
        {
            **_summary_row(r), metric: _num(r[col]),
            "metric": metric, "comparison": comparison,
            "median_value": median_value, "unit": _METRIC_UNITS[metric],
        }
        for _, r in matched.iterrows()
    ]


def get_metric_extreme(metric: str, operation: str, quarter: str | None = None,
                        emirate: str | None = None) -> dict:
    """The single verified min/max/median value of one raw measurement (download_mbps,
    upload_mbps, or latency_ms) over the requested quarter/emirate scope -- computed here, once,
    deterministically (pandas .min()/.max()/.median()), never estimated or calculated by an LLM.
    For 'min'/'max' this also returns the exact H3 zone that achieves it, so the frontend can
    highlight it; 'median' has no single zone that IS the median, so `zone_id` is None there --
    a caller must not invent one."""
    col = _METRIC_COLUMNS.get(metric)
    if col is None:
        return {"error": f"unknown metric '{metric}', expected one of {list(_METRIC_COLUMNS)}"}
    if operation not in ("min", "max", "median"):
        return {"error": f"unknown operation '{operation}', expected 'min', 'max', or 'median'"}

    sub, quarter = _classified(quarter, emirate)
    base = {"metric": metric, "operation": operation, "quarter": quarter,
            "emirate": emirate or "All UAE", "unit": _METRIC_UNITS[metric]}
    if sub.empty:
        return {**base, "value": None, "zone_id": None, "error": "no_classified_zones_in_scope"}

    if operation == "median":
        return {**base, "value": round(float(sub[col].median()), 2), "zone_id": None}

    idx = sub[col].idxmin() if operation == "min" else sub[col].idxmax()
    row = sub.loc[idx]
    return {
        **base, "value": _num(row[col]), "zone_id": row["h3_cell"],
        "zone_emirate": row["emirate"], "zone_area_name": row["area_name"],
        "zone_peer_group": row["peer_group"],
    }


# ---------------------------------------------------------------------------
# Area/place lookup -- "show me Seyouh", "where is Deira". The valid area-name universe for the
# copilot's text-matching router (src/copilot.py) to match against -- deliberately NOT a
# hardcoded list: it is read straight from the same processed dataset every other tool reads,
# so a name is only ever "findable" if it is a real, computed geographic fact on disk, never the
# LLM's own world knowledge of UAE geography. The "Unnamed area -- <emirate>" fallback label
# (scripts/build_area_name_field.py) is never included here -- it is an honest placeholder for a
# zone with no mapped place name, not a real place someone could search for.
# ---------------------------------------------------------------------------

def get_known_area_names() -> list[dict]:
    """Every distinct (area_name, emirate) pair actually present in the processed zone data,
    each with its full list of H3 cells -- static geographic facts, so this reads across every
    quarter's rows at once (drop_duplicates on h3_cell) rather than one quarter's slice. Most
    names have exactly one row here; a name that legitimately carries zones in more than one
    emirate (e.g. 'Al Seyouh Suburb', which straddles the Dubai/Sharjah border) gets one row per
    emirate, each with its own h3_cells -- that split is what lets locate_area's own emirate-
    disambiguation check below tell "one place, needs an emirate to pick a side" apart from "one
    unambiguous place, multiple cells"."""
    df = _load()
    real = df[~df["area_name"].str.startswith("Unnamed area")].drop_duplicates("h3_cell")
    grouped = real.groupby(["area_name", "emirate"])["h3_cell"].apply(lambda s: sorted(s.tolist()))
    return [
        {"area_name": name, "emirate": emirate, "h3_cells": cells}
        for (name, emirate), cells in grouped.items()
    ]


def locate_area(area_name: str, emirate: str | None = None) -> list[dict]:
    """Every verified H3 zone carrying the exact `area_name` string, optionally confined to one
    canonical `emirate` -- a plain, deterministic lookup against `get_known_area_names()` above.
    This function does no fuzzy/typo matching or free-text interpretation itself: resolving free
    text (exact/normalized/alias/fuzzy) down to one canonical `area_name` (and, when a name spans
    more than one emirate, resolving which one) is entirely src/copilot.py's `route()`'s job,
    BEFORE this tool is ever called -- exactly like every other tool here only ever receives
    already-resolved arguments, never raw question text.

    Returns the same "list of zone dicts, each with its own zone_id" shape every other listing
    tool returns (get_weakest_zones, get_metric_threshold_zones, ...), so the existing frontend
    highlight path (extractZoneRecords in src/build_dashboard.py) needs no new code path -- only
    this tool's own name added to GEO_TOOLS there.

    A caller that supplies no `emirate` for a name spanning more than one is refused here (an
    `error` dict, not a guess or a silently-merged answer) rather than highlighting zones from
    both emirates as if they were one place -- in the normal Copilot flow `route()` already
    resolves this before calling in, so a real caller only ever hits this path by calling
    locate_area directly with an ambiguous name and no emirate."""
    known = get_known_area_names()
    matches = [row for row in known if row["area_name"] == area_name and (emirate is None or row["emirate"] == emirate)]
    if not matches:
        return [{"error": "unknown_area_name", "area_name": area_name, "emirate": emirate}]
    emirates = sorted({row["emirate"] for row in matches})
    if len(emirates) > 1:
        return [{"error": "ambiguous_emirate", "area_name": area_name, "ambiguous_emirates": emirates}]
    resolved_emirate = emirates[0]
    h3_ids = sorted({cell for row in matches for cell in row["h3_cells"]})
    return [{"zone_id": cell, "area_name": area_name, "emirate": resolved_emirate} for cell in h3_ids]


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
        "formula": "Experience = w_download*D + w_upload*U + w_latency*L (each percentile-"
                   "ranked 0-1, latency inverted so higher is always better), rescaled to 0-100.",
        "weights": EXPERIENCE_WEIGHTS,
        "note": "Deterministic. Computed once per (zone, quarter); not classified at all "
               "below the evidence-tier minimum (see evidence_tier below).",
    },
    "confidence_score": {
        "formula": "Confidence = 50*min(1, tests/200) + 30*min(1, devices/50) "
                   "+ 20*(quarters_observed/8).",
        "weights": CONFIDENCE_WEIGHTS,
        "tests_cap": CONFIDENCE_TESTS_CAP,
        "devices_cap": CONFIDENCE_DEVICES_CAP,
        "note": "A continuous strength-of-evidence signal, separate from the evidence_tier "
               "eligibility gate (min_tests_for_reliable_evidence below is the 'full' tier bar, "
               "not this score) and from the Priority confidence gate G(C) below.",
    },
    "evidence_tier": {
        "formula": "'insufficient' (tests<10 or devices<3, no Experience Index at all) / "
                   "'low' (10-29 tests, Experience Index shown but not shortlist-eligible) / "
                   "'full' (>=30 tests, fully scored and shortlist-eligible).",
        "min_tests_for_reliable_evidence": MIN_TESTS_FOR_RELIABLE_EVIDENCE,
    },
    "peer_gap": {
        "formula": "Peer Gap = this zone's Experience Index - its peer group's median "
                   "Experience Index, in the same quarter. Peer group is a composite of "
                   "population/building/POI/road density (never OSM land-use tag, and "
                   "never the UAE-wide distribution).",
    },
    "trend": {
        "formula": f"A zone is flagged 'deteriorating' once its gap to the national median "
                   f"Experience Index has widened for {CONSECUTIVE_DECLINES_REQUIRED} consecutive "
                   f"quarters -- relative to the national trend, never raw Mbps, and never on a "
                   f"single bad quarter. The Deterioration factor used in Priority is continuous: "
                   f"zero unless that persistence bar is cleared, then scaled by how severe the "
                   f"relative decline is compared to other currently-deteriorating zones.",
        "consecutive_declines_required": CONSECUTIVE_DECLINES_REQUIRED,
    },
    "priority_score": {
        "formula": "Priority = 100 * G(Confidence) * (w_peer_gap*PeerGapFactor "
                   "+ w_deterioration*DeteriorationFactor + w_temporal_anomaly*TemporalAnomalyFactor "
                   "+ w_population*PopulationExposureFactor), where each factor is a percentile "
                   "rank (0-1) within its quarter and PopulationExposure = log(1 + population). "
                   "G(C) = 0 if C<40, else 0.5 + 0.5*(C/100).",
        "weights": PRIORITY_WEIGHTS,
        "confidence_gate_threshold": CONFIDENCE_GATE_THRESHOLD,
        "priority_zone_fraction": PRIORITY_ZONE_FRACTION,
        "note": "Confidence is a gate, not a linear multiplier: below the threshold a zone "
               "cannot reach the shortlist at all, however bad its other factors look; above "
               "it, the gate runs 0.7-1.0, so confidence tunes the ranking without dominating "
               "it. TemporalAnomaly (Isolation Forest vs. the zone's own history) is a "
               "distinct factor; the Peer Gap ML score is not a Priority factor (the "
               "deterministic PeerGap factor already covers that ground).",
    },
}


def get_methodology(metric: str) -> dict:
    """Returns the formula/weights/thresholds actually used for one named metric -- the
    same CONFIG values compute_scores.py/trends.py/priority.py import, not a paraphrase."""
    result = _METHODOLOGY.get(metric)
    if result is None:
        return {"error": f"unknown metric '{metric}'", "known_metrics": list(_METHODOLOGY)}
    return {"metric": metric, **result}
