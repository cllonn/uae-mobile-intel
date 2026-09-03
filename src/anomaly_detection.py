"""
ML anomaly detection: Peer Gap and Temporal Anomaly, via two separate Isolation Forest models,
each compared against a simple deterministic baseline. Two analytically distinct outputs, per
the brief -- same technique, different inputs: Peer Gap asks "unusual vs. its peers, right
now?", Temporal Anomaly asks "unusual vs. its own history?". Mentor-specified methodology,
2026-09-01 -- see `tests/test_t2_anomaly_benchmark.py` for the formal synthetic-injection
benchmark this feeds into.

Why two models rather than one: putting spatial and temporal features in one model makes the
anomaly score uninterpretable -- you can't tell a panel whether a zone was flagged because it's
worse than its peers or because it changed. Two models give two sentences you can actually say
out loud.

Why every feature is relative, never raw: Isolation Forest finds points that are unusual in the
feature space it's given. Raw download values make it flag rural zones for being rural --
correctly by its own logic, while missing an urban zone that's 20 points below comparable urban
zones. Relative features (peer-group z-scores, national-trend-relative changes) ask "unusual
for what it is," which is the actual question.

Why only tests>=30 ('full' evidence tier) zones train/score here: the zones below that bar have
quarterly values that are mostly noise, which would widen the model's notion of "normal" so far
that genuine anomalies stop looking unusual. Excluding them from the model is a modelling
decision, not a product one -- they stay visible on the map, labelled low/insufficient evidence.

Input: the output of `compute_scores.score_zone_quarters` (needs `evidence_tier`, `peer_gap`,
`peer_group`, `download_mbps`, `upload_mbps`, `latency_effective_ms`, `experience_index`).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
N_ESTIMATORS = 300
# Fraction of zones the model is allowed to flag (mentor-specified, 2026-09-01) -- not the
# default, because leaving it at "auto" gives no control over how many flags you're asking for,
# which is one of the most common reasons a synthetic test appears to fail (inject anomalies
# into 10% of zones, model only permitted to find 5%). Matches the T2 injection rate so the
# benchmark is fair, and roughly matches "a couple of dozen zones out of ~150" for the shortlist.
CONTAMINATION = 0.08

# Minimum trailing quarters of Experience Index history required before a zone gets a temporal
# feature row at all -- a 3-quarter slope/volatility estimate from 2 points is noise.
MIN_QUARTERS_FOR_TEMPORAL = 4


def _peer_zscore(df: pd.DataFrame, col: str) -> pd.Series:
    """How many peer-group standard deviations this zone's value is from its peer group's
    mean, computed separately within each (peer_group, quarter) -- never against the national
    distribution, per the brief's mandatory peer-group rule."""
    grouped = df.groupby(["peer_group", "quarter"])[col]
    mean = grouped.transform("mean")
    std = grouped.transform("std").replace(0, np.nan)
    return (df[col] - mean) / std


def _fit_score_flag(features: pd.DataFrame, contamination: float = CONTAMINATION):
    """StandardScaler -> IsolationForest(n_estimators=300, contamination, random_state=42) ->
    (score, flag), mentor-specified 2026-09-01. `score` is `-score_samples` (higher = more
    anomalous); `flag` is score at or above the (1 - contamination) quantile of its own score
    distribution -- the model's own top `contamination` fraction, not sklearn's internal
    `predict()` threshold (which uses a slightly different offset convention). Both approaches
    agree on which points are flagged in the vast majority of cases; this follows the mentor's
    exact snippet for reproducibility."""
    X = StandardScaler().fit_transform(features)
    model = IsolationForest(n_estimators=N_ESTIMATORS, contamination=contamination, random_state=RANDOM_STATE)
    model.fit(X)
    score = -model.score_samples(X)
    flag = score >= np.quantile(score, 1 - contamination)
    return score, flag


# ---------------------------------------------------------------------------
# Spatial ("Peer Gap") model: exp_gap, dl_z, ul_z, lat_z, lat_ratio
# ---------------------------------------------------------------------------

def build_spatial_features(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (h3_cell, quarter) among 'full' evidence-tier zones -- 5 features, all
    relative to peer group, never raw (mentor-specified, 2026-09-01):
      - exp_gap: zone Experience Index minus its peer group's median (same as `peer_gap`).
      - dl_z, ul_z, lat_z: peer-group z-scores of download/upload/latency.
      - lat_ratio: latency divided by the peer group's median latency."""
    full = df.loc[df["evidence_tier"] == "full"].copy()
    full["exp_gap"] = full["peer_gap"]
    full["dl_z"] = _peer_zscore(full, "download_mbps")
    full["ul_z"] = _peer_zscore(full, "upload_mbps")
    full["lat_z"] = _peer_zscore(full, "latency_effective_ms")
    peer_median_lat = full.groupby(["peer_group", "quarter"])["latency_effective_ms"].transform("median")
    full["lat_ratio"] = full["latency_effective_ms"] / peer_median_lat.replace(0, np.nan)
    return full


SPATIAL_FEATURE_COLS = ["exp_gap", "dl_z", "ul_z", "lat_z", "lat_ratio"]


def add_peer_gap_ml(df: pd.DataFrame) -> pd.DataFrame:
    """Isolation Forest over `SPATIAL_FEATURE_COLS`, fit on 'full' evidence-tier zone-quarters
    only. Adds `peer_gap_ml_score`/`peer_gap_ml_anomaly` (False/NaN for 'low'/'insufficient'
    tier rows, which never enter the model)."""
    df = df.copy()
    full = build_spatial_features(df)
    features = full[SPATIAL_FEATURE_COLS].dropna()

    score, flag = _fit_score_flag(features)
    full.loc[features.index, "peer_gap_ml_score"] = score
    full.loc[features.index, "peer_gap_ml_anomaly"] = flag

    df = df.merge(
        full[["h3_cell", "quarter", "peer_gap_ml_score", "peer_gap_ml_anomaly"]],
        on=["h3_cell", "quarter"], how="left",
    )
    df["peer_gap_ml_anomaly"] = df["peer_gap_ml_anomaly"].fillna(False)
    return df


# ---------------------------------------------------------------------------
# Temporal ("Temporal Anomaly") model: qoq_rel, slope_rel, volatility, consec_decline
# ---------------------------------------------------------------------------

def _national_series(full: pd.DataFrame) -> pd.Series:
    """Median Experience Index per quarter across 'full' evidence-tier zones -- the national
    reference every temporal feature is measured relative to, never a zone's raw own history."""
    return full.groupby("quarter")["experience_index"].median().sort_index()


def _zone_temporal_rows(h3_cell: str, group: pd.DataFrame, national: pd.Series,
                         min_quarters: int) -> list[dict]:
    """qoq_rel, slope_rel, volatility, consec_decline for one zone's timeline -- a plain loop
    over quarters (mirrors `trends.py`'s style: at 8 quarters per zone the performance cost is
    negligible, and the loop reads like the rule it implements). Emits one row per quarter that
    has >= min_quarters of trailing history available; earlier quarters are skipped, not padded
    with fabricated values."""
    group = group.sort_values("quarter")
    quarters = group["quarter"].tolist()
    exp = group["experience_index"].to_numpy(dtype=float)
    nat = national.reindex(quarters).to_numpy(dtype=float)

    own_qoq = np.diff(exp, prepend=np.nan)
    nat_qoq = np.diff(nat, prepend=np.nan)
    qoq_rel = own_qoq - nat_qoq

    rows = []
    consec = 0
    for i in range(len(quarters)):
        consec = consec + 1 if (pd.notna(qoq_rel[i]) and qoq_rel[i] < 0) else 0
        if i + 1 < min_quarters:
            continue
        window = exp[i - min_quarters + 1: i + 1]
        nat_window = nat[i - min_quarters + 1: i + 1]
        if np.isnan(window).any() or np.isnan(nat_window).any():
            continue
        zone_slope = np.polyfit(range(min_quarters), window, 1)[0]
        nat_slope = np.polyfit(range(min_quarters), nat_window, 1)[0]
        vol_window = qoq_rel[max(0, i - min_quarters + 1): i + 1]
        vol_window = vol_window[~np.isnan(vol_window)]
        rows.append({
            "h3_cell": h3_cell, "quarter": quarters[i],
            "qoq_rel": qoq_rel[i],
            "slope_rel": zone_slope - nat_slope,
            "volatility": float(np.std(vol_window, ddof=1)) if len(vol_window) > 1 else np.nan,
            "consec_decline": consec,
        })
    return rows


def build_temporal_features(df: pd.DataFrame, min_quarters: int = MIN_QUARTERS_FOR_TEMPORAL) -> pd.DataFrame:
    """One row per (h3_cell, quarter) among 'full' evidence-tier zones with >= min_quarters of
    trailing Experience Index history -- 4 features, changes relative to the national trend,
    never raw (mentor-specified, 2026-09-01): qoq_rel, slope_rel, volatility, consec_decline."""
    full = df.loc[df["evidence_tier"] == "full"].copy()
    national = _national_series(full)
    all_rows = []
    for h3_cell, group in full.groupby("h3_cell"):
        all_rows.extend(_zone_temporal_rows(h3_cell, group, national, min_quarters))
    return pd.DataFrame(all_rows)


TEMPORAL_FEATURE_COLS = ["qoq_rel", "slope_rel", "volatility", "consec_decline"]


def add_temporal_anomaly_ml(df: pd.DataFrame) -> pd.DataFrame:
    """Isolation Forest over `TEMPORAL_FEATURE_COLS`, fit on 'full' evidence-tier zone-quarters
    with >= MIN_QUARTERS_FOR_TEMPORAL of history. Adds `temporal_anomaly_ml_score`/
    `temporal_anomaly_ml_flag` (False/NaN for every zone-quarter that doesn't qualify)."""
    df = df.copy()
    feat_table = build_temporal_features(df)
    features = feat_table[TEMPORAL_FEATURE_COLS].dropna()

    if len(features):
        score, flag = _fit_score_flag(features)
        feat_table.loc[features.index, "temporal_anomaly_ml_score"] = score
        feat_table.loc[features.index, "temporal_anomaly_ml_flag"] = flag

    df = df.merge(
        feat_table[["h3_cell", "quarter", "temporal_anomaly_ml_score", "temporal_anomaly_ml_flag"]],
        on=["h3_cell", "quarter"], how="left",
    )
    df["temporal_anomaly_ml_flag"] = df["temporal_anomaly_ml_flag"].fillna(False)
    return df


# ---------------------------------------------------------------------------
# Deterministic comparison baseline
# ---------------------------------------------------------------------------

def add_baseline_bottom_decile(df: pd.DataFrame, decile: float = 0.10) -> pd.DataFrame:
    """Deterministic comparison baseline: bottom decile of **Experience Index** within peer
    group and quarter (mentor-specified, 2026-09-01) -- not download speed, and not nationally.
    A national bottom decile is mostly rural zones, which makes the baseline trivially weak and
    any ML result look better than it is by comparison; ranking by Experience Index (the same
    metric the ML/Priority engine actually judge zones on) within peer group is the strongest
    fair baseline. Computed over the same tests>=30 ('full' tier) population the ML models use."""
    df = df.copy()
    full_mask = df["evidence_tier"] == "full"
    threshold = df.loc[full_mask].groupby(["peer_group", "quarter"])["experience_index"].transform(
        lambda s: s.quantile(decile)
    )
    df["baseline_bottom_decile_flag"] = False
    df.loc[full_mask, "baseline_bottom_decile_flag"] = df.loc[full_mask, "experience_index"] <= threshold
    return df


def compare_flags_to_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """How much does each ML detector's flag set overlap with the simple baseline's? Not a
    formal precision/recall benchmark (there's no ground truth on real data -- that requires
    the synthetic injection test, T2), but a first honest check on whether the ML is finding
    something different from -- or just rediscovering -- the deterministic rule. Restricted to
    'full' evidence tier, the only population any of these three flags are ever computed for."""
    full = df.loc[df["evidence_tier"] == "full"]
    rows = []
    for ml_col, label in [
        ("peer_gap_ml_anomaly", "Peer Gap ML"),
        ("temporal_anomaly_ml_flag", "Temporal Anomaly ML"),
    ]:
        ml_flagged = full[ml_col].sum()
        baseline_flagged = full["baseline_bottom_decile_flag"].sum()
        both = (full[ml_col] & full["baseline_bottom_decile_flag"]).sum()
        rows.append({
            "detector": label,
            "ml_flagged": int(ml_flagged),
            "baseline_flagged": int(baseline_flagged),
            "overlap": int(both),
            "ml_only": int(ml_flagged - both),
            "baseline_only": int(baseline_flagged - both),
        })
    return pd.DataFrame(rows)
