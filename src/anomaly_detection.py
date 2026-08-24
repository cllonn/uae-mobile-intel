"""
ML anomaly detection: Peer Gap and Temporal Anomaly, via Isolation Forest, each compared
against a simple deterministic baseline. Two analytically distinct outputs, per the brief --
same technique, different inputs: Peer Gap asks "unusual vs. its peers, right now?",
Temporal Anomaly asks "unusual vs. its own history?"

This is a first-pass detector, not the full formal T2 benchmark (synthetic anomaly injection
with precision/recall/F1 on a blind hold-out set) -- that's Phase 5 testing-pack work, run
once this feature set is settled. What's here is real, though: a genuine unsupervised model,
fit on real data, compared against a real deterministic baseline, not a placeholder.

Input: the output of `compute_scores.score_zone_quarters` (+ `trends.add_trend` for the
temporal detector, which needs `experience_index` history).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

RANDOM_STATE = 42
# Expected fraction of genuinely anomalous zone-quarters -- a starting estimate, not tuned;
# revisit once the formal synthetic-injection benchmark (T2) is run.
CONTAMINATION = 0.05


def _zscore_vs_peer_group(df: pd.DataFrame, col: str) -> pd.Series:
    """How many peer-group standard deviations this zone's value is from its peer group's
    mean, computed separately within each (peer_group, quarter) -- never against the national
    distribution, per the brief's mandatory peer-group rule."""
    grouped = df.groupby(["peer_group", "quarter"])[col]
    mean = grouped.transform("mean")
    std = grouped.transform("std").replace(0, np.nan)
    return (df[col] - mean) / std


def add_peer_gap_ml(df: pd.DataFrame) -> pd.DataFrame:
    """Isolation Forest over how a zone compares to its peers *this quarter*: download,
    upload, and latency (inverted so higher is always better), each as a peer-group z-score.
    Only fit on classified (non-insufficient-evidence) zone-quarters -- an insufficient-
    evidence row has no reliable metrics to compare in the first place."""
    df = df.copy()
    classified = df.loc[~df["insufficient_evidence"]].copy()

    classified["z_download"] = _zscore_vs_peer_group(classified, "download_mbps")
    classified["z_upload"] = _zscore_vs_peer_group(classified, "upload_mbps")
    classified["z_latency_inv"] = -_zscore_vs_peer_group(classified, "latency_effective_ms")

    feature_cols = ["z_download", "z_upload", "z_latency_inv"]
    features = classified[feature_cols].dropna()

    model = IsolationForest(contamination=CONTAMINATION, random_state=RANDOM_STATE)
    model.fit(features)
    # decision_function: higher = more normal. Flipped so higher = more anomalous, easier to
    # read as a "how unusual is this" score for the priority engine and the UI.
    classified.loc[features.index, "peer_gap_ml_score"] = -model.decision_function(features)
    classified.loc[features.index, "peer_gap_ml_anomaly"] = model.predict(features) == -1  # -1 = anomaly

    # Merge on the actual key columns rather than the row index -- the index isn't guaranteed
    # to be unique or aligned coming in, and a join on a non-unique index silently explodes
    # into a cross-join instead of erroring cleanly.
    df = df.merge(
        classified[["h3_cell", "quarter", "peer_gap_ml_score", "peer_gap_ml_anomaly"]],
        on=["h3_cell", "quarter"], how="left",
    )
    df["peer_gap_ml_anomaly"] = df["peer_gap_ml_anomaly"].fillna(False)
    return df


def add_temporal_anomaly_ml(df: pd.DataFrame) -> pd.DataFrame:
    """Isolation Forest over how a zone compares to *its own history*: the quarter-over-quarter
    change in Experience Index, and the zone's own volatility (std dev of Experience Index
    across all its classified quarters). A zone that normally swings +/-20 points isn't
    unusual for moving 15 points; a zone that's always steady within +/-2 points is, even
    though the two might land on the same raw score this quarter."""
    df = df.copy()
    classified = df.loc[~df["insufficient_evidence"]].copy()

    classified["own_qoq_change"] = classified.groupby("h3_cell")["experience_index"].diff()
    classified["own_volatility"] = classified.groupby("h3_cell")["experience_index"].transform("std")

    feature_cols = ["own_qoq_change", "own_volatility"]
    features = classified[feature_cols].dropna()

    model = IsolationForest(contamination=CONTAMINATION, random_state=RANDOM_STATE)
    model.fit(features)
    classified.loc[features.index, "temporal_anomaly_ml_score"] = -model.decision_function(features)
    classified.loc[features.index, "temporal_anomaly_ml_flag"] = model.predict(features) == -1

    df = df.merge(
        classified[["h3_cell", "quarter", "temporal_anomaly_ml_score", "temporal_anomaly_ml_flag"]],
        on=["h3_cell", "quarter"], how="left",
    )
    df["temporal_anomaly_ml_flag"] = df["temporal_anomaly_ml_flag"].fillna(False)
    return df


def add_baseline_bottom_decile(df: pd.DataFrame, decile: float = 0.10) -> pd.DataFrame:
    """Simple deterministic baseline for comparison: flag a zone if its download speed is in
    the bottom `decile` within its own peer group and quarter. The brief requires the ML to be
    rigorously compared against exactly this kind of simple rule -- 'if your evaluation shows
    the ML does not beat the simple baseline, report that honestly.'"""
    df = df.copy()
    threshold = df.groupby(["peer_group", "quarter"])["download_mbps"].transform(
        lambda s: s.quantile(decile)
    )
    df["baseline_bottom_decile_flag"] = (df["download_mbps"] <= threshold) & ~df["insufficient_evidence"]
    return df


def compare_flags_to_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """How much does each ML detector's flag set overlap with the simple baseline's? Not a
    formal precision/recall benchmark (there's no ground truth on real data -- that requires
    the synthetic injection test, Phase 5), but a first honest check on whether the ML is
    finding something different from -- or just rediscovering -- the deterministic rule."""
    classified = df.loc[~df["insufficient_evidence"]]
    rows = []
    for ml_col, label in [
        ("peer_gap_ml_anomaly", "Peer Gap ML"),
        ("temporal_anomaly_ml_flag", "Temporal Anomaly ML"),
    ]:
        ml_flagged = classified[ml_col].sum()
        baseline_flagged = classified["baseline_bottom_decile_flag"].sum()
        both = (classified[ml_col] & classified["baseline_bottom_decile_flag"]).sum()
        rows.append({
            "detector": label,
            "ml_flagged": int(ml_flagged),
            "baseline_flagged": int(baseline_flagged),
            "overlap": int(both),
            "ml_only": int(ml_flagged - both),
            "baseline_only": int(baseline_flagged - both),
        })
    return pd.DataFrame(rows)
