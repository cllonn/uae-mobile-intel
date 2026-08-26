"""
T2 -- Synthetic anomaly benchmark, reproducible.

Reproduces the methodology already reported in docs/validation_summary.md: inject known
synthetic anomalies (download -40%, latency +80%, combined, and a gradual 3-quarter decline)
into a blind hold-out slice of classified zones that model fitting never sees, then compare
Isolation Forest against the deterministic bottom-decile baseline on precision/recall/F1/FPR.

This is a REPRODUCIBILITY script, not a re-tuning exercise:
- Isolation Forest's hyperparameters and feature choices are unchanged from
  src/anomaly_detection.py (same RANDOM_STATE, same CONTAMINATION) -- nothing here redesigns
  the model.
- The train/hold-out split and which zones get an injected fault are fixed by SPLIT_SEED,
  chosen once. This script is meant to be run once and reported, not iterated against until
  the numbers look better -- doing that would defeat the point of a blind hold-out.
- Reference statistics (peer-group mean/std for z-scores, the baseline's decile threshold) are
  computed from the TRAIN split only, then applied to the hold-out -- the hold-out's own
  (possibly injected) values never leak into the numbers used to judge it.

Numbers here will differ from the earlier externally-reported T2 run, and that is expected, not
a bug: that run used the 868-zone classified population under the old (incorrect) 5-test
threshold; this repo's classified population is now 314 zones under the corrected 30-test
threshold (see docs/validation_summary.md, "Evidence threshold correction"). Report whatever
this produces -- do not adjust the injection or split to chase the old numbers.

Run: python -m tests.test_t2_anomaly_benchmark
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.anomaly_detection import CONTAMINATION, RANDOM_STATE

ZONE_PRIORITY_PATH = Path("data/processed/zone_priority.parquet")
QUARTER = "2026Q2"
HOLDOUT_FRACTION = 0.20
ANOMALY_FRACTION_OF_HOLDOUT = 0.35  # share of the hold-out given an injected fault
SPLIT_SEED = 123  # fixed once; never touched again after seeing results


def _split_train_holdout(zone_ids: list[str], seed: int) -> tuple[list[str], list[str]]:
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(zone_ids)
    n_holdout = max(1, round(len(shuffled) * HOLDOUT_FRACTION))
    return list(shuffled[n_holdout:]), list(shuffled[:n_holdout])


def _precision_recall_f1_fpr(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {"precision": round(precision, 3), "recall": round(recall, 3),
            "f1": round(f1, 3), "fpr": round(fpr, 3), "n_flagged": int(y_pred.sum()), "n_true": int(y_true.sum())}


# ---------------------------------------------------------------------------
# Benchmark A -- cross-sectional (Peer Gap ML): download -40%, latency +80%, combined
# ---------------------------------------------------------------------------

def _injected_types_for(rng, holdout_ids: list[str], anomaly_fraction: float) -> dict[str, str]:
    """Assigns each of a random subset of hold-out zones one of the three cross-sectional
    fault types, evenly split; the rest are labeled 'normal' (no injection)."""
    n_anomalous = round(len(holdout_ids) * anomaly_fraction)
    anomalous_ids = rng.choice(holdout_ids, size=n_anomalous, replace=False)
    types = ["download_drop", "latency_spike", "combined"]
    return {zid: types[i % 3] for i, zid in enumerate(anomalous_ids)}


def run_benchmark_a(df: pd.DataFrame, train_ids: list[str], holdout_ids: list[str], rng) -> pd.DataFrame:
    train = df[df["h3_cell"].isin(train_ids)].copy()
    holdout = df[df["h3_cell"].isin(holdout_ids)].copy()

    injection = _injected_types_for(rng, holdout_ids, ANOMALY_FRACTION_OF_HOLDOUT)
    holdout["injected_type"] = holdout["h3_cell"].map(injection).fillna("normal")
    holdout["is_anomaly"] = (holdout["injected_type"] != "normal").astype(int)

    # Apply the fault to a COPY of the raw metrics -- only the hold-out is ever touched.
    dl = holdout["download_mbps"].copy()
    lat = holdout["latency_effective_ms"].copy()
    drop_mask = holdout["injected_type"].isin(["download_drop", "combined"])
    spike_mask = holdout["injected_type"].isin(["latency_spike", "combined"])
    dl.loc[drop_mask] *= 0.60          # download -40%
    lat.loc[spike_mask] *= 1.80        # latency +80%
    holdout["download_mbps_injected"] = dl
    holdout["latency_injected"] = lat

    # Reference stats from TRAIN only -- the hold-out's (injected) values never inform them.
    ref = train.groupby("peer_group")[["download_mbps", "upload_mbps", "latency_effective_ms"]].agg(["mean", "std"])

    def zscore(row, col, injected_col=None):
        val = row[injected_col] if injected_col else row[col]
        mean = ref.loc[row["peer_group"], (col, "mean")]
        std = ref.loc[row["peer_group"], (col, "std")]
        return (val - mean) / std if std and not pd.isna(std) and std != 0 else np.nan

    train["z_download"] = train.apply(lambda r: zscore(r, "download_mbps"), axis=1)
    train["z_upload"] = train.apply(lambda r: zscore(r, "upload_mbps"), axis=1)
    train["z_latency_inv"] = -train.apply(lambda r: zscore(r, "latency_effective_ms"), axis=1)
    holdout["z_download"] = holdout.apply(lambda r: zscore(r, "download_mbps", "download_mbps_injected"), axis=1)
    holdout["z_upload"] = holdout.apply(lambda r: zscore(r, "upload_mbps"), axis=1)
    holdout["z_latency_inv"] = -holdout.apply(lambda r: zscore(r, "latency_effective_ms", "latency_injected"), axis=1)

    feature_cols = ["z_download", "z_upload", "z_latency_inv"]
    train_features = train[feature_cols].dropna()
    holdout_features = holdout[feature_cols].dropna()
    holdout_eval = holdout.loc[holdout_features.index]

    model = IsolationForest(contamination=CONTAMINATION, random_state=RANDOM_STATE)
    model.fit(train_features)
    ml_pred = (model.predict(holdout_features) == -1).astype(int)

    # Baseline: bottom-decile download within peer group, threshold from TRAIN only.
    decile_threshold = train.groupby("peer_group")["download_mbps"].quantile(0.10)
    baseline_pred = (holdout_eval["download_mbps_injected"] <= holdout_eval["peer_group"].map(decile_threshold)).astype(int).to_numpy()

    y_true = holdout_eval["is_anomaly"].to_numpy()
    results = []
    for label, pred in [("Peer Gap ML", ml_pred), ("Baseline (bottom-decile download)", baseline_pred)]:
        results.append({"detector": label, **_precision_recall_f1_fpr(y_true, pred)})

    # Per-injection-type recall, dropped rows excluded (same features.dropna() population)
    for inj_type in ["download_drop", "latency_spike", "combined"]:
        mask = holdout_eval["injected_type"] == inj_type
        if mask.sum() == 0:
            continue
        ml_recall = ml_pred[mask.to_numpy()].mean()
        results.append({"detector": f"  (Peer Gap ML recall on '{inj_type}' only, n={int(mask.sum())})",
                        "precision": None, "recall": round(float(ml_recall), 3), "f1": None, "fpr": None,
                        "n_flagged": None, "n_true": int(mask.sum())})

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Benchmark B -- temporal (gradual 3-quarter decline), no baseline exists for this shape
# ---------------------------------------------------------------------------

def run_benchmark_b(df: pd.DataFrame, train_ids: list[str], holdout_ids: list[str], rng) -> pd.DataFrame:
    all_history = df.sort_values("quarter")

    def own_features(zone_ids, inject: bool):
        rows = []
        anomalous = set()
        if inject:
            n_anomalous = round(len(zone_ids) * ANOMALY_FRACTION_OF_HOLDOUT)
            anomalous = set(rng.choice(zone_ids, size=n_anomalous, replace=False))
        for zid in zone_ids:
            hist = all_history[(all_history["h3_cell"] == zid) & (~all_history["insufficient_evidence"])]
            exp = hist["experience_index"].to_numpy(dtype=float)
            if len(exp) < 4:
                continue
            if zid in anomalous:
                # Gradual 3-quarter decline: taper the last 3 observed quarters down by an
                # increasing amount, simulating exactly the pattern the brief describes.
                exp = exp.copy()
                exp[-3:] = exp[-3:] * [0.92, 0.84, 0.76]
            qoq_change = exp[-1] - exp[-2]
            volatility = np.std(exp, ddof=1) if len(exp) > 1 else np.nan
            rows.append({"h3_cell": zid, "own_qoq_change": qoq_change, "own_volatility": volatility,
                        "is_anomaly": int(zid in anomalous)})
        return pd.DataFrame(rows)

    train_feat = own_features(train_ids, inject=False).dropna(subset=["own_qoq_change", "own_volatility"])
    holdout_feat = own_features(holdout_ids, inject=True).dropna(subset=["own_qoq_change", "own_volatility"])

    feature_cols = ["own_qoq_change", "own_volatility"]
    model = IsolationForest(contamination=CONTAMINATION, random_state=RANDOM_STATE)
    model.fit(train_feat[feature_cols])
    pred = (model.predict(holdout_feat[feature_cols]) == -1).astype(int)

    result = _precision_recall_f1_fpr(holdout_feat["is_anomaly"].to_numpy(), pred)
    return pd.DataFrame([{"detector": "Temporal Anomaly ML (no baseline exists for gradual decline)", **result}])


def test_t2_benchmark_runs_and_reports_honestly():
    df = pd.read_parquet(ZONE_PRIORITY_PATH)
    classified_latest = df[(df["quarter"] == QUARTER) & (~df["insufficient_evidence"])]
    zone_ids = classified_latest["h3_cell"].unique().tolist()
    train_ids, holdout_ids = _split_train_holdout(zone_ids, SPLIT_SEED)
    rng = np.random.default_rng(SPLIT_SEED)

    print(f"Classified 2026Q2 zones: {len(zone_ids)} -> train {len(train_ids)}, hold-out {len(holdout_ids)}")
    print()
    print("=== Benchmark A: cross-sectional (Peer Gap ML) ===")
    a = run_benchmark_a(df, train_ids, holdout_ids, rng)
    print(a.to_string(index=False))

    # Sanity, not a performance bar: every metric must be a real, in-range number.
    ml_row = a[a["detector"] == "Peer Gap ML"].iloc[0]
    for metric in ["precision", "recall", "f1", "fpr"]:
        assert 0.0 <= ml_row[metric] <= 1.0, f"{metric} out of range: {ml_row[metric]}"

    print()
    print("=== Benchmark B: temporal (gradual 3-quarter decline) ===")
    b = run_benchmark_b(df, train_ids, holdout_ids, rng)
    print(b.to_string(index=False))
    b_row = b.iloc[0]
    for metric in ["precision", "recall", "f1", "fpr"]:
        assert 0.0 <= b_row[metric] <= 1.0, f"{metric} out of range: {b_row[metric]}"

    print()
    print("T2 ran end to end on the current (tests>=30) classified population. Report these "
          "numbers as-is in docs/validation_summary.md -- do not re-run with a different seed "
          "or injection fraction to chase a more favorable result.")


if __name__ == "__main__":
    test_t2_benchmark_runs_and_reports_honestly()
    print("\nT2 PASSED (ran end to end; see printed metrics above for the actual result).")
