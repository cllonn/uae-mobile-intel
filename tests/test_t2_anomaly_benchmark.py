"""
T2 -- Synthetic anomaly benchmark, mentor's exact protocol (2026-09-01). Supersedes the earlier
T2 script entirely (different feature sets, different injection/split procedure, different
baseline) -- do not compare numbers across the two.

Procedure, in the mentor's own order:
  1. Measure natural noise first: the standard deviation of qoq_rel across real (pre-injection)
     'full' evidence-tier zones. Injected changes must clear this by a wide margin or no method
     could find them regardless of how good it is.
  2. Pick 10% of eligible ('full' tier, tests>=30, latest quarter) zones at random, inject one
     of four faults (brief's own list): download -40% (spatial), latency +80% (spatial), a
     gradual relative decline (-8%/quarter for 3 quarters, temporal), or a combined degradation
     (download -40% AND latency +80% on the same zone at once, spatial) -- split as evenly as
     the eligible population allows across the four types.
  3. Split the INJECTED zones 50/50 into a tuning half and a blind hold-out half (the untouched
     'normal' zones are split 50/50 too, so both halves have a realistic, comparable mix to
     evaluate precision/recall against). Fit each model on the full modified table (unsupervised
     -- fitting never looks at the injected/normal labels, so this isn't label leakage even
     though the hold-out's own [possibly injected] feature values are part of the fit
     population). Contamination and the feature set are fixed by explicit spec here, so
     "tuning" reduces to a consistency check (printed for reference) rather than an actual
     hyperparameter search -- the hold-out half's numbers are what's reported as the result.
  4. Report precision/recall/F1/false-positive-rate on the hold-out half only (brief's own
     required metric list), plus a per-fault-type recall breakdown for Isolation Forest so the
     4 injected fault types (download-only, latency-only, combined, gradual decline) can be
     told apart rather than averaged into one number.
  5. Baseline = bottom-decile Experience Index within peer group (src/anomaly_detection.py's
     production `add_baseline_bottom_decile`), threshold fit on the tuning half, applied to
     hold-out.
  6. LOF (LocalOutlierFactor(n_neighbors=20, contamination=0.08)) as a second comparison method
     alongside Isolation Forest, same features, same StandardScaler, same fit population.

Recall target (brief's own requirement: set it from the data distribution, justify it, never
tune it to flatter the result) -- see SINGLE_FAULT_RECALL_TARGET / COMBINED_FAULT_RECALL_TARGET
below for the reasoning, fixed before this script ever inspects a hold-out result.

Both models reuse `src/anomaly_detection.py`'s own feature-building and fit/score functions
(`build_spatial_features`, `build_temporal_features`, `_fit_score_flag`) rather than
reimplementing them -- this exercises the exact same code path production uses, not a parallel
approximation of it.

Run: python -m tests.test_t2_anomaly_benchmark
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from src.anomaly_detection import (
    CONTAMINATION, MIN_QUARTERS_FOR_TEMPORAL, SPATIAL_FEATURE_COLS, TEMPORAL_FEATURE_COLS,
    _fit_score_flag, build_spatial_features, build_temporal_features,
)
from src.compute_scores import experience_index as _recompute_experience_index
from src.compute_scores import peer_gap as _recompute_peer_gap

ZONE_PRIORITY_PATH = Path("data/processed/zone_priority.parquet")
QUARTER = "2026Q2"
INJECTION_FRACTION = 0.10
SEED = 123  # fixed once; never touched again after seeing results
TAPER = [0.92, 0.84, 0.76]  # gradual -8%/quarter cumulative decline, 3 quarters

# --- Recall target -- set here, from the data/model distribution, BEFORE this script ever runs
# a detector or looks at a hold-out result (brief's own instruction: never tune to flatter the
# result). Reasoning: CONTAMINATION=0.08 caps Isolation Forest at flagging 8% of the scored
# population, and injected zones are ~10% of the eligible pool split ~evenly across 4 fault
# types (~2.5% each) -- comfortably inside that budget, so the flagging budget isn't the binding
# constraint here. Separability is: SPATIAL_FEATURE_COLS has 5 features (exp_gap, dl_z, ul_z,
# lat_z, lat_ratio). A single-fault injection (download -40% OR latency +80% alone) pushes only
# 1-2 of those 5 into the tail (the directly-hit z-score, plus a smaller knock-on shift in
# exp_gap via the recomputed Experience Index) while the other 3-4 stay ordinary for that zone's
# peer group -- Isolation Forest's isolation-path score is an average over all 5 dimensions, so
# one strong outlier feature gets diluted by several ordinary ones. A combined injection (both
# faults on the same zone) pushes 2-3 of 5 features into the tail simultaneously, which should
# isolate faster (shorter average path length) and recall meaningfully better than either single
# fault alone. Targets below are deliberately modest, not aspirational -- a diluted single-
# feature signal is a genuinely hard case for this feature set, not a free win.
SINGLE_FAULT_RECALL_TARGET = 0.30
COMBINED_FAULT_RECALL_TARGET = 0.50


# ---------------------------------------------------------------------------
# Injection
# ---------------------------------------------------------------------------

FAULT_TYPES = ["download_drop", "latency_spike", "combined", "gradual_decline"]


def _assign_injections(rng: np.random.Generator, eligible_ids: list[str],
                        eligible_temporal_ids: set[str], fraction: float) -> dict[str, str]:
    """10% of eligible zones, split as evenly as possible across the brief's four fault types
    (download -40% alone, latency +80% alone, both at once ["combined"], and the temporal
    gradual decline) -- temporal-eligible zones drawn from `eligible_temporal_ids` (>= 4 quarters
    of history) specifically, since a gradual-decline injection is meaningless without one."""
    n_inject = round(len(eligible_ids) * fraction)
    n_each = n_inject // 4
    counts = {t: n_each for t in FAULT_TYPES}
    for i in range(n_inject - n_each * 4):
        counts[FAULT_TYPES[i]] += 1

    counts["gradual_decline"] = min(counts["gradual_decline"], len(eligible_temporal_ids))
    temporal_ids = rng.choice(np.array(sorted(eligible_temporal_ids)), size=counts["gradual_decline"], replace=False)
    remaining_pool = rng.permutation([z for z in eligible_ids if z not in set(temporal_ids)])
    n_download, n_latency, n_combined = counts["download_drop"], counts["latency_spike"], counts["combined"]
    download_ids = remaining_pool[:n_download]
    latency_ids = remaining_pool[n_download:n_download + n_latency]
    combined_ids = remaining_pool[n_download + n_latency:n_download + n_latency + n_combined]

    assignment = {}
    for z in temporal_ids:
        assignment[z] = "gradual_decline"
    for z in download_ids:
        assignment[z] = "download_drop"
    for z in latency_ids:
        assignment[z] = "latency_spike"
    for z in combined_ids:
        assignment[z] = "combined"
    return assignment


def _apply_injections(df: pd.DataFrame, assignment: dict[str, str], quarter: str) -> pd.DataFrame:
    """Returns a modified copy of `df` (all quarters, all zones) with the injected faults
    applied, and `experience_index`/`peer_gap` recomputed via the real production formulas
    (src/compute_scores.py) so `exp_gap` for a spatially-injected zone genuinely reflects its
    perturbed download/latency, not an approximation of it."""
    df = df.copy()
    latest_mask = df["quarter"] == quarter
    # 'combined' zones get BOTH the download and latency fault applied at once (the brief's
    # fourth fault type) -- simplest possible way to express that: membership in both sets.
    download_ids = {z for z, t in assignment.items() if t in ("download_drop", "combined")}
    latency_ids = {z for z, t in assignment.items() if t in ("latency_spike", "combined")}
    temporal_ids = {z for z, t in assignment.items() if t == "gradual_decline"}

    df.loc[latest_mask & df["h3_cell"].isin(download_ids), "download_mbps"] *= 0.60   # -40%
    df.loc[latest_mask & df["h3_cell"].isin(latency_ids), "latency_effective_ms"] *= 1.80  # +80%

    # zone_priority.parquet already carries peer_group_median_experience/peer_gap/peer_gap_pct
    # from the real production run -- drop them first so compute_scores.peer_gap()'s merge
    # creates fresh columns instead of colliding with (and pandas-suffixing) the stale ones.
    df = df.drop(columns=["peer_group_median_experience", "peer_gap", "peer_gap_pct"], errors="ignore")
    df["experience_index"] = _recompute_experience_index(df)
    df.loc[df["insufficient_evidence"], "experience_index"] = np.nan
    df = _recompute_peer_gap(df)

    # Gradual decline: -8%/-16%/-24% off a single fixed anchor (the last quarter BEFORE the
    # decline starts), not each quarter scaled independently off its own noisy original value.
    # The latter (tried first) produced a barely-there signal: a zone whose real history was
    # already trending up or down by chance could see two of the three tapered points end up
    # closer together than 8% apart, since each was scaled off a different noisy starting
    # number. Anchoring to one stable point guarantees a clean, monotonic, exactly-8%-per-step
    # decline -- what "gradual relative decline" is actually supposed to look like -- and is
    # applied to experience_index directly (this is "Experience declining," not a
    # download/latency effect, so it isn't backed out through the raw metrics).
    for zid in temporal_ids:
        zone_rows = df[(df["h3_cell"] == zid) & df["experience_index"].notna()].sort_values("quarter")
        if len(zone_rows) < 4:
            continue
        last3 = zone_rows.index[-3:]
        anchor = zone_rows["experience_index"].iloc[-4]
        df.loc[last3, "experience_index"] = anchor * np.array(TAPER)

    df = df.drop(columns=["peer_group_median_experience", "peer_gap", "peer_gap_pct"], errors="ignore")
    df = _recompute_peer_gap(df)  # again -- the taper shifts peer medians too, however slightly
    return df


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------

def _split_half(ids: list[str], rng: np.random.Generator) -> tuple[set[str], set[str]]:
    shuffled = rng.permutation(ids)
    half = len(shuffled) // 2
    return set(shuffled[:half]), set(shuffled[half:])


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _prf(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0  # false-positive rate -- brief's own required metric
    return {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3),
            "fpr": round(fpr, 3), "n_flagged": int(y_pred.sum()), "n_true": int(y_true.sum()), "n": len(y_true)}


def _lof_flag(features: pd.DataFrame, contamination: float, n_neighbors: int = 20) -> np.ndarray:
    """LocalOutlierFactor(n_neighbors=20, contamination=0.08) -- mentor-specified, 2026-09-01.
    Transductive (no novelty=True), so fit_predict runs on the whole fit population at once,
    matching how Isolation Forest is fit here too."""
    X = StandardScaler().fit_transform(features)
    n_neighbors = min(n_neighbors, len(features) - 1)
    pred = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination).fit_predict(X)
    return pred == -1


# ---------------------------------------------------------------------------
# Benchmark A -- spatial (Peer Gap ML): download -40%, latency +80%, combined
# ---------------------------------------------------------------------------

SPATIAL_FAULT_TYPES = ("download_drop", "latency_spike", "combined")


def _per_fault_type_recall(assignment: dict, flag: pd.Series, idx) -> dict:
    """Isolation Forest recall broken down by fault type, hold-out only -- lets "single-fault
    signal is diluted across 5 features" (the SINGLE_FAULT_RECALL_TARGET reasoning above) be
    checked directly against real numbers instead of buried inside one averaged recall."""
    out = {}
    for fault in SPATIAL_FAULT_TYPES:
        fault_ids = [z for z in idx if assignment.get(z) == fault]
        out[fault] = round(float(flag.loc[fault_ids].mean()), 3) if fault_ids else None
    return out


def run_benchmark_a(df_mod: pd.DataFrame, assignment: dict, tuning_ids: set, holdout_ids: set) -> pd.DataFrame:
    spatial = build_spatial_features(df_mod)
    latest = spatial[spatial["quarter"] == QUARTER].set_index("h3_cell")
    features = latest[SPATIAL_FEATURE_COLS].dropna()

    y_true_all = pd.Series(
        {zid: int(assignment.get(zid) in SPATIAL_FAULT_TYPES) for zid in features.index}
    )

    iso_score, iso_flag = _fit_score_flag(features, contamination=CONTAMINATION)
    iso_flag = pd.Series(iso_flag, index=features.index)
    lof_flag = pd.Series(_lof_flag(features, contamination=CONTAMINATION), index=features.index)

    threshold_by_peer_group = df_mod[
        (df_mod["quarter"] == QUARTER) & (df_mod["evidence_tier"] == "full") & df_mod["h3_cell"].isin(tuning_ids)
    ].groupby("peer_group")["experience_index"].quantile(0.10)
    exp_by_zone = latest["exp_gap"].index.to_series().map(
        df_mod[df_mod["quarter"] == QUARTER].set_index("h3_cell")["experience_index"]
    )
    peer_by_zone = df_mod[df_mod["quarter"] == QUARTER].set_index("h3_cell")["peer_group"]
    baseline_flag = pd.Series(
        {zid: bool(exp_by_zone[zid] <= threshold_by_peer_group.get(peer_by_zone.get(zid), np.inf))
         for zid in features.index}
    )

    rows = []
    holdout_idx = features.index.intersection(holdout_ids)
    for half_name, half_ids in [("tuning", tuning_ids), ("hold-out", holdout_ids)]:
        idx = features.index.intersection(half_ids)
        y_true = y_true_all.loc[idx].to_numpy()
        for label, pred in [("Isolation Forest", iso_flag.loc[idx].to_numpy()),
                             ("Baseline (bottom-decile Experience, peer group)", baseline_flag.loc[idx].to_numpy()),
                             ("LOF", lof_flag.loc[idx].to_numpy())]:
            rows.append({"half": half_name, "detector": label, **_prf(y_true, pred)})
    per_fault_recall = _per_fault_type_recall(assignment, iso_flag, holdout_idx)
    return pd.DataFrame(rows), per_fault_recall


# ---------------------------------------------------------------------------
# Benchmark B -- temporal (gradual 3-quarter decline): no baseline exists for this shape
# ---------------------------------------------------------------------------

def run_benchmark_b(df_mod: pd.DataFrame, assignment: dict, tuning_ids: set, holdout_ids: set) -> pd.DataFrame:
    temporal = build_temporal_features(df_mod, min_quarters=MIN_QUARTERS_FOR_TEMPORAL)
    latest = temporal[temporal["quarter"] == QUARTER].set_index("h3_cell")
    features = latest[TEMPORAL_FEATURE_COLS].dropna()

    y_true_all = pd.Series(
        {zid: int(assignment.get(zid) == "gradual_decline") for zid in features.index}
    )

    iso_score, iso_flag = _fit_score_flag(features, contamination=CONTAMINATION)
    iso_flag = pd.Series(iso_flag, index=features.index)
    lof_flag = pd.Series(_lof_flag(features, contamination=CONTAMINATION), index=features.index)

    rows = []
    for half_name, half_ids in [("tuning", tuning_ids), ("hold-out", holdout_ids)]:
        idx = features.index.intersection(half_ids)
        y_true = y_true_all.loc[idx].to_numpy()
        for label, pred in [("Isolation Forest", iso_flag.loc[idx].to_numpy()),
                             ("LOF", lof_flag.loc[idx].to_numpy())]:
            rows.append({"half": half_name, "detector": label, **_prf(y_true, pred)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------

def test_t2_mentor_protocol():
    df = pd.read_parquet(ZONE_PRIORITY_PATH)
    rng = np.random.default_rng(SEED)

    eligible_ids = df[(df["quarter"] == QUARTER) & (df["evidence_tier"] == "full")]["h3_cell"].tolist()
    hist_counts = df[df["h3_cell"].isin(eligible_ids)].groupby("h3_cell")["experience_index"].apply(lambda s: s.notna().sum())
    eligible_temporal_ids = set(hist_counts[hist_counts >= MIN_QUARTERS_FOR_TEMPORAL].index)
    print(f"Eligible ('full' tier, {QUARTER}): {len(eligible_ids)} zones "
          f"({len(eligible_temporal_ids)} with >= {MIN_QUARTERS_FOR_TEMPORAL} quarters of history for the temporal model).")

    # Step 1: measure natural noise first.
    natural_temporal = build_temporal_features(df, min_quarters=MIN_QUARTERS_FOR_TEMPORAL)
    natural_noise_std = natural_temporal["qoq_rel"].std()
    print(f"Natural noise (std of qoq_rel across real, pre-injection zones): {natural_noise_std:.2f} pts")

    # Step 2: inject.
    assignment = _assign_injections(rng, eligible_ids, eligible_temporal_ids, INJECTION_FRACTION)
    n_by_type = pd.Series(assignment.values()).value_counts()
    print(f"Injected {len(assignment)} zones ({INJECTION_FRACTION:.0%} of eligible): {n_by_type.to_dict()}")
    df_mod = _apply_injections(df, assignment, QUARTER)

    injected_qoq_shift = build_temporal_features(df_mod, min_quarters=MIN_QUARTERS_FOR_TEMPORAL)
    injected_qoq_shift = injected_qoq_shift[
        (injected_qoq_shift["quarter"] == QUARTER)
        & injected_qoq_shift["h3_cell"].isin([z for z, t in assignment.items() if t == "gradual_decline"])
    ]["qoq_rel"]
    if len(injected_qoq_shift):
        print(f"Injected temporal zones' post-injection qoq_rel: mean {injected_qoq_shift.mean():.2f} pts "
              f"(clears 2x natural noise = {2*natural_noise_std:.2f}: {bool(abs(injected_qoq_shift.mean()) >= 2*natural_noise_std)})")

    # Step 3: split injected zones (and normal zones, for a realistic evaluation mix) 50/50.
    injected_ids = list(assignment.keys())
    normal_ids = [z for z in eligible_ids if z not in assignment]
    tuning_inj, holdout_inj = _split_half(injected_ids, rng)
    tuning_norm, holdout_norm = _split_half(normal_ids, rng)
    tuning_ids, holdout_ids = tuning_inj | tuning_norm, holdout_inj | holdout_norm
    print(f"Tuning: {len(tuning_ids)} zones ({len(tuning_inj)} injected) | "
          f"Hold-out: {len(holdout_ids)} zones ({len(holdout_inj)} injected)")

    print()
    print("=== Benchmark A: spatial / Peer Gap (download -40%, latency +80%, combined) ===")
    a, per_fault_recall = run_benchmark_a(df_mod, assignment, tuning_ids, holdout_ids)
    print(a.to_string(index=False))
    a_holdout = a[a["half"] == "hold-out"]
    for _, row in a_holdout.iterrows():
        for metric in ["precision", "recall", "f1", "fpr"]:
            assert 0.0 <= row[metric] <= 1.0, f"{metric} out of range: {row[metric]}"

    print()
    print("Isolation Forest recall by fault type (hold-out only):")
    for fault in SPATIAL_FAULT_TYPES:
        target = COMBINED_FAULT_RECALL_TARGET if fault == "combined" else SINGLE_FAULT_RECALL_TARGET
        value = per_fault_recall[fault]
        if value is None:
            print(f"  {fault:15} no hold-out zones of this type this run")
        else:
            met = "MET" if value >= target else "NOT MET"
            print(f"  {fault:15} recall={value:.3f}  target={target:.2f}  [{met}] "
                  f"(target set in advance -- see SINGLE_FAULT_RECALL_TARGET/COMBINED_FAULT_RECALL_TARGET)")

    print()
    print("=== Benchmark B: temporal (gradual -8%/quarter decline, 3 quarters) ===")
    b = run_benchmark_b(df_mod, assignment, tuning_ids, holdout_ids)
    print(b.to_string(index=False))
    b_holdout = b[b["half"] == "hold-out"]
    for _, row in b_holdout.iterrows():
        for metric in ["precision", "recall", "f1", "fpr"]:
            assert 0.0 <= row[metric] <= 1.0, f"{metric} out of range: {row[metric]}"

    print()
    print("=== Hold-out only (the reported result) ===")
    combined = pd.concat([a_holdout, b_holdout], ignore_index=True).drop(columns=["half"])
    print(combined.to_string(index=False))

    print()
    print("T2 (mentor protocol) ran end to end on real production feature-building/fit code "
          "(src/anomaly_detection.py), single fixed seed, hold-out never seen during tuning. "
          "Report these numbers as-is -- do not re-run with a different seed to chase a more "
          "favorable result.")


if __name__ == "__main__":
    test_t2_mentor_protocol()
    print("\nT2 PASSED (ran end to end; see printed metrics above for the actual result).")
