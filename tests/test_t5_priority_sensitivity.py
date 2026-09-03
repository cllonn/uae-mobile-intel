"""
T5 -- Priority sensitivity, mentor's exact protocol (2026-09-01). Supersedes the earlier
ad-hoc +-15% Spearman version entirely -- different variant set, different metrics (top-10
overlap + Kendall's tau, not Spearman), different reporting shape. Do not compare numbers
across the two; they are not the same test.

Two variant families, both perturbing `src/priority.py::PRIORITY_WEIGHTS`
([0.35, 0.25, 0.20, 0.20] = peer_gap, deterioration, temporal_anomaly, population):

  1. 16 one-at-a-time variants: multiply one weight by 0.8, 0.9, 1.1 or 1.2 in turn, renormalize
     the four weights to sum to 1. Tells you which single weight the ranking is most sensitive to.
  2. ~50 random draws from a Dirichlet distribution centred on the baseline (mean = baseline by
     construction; concentration chosen so the typical per-weight spread is in the same ballpark
     as the one-at-a-time +-20% range -- see DIRICHLET_CONCENTRATION below). Tests COMBINATIONS
     of changes at once, which is where instability usually hides that one-at-a-time misses.

Per variant: top-10 overlap with the baseline top-10, and Kendall's tau between the full
priority-score rankings (scipy.stats.kendalltau) -- both computed over `evidence_tier=="full"`
zones in the latest quarter only, the real Priority shortlist population (NOT
`~insufficient_evidence`, which would pull in 'low' tier zones that never get a Priority Score
at all post the 2026-09-01 evidence-tier change).

Zones that stay in the top 10 across at least 90% of the 66 variants are reported as the robust
shortlist -- present those on pitch day.

Run: python -m tests.test_t5_priority_sensitivity   (or `pytest tests/` if pytest is installed)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from src.priority import PRIORITY_WEIGHTS, add_priority

ZONE_PRIORITY_PATH = Path("data/processed/zone_priority.parquet")

BASELINE_WEIGHTS = [0.35, 0.25, 0.20, 0.20]  # peer_gap, deterioration, temporal_anomaly, population -- mentor-specified order
FACTOR_KEYS = ["peer_gap", "deterioration", "temporal_anomaly", "population"]
ONE_AT_A_TIME_MULTIPLIERS = [0.8, 0.9, 1.1, 1.2]
N_DIRICHLET_DRAWS = 50
DIRICHLET_SEED = 42
# alpha = baseline * this. Dirichlet(alpha) has mean = baseline exactly regardless of this
# scalar; it controls spread only. 40 was chosen so the per-weight standard deviation lands
# roughly in the same range as the one-at-a-time +-20% perturbations (~0.06-0.07 absolute on
# a 0.20-0.35 baseline weight) -- comparable "typical" perturbation size across both variant
# families, not an arbitrary pick.
DIRICHLET_CONCENTRATION = 40

TOP_N = 10
ROBUST_SHORTLIST_THRESHOLD = 0.90


def _renormalized(weights: dict) -> dict:
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}


def _one_at_a_time_variants() -> list[dict]:
    variants = []
    for key in FACTOR_KEYS:
        for mult in ONE_AT_A_TIME_MULTIPLIERS:
            w = dict(zip(FACTOR_KEYS, BASELINE_WEIGHTS))
            w[key] = w[key] * mult
            variants.append({
                "label": f"{key} x{mult}",
                "weights": _renormalized(w),
            })
    return variants


def _dirichlet_variants() -> list[dict]:
    rng = np.random.default_rng(DIRICHLET_SEED)
    alpha = np.array(BASELINE_WEIGHTS) * DIRICHLET_CONCENTRATION
    draws = rng.dirichlet(alpha, size=N_DIRICHLET_DRAWS)  # each row already sums to 1
    return [
        {"label": f"dirichlet_{i+1:02d}", "weights": dict(zip(FACTOR_KEYS, draw))}
        for i, draw in enumerate(draws)
    ]


def _rank_latest_full_tier(df: pd.DataFrame, weights: dict, latest_quarter: str) -> pd.DataFrame:
    """Runs add_priority with `weights`, restricts to the real Priority-shortlist population
    (latest quarter, evidence_tier == 'full'), sorted by priority_score descending."""
    scored = add_priority(df, weights=weights)
    latest = scored[(scored["quarter"] == latest_quarter) & (scored["evidence_tier"] == "full")]
    return latest.sort_values("priority_score", ascending=False)[["h3_cell", "priority_score"]]


def test_priority_ranking_mentor_protocol():
    assert list(PRIORITY_WEIGHTS.values()) == BASELINE_WEIGHTS, (
        "PRIORITY_WEIGHTS in src/priority.py no longer matches the mentor's baseline "
        "[0.35, 0.25, 0.20, 0.20] -- this test's baseline would silently stop testing production."
    )

    df = pd.read_parquet(ZONE_PRIORITY_PATH)
    latest_quarter = df["quarter"].max()
    # zone_priority.parquet already carries priority_score etc. from the production run --
    # drop them first so add_priority() recomputes fresh instead of merge-suffixing next to
    # the existing columns (mirrors the column set run_pipeline.py passes in, pre-priority).
    priority_cols = ["priority_score", "priority_zone", "factor_peer_gap", "factor_deterioration",
                     "factor_temporal_anomaly", "factor_population", "peer_gap_band",
                     "deterioration_band", "temporal_anomaly_band", "population_band"]
    df = df.drop(columns=[c for c in priority_cols if c in df.columns])

    baseline = _rank_latest_full_tier(df, dict(zip(FACTOR_KEYS, BASELINE_WEIGHTS)), latest_quarter)
    baseline_top10 = set(baseline.head(TOP_N)["h3_cell"])
    print(f"Baseline: {len(baseline)} full-tier zones ranked, {latest_quarter}.")

    variants = _one_at_a_time_variants() + _dirichlet_variants()
    assert len(variants) == 16 + N_DIRICHLET_DRAWS

    results = []
    top10_membership: dict[str, int] = {}  # h3_cell -> how many variants it lands in the top 10

    for variant in variants:
        ranked = _rank_latest_full_tier(df, variant["weights"], latest_quarter)
        merged = baseline.merge(ranked, on="h3_cell", suffixes=("_base", "_var"))
        assert len(merged) == len(baseline), (
            f"variant '{variant['label']}' scored a different set of zones than baseline -- "
            "evidence_tier gating should be weight-independent"
        )
        tau = kendalltau(merged["priority_score_base"], merged["priority_score_var"]).correlation

        variant_top10 = set(ranked.head(TOP_N)["h3_cell"])
        overlap = len(baseline_top10 & variant_top10)

        for h3_cell in variant_top10:
            top10_membership[h3_cell] = top10_membership.get(h3_cell, 0) + 1

        results.append({"variant": variant["label"], "top10_overlap": f"{overlap}/{TOP_N}", "kendall_tau": round(tau, 4)})

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    taus = results_df["kendall_tau"]
    overlaps = results_df["top10_overlap"].str.split("/").str[0].astype(int)
    n_variants = len(variants)

    robust_shortlist = sorted(
        h3_cell for h3_cell, count in top10_membership.items()
        if count / n_variants >= ROBUST_SHORTLIST_THRESHOLD
    )

    print(f"\nMean Kendall's tau: {taus.mean():.4f} (min {taus.min():.4f}, max {taus.max():.4f})")
    print(f"Top-10 overlap: mean {overlaps.mean():.1f}/{TOP_N}, worst-case {overlaps.min()}/{TOP_N}")
    print(f"Robust shortlist (top 10 in >={ROBUST_SHORTLIST_THRESHOLD:.0%} of {n_variants} variants): "
          f"{len(robust_shortlist)} zones -- {robust_shortlist}")
    print(f"({len(baseline_top10 & set(robust_shortlist))}/{len(robust_shortlist)} of the robust "
          f"shortlist also sits in the baseline top {TOP_N}.)")

    # This test reports honestly rather than gating on an arbitrary stability bar (the mentor's
    # own instruction: fragility, found and reported, is a legitimate result). What it does
    # assert is that the protocol itself ran correctly on real production weights/data.
    assert taus.between(-1, 1).all(), "Kendall's tau out of valid range -- computation bug"
    assert len(variants) == 66
    print("\nT5 (mentor protocol) ran end to end -- see the honest conclusion above the assert line.")


if __name__ == "__main__":
    test_priority_ranking_mentor_protocol()
    print("T5 PASSED.")
