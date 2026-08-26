"""
T5 -- Priority sensitivity: perturb priority weights by +-10-20%; if the top-ranked areas
reshuffle dramatically, the methodology is fragile (brief's own wording).

Re-runs the real src/priority.py::add_priority on the real pipeline output with each weight
perturbed +-15% in turn (renormalized to sum to 1), and compares the resulting ranking to the
original via Spearman correlation and top-20 overlap.

Run: python -m tests.test_t5_priority_sensitivity   (or `pytest tests/` if pytest is installed)
"""
from pathlib import Path

import pandas as pd

from src.priority import PRIORITY_WEIGHTS, add_priority

ZONE_PRIORITY_PATH = Path("data/processed/zone_priority.parquet")
PERTURBATION = 0.15
TOP_N = 20
MIN_SPEARMAN = 0.95  # below this, the brief would call the ranking fragile
MIN_TOP_N_OVERLAP = 0.75


def _perturbed_weights(key: str, direction: float) -> dict:
    weights = dict(PRIORITY_WEIGHTS)
    weights[key] = weights[key] * (1 + direction)
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}


def test_priority_ranking_is_stable_under_weight_perturbation():
    df = pd.read_parquet(ZONE_PRIORITY_PATH)
    latest_quarter = df["quarter"].max()
    # zone_priority.parquet already carries priority_score etc. from the production run --
    # drop them first so add_priority() recomputes fresh instead of merge-suffixing next to
    # the existing columns (mirrors the column set run_pipeline.py passes in, pre-priority).
    priority_cols = ["priority_score", "priority_zone", "factor_peer_gap", "factor_ml_anomaly",
                     "factor_deterioration", "factor_population", "peer_gap_band", "ml_anomaly_band",
                     "deterioration_band", "population_band"]
    df = df.drop(columns=[c for c in priority_cols if c in df.columns])
    original = add_priority(df)
    original_latest = original[(original["quarter"] == latest_quarter) & (~original["insufficient_evidence"])]
    original_ranked = original_latest.sort_values("priority_score", ascending=False)
    original_top_n = set(original_ranked.head(TOP_N)["h3_cell"])

    results = []
    for key in PRIORITY_WEIGHTS:
        for direction in (+PERTURBATION, -PERTURBATION):
            perturbed = add_priority(df, weights=_perturbed_weights(key, direction))
            perturbed_latest = perturbed[(perturbed["quarter"] == latest_quarter) & (~perturbed["insufficient_evidence"])]

            merged = original_latest[["h3_cell", "priority_score"]].merge(
                perturbed_latest[["h3_cell", "priority_score"]], on="h3_cell", suffixes=("_orig", "_pert")
            )
            spearman = merged["priority_score_orig"].corr(merged["priority_score_pert"], method="spearman")

            perturbed_top_n = set(perturbed_latest.sort_values("priority_score", ascending=False).head(TOP_N)["h3_cell"])
            overlap = len(original_top_n & perturbed_top_n) / TOP_N

            results.append({"weight": key, "direction": f"{direction:+.0%}", "spearman": round(spearman, 4),
                            "top20_overlap": round(overlap, 2)})

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    min_spearman = results_df["spearman"].min()
    min_overlap = results_df["top20_overlap"].min()
    assert min_spearman >= MIN_SPEARMAN, f"Spearman correlation dropped to {min_spearman} under perturbation -- ranking is fragile"
    assert min_overlap >= MIN_TOP_N_OVERLAP, f"Top-{TOP_N} overlap dropped to {min_overlap} under perturbation -- ranking is fragile"
    print(f"\nT5: worst-case Spearman={min_spearman}, worst-case top-{TOP_N} overlap={min_overlap} "
          f"across {len(results)} +-{PERTURBATION:.0%} perturbations -- ranking is stable.")


if __name__ == "__main__":
    test_priority_ranking_is_stable_under_weight_perturbation()
    print("T5 PASSED.")
