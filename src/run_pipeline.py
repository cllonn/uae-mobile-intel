"""
Runs the full deterministic + ML pipeline end to end: zone_quarter_table.parquet ->
Experience/Confidence/Peer Gap -> trend/deterioration -> ML anomaly detection (+ baseline
comparison) -> Priority engine -> zone_priority.parquet.

This is the one script to re-run whenever an upstream input changes (a new Ookla quarter, a
reclassified peer group, a tuned weight) -- everything downstream, including the dashboard
(`build_dashboard.py`), reads from its output rather than recomputing anything itself.

Run: `python -m src.run_pipeline` from the repo root.
"""
from pathlib import Path

import pandas as pd

from src.compute_scores import score_zone_quarters
from src.trends import add_trend
from src.anomaly_detection import (
    add_peer_gap_ml, add_temporal_anomaly_ml, add_baseline_bottom_decile, compare_flags_to_baseline,
)
from src.priority import add_priority

ZONE_QUARTER_PATH = Path("data/processed/zone_quarter_table.parquet")
OUTPUT_PATH = Path("data/processed/zone_priority.parquet")


def main():
    zq = pd.read_parquet(ZONE_QUARTER_PATH)
    print(f"Input: {len(zq)} zone-quarter rows")

    df = score_zone_quarters(zq)
    print(f"Scores:   {(~df['insufficient_evidence']).sum()} classified of {len(df)}")

    df = add_trend(df)
    print(f"Trend:    {df['deteriorating'].sum()} zone-quarters flagged deteriorating "
          f"({df.loc[df['deteriorating'], 'h3_cell'].nunique()} unique zones, any quarter)")

    df = add_peer_gap_ml(df)
    df = add_temporal_anomaly_ml(df)
    df = add_baseline_bottom_decile(df)
    print(f"Anomaly:  Peer Gap ML flagged {df['peer_gap_ml_anomaly'].sum()}, "
          f"Temporal Anomaly ML flagged {df['temporal_anomaly_ml_flag'].sum()}, "
          f"baseline (bottom-decile download) flagged {df['baseline_bottom_decile_flag'].sum()}")
    print(compare_flags_to_baseline(df).to_string(index=False))

    df = add_priority(df)
    latest = df[df["quarter"] == df["quarter"].max()]
    classified_latest = latest[~latest["insufficient_evidence"]]
    print(f"Priority: {classified_latest['priority_zone'].sum()} priority zones flagged "
          f"in the latest quarter ({len(classified_latest)} classified)")

    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved: {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size / 1024:.1f} KB, {len(df)} rows)")


if __name__ == "__main__":
    main()
