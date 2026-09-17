"""
T3 -- Trend correctness: independently recompute quarter-over-quarter change for random
zones; require 100% agreement with the platform (brief's own wording).

Deliberately does NOT import src/trends.py -- it reimplements the "2 consecutive
quarter-over-quarter national-gap declines" rule (mentor methodology, 2026-09-15: relative to
the NATIONAL median Experience Index, not peer group -- a zone must not be penalized just
because the whole country moved similarly), the continuous deterioration magnitude that rule
gates, and the 4-quarter least-squares slope, from scratch, from zone_priority.parquet's raw
columns, so a bug shared between the test and the production code wouldn't hide itself.

Run: python -m tests.test_t3_trends   (or `pytest tests/` if pytest is installed)
"""
import random
from pathlib import Path

import numpy as np
import pandas as pd

ZONE_PRIORITY_PATH = Path("data/processed/zone_priority.parquet")
QUARTER_ORDER = ["2024Q3", "2024Q4", "2025Q1", "2025Q2", "2025Q3", "2025Q4", "2026Q1", "2026Q2"]
CONSECUTIVE_DECLINES_REQUIRED = 2  # must match src/trends.py's constant of the same name
N_ZONES = 20


def _independent_deteriorating_flag_and_magnitude(national_gaps: list[float | None]) -> tuple[bool, float]:
    """(deteriorating, deterioration_magnitude) for the LAST quarter in the series -- True once a
    streak of CONSECUTIVE_DECLINES_REQUIRED consecutive quarter-over-quarter declines in
    (own Experience Index - national median Experience Index) closes on it; magnitude is the mean
    per-quarter decline size across that streak, 0 whenever the persistence gate isn't met."""
    streak = 0
    deltas: list[float] = []
    for i in range(1, len(national_gaps)):
        prev, cur = national_gaps[i - 1], national_gaps[i]
        if prev is not None and cur is not None and cur < prev:
            streak += 1
            deltas.append(prev - cur)
        else:
            streak = 0
            deltas = []
    deteriorating = streak >= CONSECUTIVE_DECLINES_REQUIRED
    magnitude = round(sum(deltas) / len(deltas), 2) if deteriorating else 0.0
    return deteriorating, magnitude


def _independent_slope(values: list[float | None], window: int = 4) -> float | None:
    recent = [v for v in values[-window:] if v is not None]
    idx = [i for i, v in enumerate(values[-window:]) if v is not None]
    if len(recent) < 2:
        return None
    return float(np.polyfit(idx, recent, 1)[0])


def test_deteriorating_flag_and_slope_agree_with_platform():
    df = pd.read_parquet(ZONE_PRIORITY_PATH)
    # Independent national reference -- median Experience Index per quarter across 'full'-tier
    # zones, recomputed directly from the loaded parquet (never src/trends.py's own copy).
    national = df.loc[df["evidence_tier"] == "full"].groupby("quarter")["experience_index"].median()

    full_history_zones = df.groupby("h3_cell")["quarter"].nunique()
    eligible = full_history_zones[full_history_zones == len(QUARTER_ORDER)].index.tolist()

    random.seed(3)
    sample = random.sample(eligible, min(N_ZONES, len(eligible)))

    flag_mismatches, magnitude_mismatches, slope_mismatches = [], [], []
    for cell in sample:
        rows = df[df["h3_cell"] == cell].set_index("quarter").reindex(QUARTER_ORDER)
        experience = [None if pd.isna(v) else float(v) for v in rows["experience_index"]]
        national_gaps = [
            None if e is None or pd.isna(national.get(q)) else e - float(national[q])
            for q, e in zip(QUARTER_ORDER, experience)
        ]

        expected_flag = bool(rows["deteriorating"].iloc[-1]) if not pd.isna(rows["insufficient_evidence"].iloc[-1]) else False
        actual_flag, actual_magnitude = _independent_deteriorating_flag_and_magnitude(national_gaps)
        if expected_flag != actual_flag:
            flag_mismatches.append((cell, expected_flag, actual_flag))

        expected_magnitude = round(float(rows["deterioration_magnitude"].iloc[-1]), 2)
        if expected_magnitude != actual_magnitude:
            magnitude_mismatches.append((cell, expected_magnitude, actual_magnitude))

        expected_slope = rows["trend_pts_per_qtr"].iloc[-1]
        actual_slope = _independent_slope(experience)
        if pd.isna(expected_slope) and actual_slope is None:
            continue
        if pd.isna(expected_slope) or actual_slope is None or round(expected_slope, 2) != round(actual_slope, 2):
            slope_mismatches.append((cell, expected_slope, actual_slope))

    assert not flag_mismatches, f"Deteriorating-flag disagreement: {flag_mismatches}"
    assert not magnitude_mismatches, f"Deterioration-magnitude disagreement: {magnitude_mismatches}"
    assert not slope_mismatches, f"Trend-slope disagreement: {slope_mismatches}"
    print(f"T3: {len(sample)} zones, deteriorating-flag agreement {len(sample)}/{len(sample)}, "
          f"deterioration-magnitude agreement {len(sample)}/{len(sample)}, "
          f"trend-slope agreement {len(sample)}/{len(sample)}.")


if __name__ == "__main__":
    test_deteriorating_flag_and_slope_agree_with_platform()
    print("T3 PASSED.")
