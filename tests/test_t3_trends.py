"""
T3 -- Trend correctness: independently recompute quarter-over-quarter change for random
zones; require 100% agreement with the platform (brief's own wording).

Deliberately does NOT import src/trends.py -- it reimplements the "3 consecutive
quarter-over-quarter Peer Gap declines" rule and the 4-quarter least-squares slope from
scratch, from zone_priority.parquet's raw columns, so a bug shared between the test and the
production code wouldn't hide itself.

Run: python -m tests.test_t3_trends   (or `pytest tests/` if pytest is installed)
"""
import random
from pathlib import Path

import numpy as np
import pandas as pd

ZONE_PRIORITY_PATH = Path("data/processed/zone_priority.parquet")
QUARTER_ORDER = ["2024Q3", "2024Q4", "2025Q1", "2025Q2", "2025Q3", "2025Q4", "2026Q1", "2026Q2"]
CONSECUTIVE_DECLINES_REQUIRED = 3  # must match src/trends.py's constant of the same name
N_ZONES = 20


def _independent_deteriorating_flag(peer_gaps: list[float | None]) -> bool:
    """True if the LAST quarter in the series closes a streak of
    CONSECUTIVE_DECLINES_REQUIRED consecutive quarter-over-quarter declines."""
    streak = 0
    for i in range(1, len(peer_gaps)):
        prev, cur = peer_gaps[i - 1], peer_gaps[i]
        if prev is not None and cur is not None and cur < prev:
            streak += 1
        else:
            streak = 0
    return streak >= CONSECUTIVE_DECLINES_REQUIRED


def _independent_slope(values: list[float | None], window: int = 4) -> float | None:
    recent = [v for v in values[-window:] if v is not None]
    idx = [i for i, v in enumerate(values[-window:]) if v is not None]
    if len(recent) < 2:
        return None
    return float(np.polyfit(idx, recent, 1)[0])


def test_deteriorating_flag_and_slope_agree_with_platform():
    df = pd.read_parquet(ZONE_PRIORITY_PATH)
    full_history_zones = df.groupby("h3_cell")["quarter"].nunique()
    eligible = full_history_zones[full_history_zones == len(QUARTER_ORDER)].index.tolist()

    random.seed(3)
    sample = random.sample(eligible, min(N_ZONES, len(eligible)))

    flag_mismatches, slope_mismatches = [], []
    for cell in sample:
        rows = df[df["h3_cell"] == cell].set_index("quarter").reindex(QUARTER_ORDER)
        peer_gaps = [None if pd.isna(v) else float(v) for v in rows["peer_gap"]]
        experience = [None if pd.isna(v) else float(v) for v in rows["experience_index"]]

        expected_flag = bool(rows["deteriorating"].iloc[-1]) if not pd.isna(rows["insufficient_evidence"].iloc[-1]) else False
        actual_flag = _independent_deteriorating_flag(peer_gaps)
        if expected_flag != actual_flag:
            flag_mismatches.append((cell, expected_flag, actual_flag))

        expected_slope = rows["trend_pts_per_qtr"].iloc[-1]
        actual_slope = _independent_slope(experience)
        if pd.isna(expected_slope) and actual_slope is None:
            continue
        if pd.isna(expected_slope) or actual_slope is None or round(expected_slope, 2) != round(actual_slope, 2):
            slope_mismatches.append((cell, expected_slope, actual_slope))

    assert not flag_mismatches, f"Deteriorating-flag disagreement: {flag_mismatches}"
    assert not slope_mismatches, f"Trend-slope disagreement: {slope_mismatches}"
    print(f"T3: {len(sample)} zones, deteriorating-flag agreement {len(sample)}/{len(sample)}, "
          f"trend-slope agreement {len(sample)}/{len(sample)}.")


if __name__ == "__main__":
    test_deteriorating_flag_and_slope_agree_with_platform()
    print("T3 PASSED.")
