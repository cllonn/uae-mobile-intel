"""
T1 -- Pipeline correctness: raw Ookla value -> processed dataset -> dashboard. Displayed
numbers must match the source (brief's own wording).

This architecture has no separate live API -- the dashboard is a static file with the scored
data built directly into it (see src/build_dashboard.py) -- so the trace is two hops, not the
brief's four: (1) zone_quarter_table.parquet (pre-score) -> zone_priority.parquet (post-score):
raw metrics must be untouched by scoring; (2) zone_priority.parquet -> the JSON embedded in
uae_dashboard.html: every displayed derived value must equal what the pipeline computed.

Run: python -m tests.test_t1_pipeline   (or `pytest tests/` if pytest is installed)
"""
import json
import random
import re
from pathlib import Path

import pandas as pd

ZONE_QUARTER_PATH = Path("data/processed/zone_quarter_table.parquet")
ZONE_PRIORITY_PATH = Path("data/processed/zone_priority.parquet")
DASHBOARD_PATH = Path("data/processed/uae_dashboard.html")
N_ZONES = 22
RAW_FIELDS = ["download_mbps", "upload_mbps", "tests", "devices"]
DASHBOARD_FIELDS = ["experience_index", "confidence_score", "priority_score", "trend_pts_per_qtr",
                    "deteriorating", "priority_zone"]


def _load_dashboard_zones() -> dict:
    html = DASHBOARD_PATH.read_text(encoding="utf-8")
    match = re.search(r"const ZONES_BY_QUARTER = (.*?);\n", html)
    assert match, "ZONES_BY_QUARTER not found in dashboard HTML -- did build_dashboard.py run?"
    return json.loads(match.group(1))


def test_raw_values_survive_scoring():
    """Hop 1: zone_quarter_table -> zone_priority. Scoring must never alter a raw metric."""
    pre = pd.read_parquet(ZONE_QUARTER_PATH).set_index(["h3_cell", "quarter"])
    post = pd.read_parquet(ZONE_PRIORITY_PATH).set_index(["h3_cell", "quarter"])

    random.seed(42)
    sample = random.sample(list(pre.index), N_ZONES)
    mismatches = []
    for key in sample:
        for field in RAW_FIELDS:
            pre_val, post_val = pre.loc[key, field], post.loc[key, field]
            if pd.isna(pre_val) and pd.isna(post_val):
                continue
            if pre_val != post_val:
                mismatches.append((key, field, pre_val, post_val))

    assert not mismatches, f"Raw values changed during scoring: {mismatches}"
    print(f"T1 hop 1/2: {N_ZONES} zones x {len(RAW_FIELDS)} fields = "
          f"{N_ZONES * len(RAW_FIELDS)} checks, 0 mismatches.")


def test_dashboard_matches_processed_dataset():
    """Hop 2: zone_priority.parquet -> the dashboard's embedded JSON."""
    df = pd.read_parquet(ZONE_PRIORITY_PATH)
    classified = df[~df["insufficient_evidence"]]
    zones_by_quarter = _load_dashboard_zones()

    random.seed(7)
    sample_idx = random.sample(list(classified.index), min(N_ZONES, len(classified)))
    mismatches = []
    checked = 0
    for idx in sample_idx:
        row = classified.loc[idx]
        cell, quarter = row["h3_cell"], row["quarter"]
        dash_zone = zones_by_quarter.get(quarter, {}).get(cell)
        if dash_zone is None:
            mismatches.append((cell, quarter, "missing from dashboard"))
            continue
        for field in DASHBOARD_FIELDS:
            checked += 1
            expected = row[field]
            actual = dash_zone[field]
            if isinstance(expected, bool) or isinstance(actual, bool):
                if bool(expected) != bool(actual):
                    mismatches.append((cell, quarter, field, expected, actual))
            elif pd.isna(expected):
                if actual is not None:
                    mismatches.append((cell, quarter, field, expected, actual))
            elif round(float(expected), 1) != round(float(actual), 1):
                mismatches.append((cell, quarter, field, expected, actual))

    assert not mismatches, f"Dashboard values don't match processed dataset: {mismatches}"
    print(f"T1 hop 2/2: {len(sample_idx)} zones x {len(DASHBOARD_FIELDS)} fields = "
          f"{checked} checks, 0 mismatches.")


if __name__ == "__main__":
    test_raw_values_survive_scoring()
    test_dashboard_matches_processed_dataset()
    print("T1 PASSED.")
