"""
T4 -- Confidence behaviour: construct a 500-tests/100-devices case and a 2-tests/1-device
case; the system must treat them differently and must label the sparse case as insufficient
evidence, not as poor performance (brief's own wording).

Calls the real src/compute_scores.py functions (this test is about verifying production
behaviour, not independently reimplementing it -- that's T3's job for trends).

Run: python -m tests.test_t4_confidence   (or `pytest tests/` if pytest is installed)
"""
from pathlib import Path

import pandas as pd

from src.compute_scores import add_effective_latency, add_quarters_observed, confidence_score, compute_evidence_tier

ZONE_PRIORITY_PATH = Path("data/processed/zone_priority.parquet")


def _make_case(tests: int, devices: int) -> pd.DataFrame:
    """Two rows with identical network conditions (download/upload/latency), differing only
    in evidence volume -- any difference in outcome must come from the evidence, not the
    network numbers."""
    return pd.DataFrame([{
        "h3_cell": "test_zone", "quarter": "2026Q2", "tests": tests, "devices": devices,
        "download_mbps": 300.0, "upload_mbps": 25.0, "latency_ms": 30.0, "latency_loaded_ms": 35.0,
    }])


def test_busy_vs_sparse_case_are_labeled_differently():
    busy = add_quarters_observed(add_effective_latency(_make_case(500, 100)))
    sparse = add_quarters_observed(add_effective_latency(_make_case(2, 1)))

    busy_score, sparse_score = confidence_score(busy), confidence_score(sparse)
    busy_tier, sparse_tier = compute_evidence_tier(busy), compute_evidence_tier(sparse)

    assert busy_tier.iloc[0] != "insufficient", "500 tests/100 devices must be classified, not insufficient"
    assert sparse_tier.iloc[0] == "insufficient", "2 tests/1 device must be labeled insufficient evidence"
    assert busy_score.iloc[0] > sparse_score.iloc[0], "busy case must score higher confidence"
    print(f"T4 constructed case: busy confidence={busy_score.iloc[0]} (tier={busy_tier.iloc[0]}), "
          f"sparse confidence={sparse_score.iloc[0]} (tier={sparse_tier.iloc[0]})")


def test_all_real_2test_1device_zones_are_insufficient():
    """Every real zone-quarter nationally with exactly 2 tests / 1 device must be flagged
    insufficient evidence -- never given a fabricated low score."""
    df = pd.read_parquet(ZONE_PRIORITY_PATH)
    matching = df[(df["tests"] == 2) & (df["devices"] == 1)]
    assert len(matching) > 0, "no real 2-test/1-device zones found -- test data may have changed"
    non_insufficient = matching[~matching["insufficient_evidence"]]
    assert non_insufficient.empty, f"{len(non_insufficient)} real 2-test/1-device zones were NOT flagged insufficient"
    print(f"T4 real-data check: {len(matching)}/{len(matching)} real 2-test/1-device zones labeled insufficient evidence.")


if __name__ == "__main__":
    test_busy_vs_sparse_case_are_labeled_differently()
    test_all_real_2test_1device_zones_are_insufficient()
    print("T4 PASSED.")
