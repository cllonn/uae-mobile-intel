"""
Deterministic scoring pipeline: Experience Index, Confidence Score, and Peer Gap.

Every function here is a pure, deterministic transformation on a pandas DataFrame -- same
input, same output, every time. No ML and no LLM is involved anywhere in this file, per the
brief's rule that these scores must be traceable, reproducible, and explainable on a slide.

Input shape expected: `data/processed/zone_quarter_table.parquet` -- one row per
(H3 zone, quarter), with a `peer_group` column already attached by
`notebooks/09_peer_group_classifier.ipynb`.

Usage:
    import pandas as pd
    from src.compute_scores import score_zone_quarters

    zone_quarter = pd.read_parquet("data/processed/zone_quarter_table.parquet")
    scored = score_zone_quarters(zone_quarter)

See `notebooks/10_scores_and_peer_gap.ipynb` for how the CONFIG values below were chosen and
validated against the real national distribution, and for the sensitivity testing this file
doesn't do itself (it's meant to be imported, not run standalone).
"""
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG -- tunable decisions, calibrated against the real national distribution.
# Not hidden defaults: each one is a judgment call the brief asks us to justify.
# ---------------------------------------------------------------------------

# Experience Index weights: w_download + w_upload + w_latency must sum to 1.
# Draft starting point -- leans on download but keeps upload and latency meaningfully in play.
# See the notebook's sensitivity test for whether the ranking this produces is stable.
EXPERIENCE_WEIGHTS = {"download": 0.50, "upload": 0.20, "latency": 0.30}

# Confidence Score weights: how much each evidence signal counts toward trustworthiness.
CONFIDENCE_WEIGHTS = {"tests": 0.50, "devices": 0.30, "quarters": 0.20}

# Three-tier evidence gate (mentor-specified, 2026-09-01), replacing the old single
# tests>=N cutoff. This is the ONE authoritative eligibility rule for the whole project: every
# downstream module (Peer Gap, trends, anomaly detection, priority, the dashboard, the copilot)
# reads the `evidence_tier` / `insufficient_evidence` fields this produces rather than
# re-deriving its own threshold -- change these here and it propagates everywhere automatically.
#   - Below MIN_TESTS_INSUFFICIENT tests, OR below MIN_DEVICES_INSUFFICIENT devices: honest
#     output is "insufficient public evidence" -- no Experience Index at all, not just a low
#     score. Below ~10 tests the quarterly average is dominated by whoever happened to test.
#   - MIN_TESTS_INSUFFICIENT..MIN_TESTS_FULL-1 tests (and enough devices): Experience Index IS
#     computed and shown -- informative on the map -- but excluded from the Priority shortlist,
#     since it's unsafe to act on as a recommendation.
#   - MIN_TESTS_FULL+ tests: fully scored and shortlist-eligible.
# Confidence Score is a separate concept from this gate -- it still differentiates strength of
# evidence *among* zones that have a tier (a 30-test zone scores lower confidence than a
# 500-test zone); it just no longer decides tier membership on its own.
MIN_TESTS_INSUFFICIENT = 10
MIN_DEVICES_INSUFFICIENT = 3
MIN_TESTS_FULL = 30

# Backward-compat alias for older call sites (e.g. src/copilot_tools.py) that surfaced a single
# "reliable evidence" bar -- now means the full-tier bar specifically, not a general threshold.
MIN_TESTS_FOR_RELIABLE_EVIDENCE = MIN_TESTS_FULL

# Confidence Score saturation points (mentor-specified, 2026-09-01) -- the points beyond which
# more evidence stops changing belief much. 200 tests: well-measured zones sit comfortably
# above this. 50 devices: 500 tests from 3 phones is one person's experience, not a zone's;
# beyond ~50 distinct devices the protection against single-user skew saturates.
CONFIDENCE_TESTS_CAP = 200
CONFIDENCE_DEVICES_CAP = 50

# Backward-compat alias for older call sites/notebooks that imported the previous (log-scaled,
# cap=500) confidence saturation constant -- now means the linear tests cap specifically.
TESTS_SATURATION = CONFIDENCE_TESTS_CAP

# Total quarters in the dataset window (2024 Q3 -> 2026 Q2). Used to turn "how many of these
# quarters does this zone have any data in" into a 0-1 fraction for Confidence.
TOTAL_QUARTERS = 8


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def percentile_rank(series: pd.Series) -> pd.Series:
    """Percentile rank (0-1) -- mentor-approved replacement (2026-09-01) for min-max
    normalization, used everywhere a raw metric needs rescaling onto 0-1: Experience Index's
    three inputs here, and every Priority factor in `src/priority.py`.

    Why percentile rank over min-max: one zone with an extreme value would compress everything
    else into a narrow band under min-max scaling (this data has a handful of 1-2-test zones
    with fluke values -- e.g. download_mbps has a national median of ~336 but a max of ~2,337).
    Percentile rank is immune to that -- the extreme zone just ranks near 1.0, everyone else's
    relative ordering is untouched -- and it's easy to explain on a slide: 0.70 means the zone
    sits at the 70th percentile of what's actually being ranked against."""
    return series.rank(pct=True)


def add_effective_latency(df: pd.DataFrame) -> pd.DataFrame:
    """Loaded latency (`latency_loaded_ms`) where populated -- the brief's preferred metric,
    since it reflects latency under real network load -- falling back to unloaded latency
    (`latency_ms`, always populated) for the ~1% of zone-quarters missing it."""
    df = df.copy()
    df["latency_effective_ms"] = df["latency_loaded_ms"].fillna(df["latency_ms"])
    return df


def add_quarters_observed(df: pd.DataFrame, total_quarters: int = TOTAL_QUARTERS) -> pd.DataFrame:
    """How many of the dataset's quarters each zone has a MEANINGFUL measurement in -- the
    temporal-coverage signal Confidence needs. 'Meaningful' uses the same tests/devices bar as
    `compute_evidence_tier`'s 'insufficient' cutoff (tests >= MIN_TESTS_INSUFFICIENT and
    devices >= MIN_DEVICES_INSUFFICIENT) -- a quarter with a single stray test is real data but
    not a real observation of the zone's typical experience, and previously counted identically
    to a well-measured quarter (e.g. a zone could show "8/8 quarters observed" with only 4 of
    those quarters actually clearing the evidence bar the rest of this pipeline already applies
    everywhere else, inflating its Confidence Score's 20%-weighted quarters component). Same
    value repeated on every quarter-row of a given zone, since it describes the zone's overall
    track record, not just the current quarter. Computed inline here (not by reading
    `evidence_tier`) so this function stays independent of pipeline order --
    `score_zone_quarters` calls this before `compute_evidence_tier` runs."""
    df = df.copy()
    meaningful = (df["tests"] >= MIN_TESTS_INSUFFICIENT) & (df["devices"] >= MIN_DEVICES_INSUFFICIENT)
    df["quarters_observed"] = df.assign(_meaningful=meaningful).groupby("h3_cell")["_meaningful"].transform("sum")
    return df


# ---------------------------------------------------------------------------
# The two deterministic scores
# ---------------------------------------------------------------------------

def experience_index(df: pd.DataFrame, weights: dict = EXPERIENCE_WEIGHTS) -> pd.Series:
    """Experience = w_d*D + w_u*U + w_l*L, each percentile-ranked (0-1, mentor-specified
    2026-09-01) then combined, rescaled to 0-100. Latency is inverted first (lower ms = better)
    so higher always means better, for all three. Requires `add_effective_latency` to have
    already been run."""
    assert abs(sum(weights.values()) - 1.0) < 1e-6, "weights must sum to 1"
    d = percentile_rank(df["download_mbps"])
    u = percentile_rank(df["upload_mbps"])
    l = percentile_rank(-df["latency_effective_ms"])
    return (100 * (weights["download"] * d + weights["upload"] * u + weights["latency"] * l)).round(1)


def confidence_score(df: pd.DataFrame, weights: dict = CONFIDENCE_WEIGHTS,
                      tests_cap: int = CONFIDENCE_TESTS_CAP,
                      devices_cap: int = CONFIDENCE_DEVICES_CAP,
                      total_quarters: int = TOTAL_QUARTERS) -> pd.Series:
    """Confidence = 50*min(1, tests/200) + 30*min(1, devices/50) + 20*(quarters_observed/8)
    (mentor-specified, 2026-09-01) -- 0-100. Tests carries the most weight because sample size
    is the main driver of whether an average means anything. Devices is capped at 50 rather
    than compared as a devices/tests ratio: 500 tests from 3 phones is one person's experience,
    not a zone's, and beyond ~50 distinct devices the protection against that saturates.
    Quarters observed carries the least weight -- it says the zone is consistently measured,
    not that this quarter's number is right. A continuous strength-of-evidence signal,
    independent of the evidence_tier eligibility gate below -- e.g. a 30-test/3-device zone
    (tier 'full') can still score a middling confidence, and two 'low' tier zones (10 vs 28
    tests) are correctly told apart by this even though they share a tier. Requires
    `add_quarters_observed` to have already been run."""
    tests_component = (df["tests"] / tests_cap).clip(upper=1)
    devices_component = (df["devices"] / devices_cap).clip(upper=1)
    quarters_component = (df["quarters_observed"] / total_quarters).clip(upper=1)

    raw = 100 * (weights["tests"] * tests_component + weights["devices"] * devices_component
                 + weights["quarters"] * quarters_component)
    return raw.round(1)


def compute_evidence_tier(df: pd.DataFrame, min_tests_insufficient: int = MIN_TESTS_INSUFFICIENT,
                           min_devices_insufficient: int = MIN_DEVICES_INSUFFICIENT,
                           min_tests_full: int = MIN_TESTS_FULL) -> pd.Series:
    """Three-tier evidence eligibility gate (mentor-specified, 2026-09-01) -- see the CONFIG
    block above for the full rationale. Returns one of 'insufficient' / 'low' / 'full' per row:

      - 'insufficient': tests < min_tests_insufficient OR devices < min_devices_insufficient.
        No Experience Index at all -- shown grey, "insufficient public evidence."
      - 'low': enough devices and >= min_tests_insufficient tests, but < min_tests_full.
        Experience Index IS computed and shown, but excluded from the Priority shortlist.
      - 'full': >= min_tests_full tests (and enough devices, implied by not being insufficient).
        Fully scored and shortlist-eligible.

    This is the brief's Case A / Case B rule, extended to three states rather than two: 500
    tests/300 devices/8-of-8 quarters = 'full', trust it outright; 2 tests/1 device =
    'insufficient', full stop; something in between is shown but not acted on."""
    insufficient = (df["tests"] < min_tests_insufficient) | (df["devices"] < min_devices_insufficient)
    full = (~insufficient) & (df["tests"] >= min_tests_full)
    tier = pd.Series("low", index=df.index, dtype="object")
    tier[full] = "full"
    tier[insufficient] = "insufficient"
    return tier


# ---------------------------------------------------------------------------
# Peer Gap -- how a zone compares to its own peer group, this quarter
# ---------------------------------------------------------------------------

def peer_gap(df: pd.DataFrame) -> pd.DataFrame:
    """Adds `peer_group_median_experience`, `peer_gap` and `peer_gap_pct`: how far each zone's
    Experience Index sits from its peer group's median *in that same quarter*. Requires
    `experience_index`, `insufficient_evidence` and `peer_group` to already be on `df`.

    The peer-group median itself is computed from confidently-scored zones only -- an
    insufficient-evidence zone (Experience Index = NaN) can't be allowed to drag down the
    benchmark that other zones in its group get compared against."""
    df = df.copy()
    trustworthy = df.loc[~df["insufficient_evidence"]]
    peer_median = (
        trustworthy.groupby(["peer_group", "quarter"])["experience_index"]
        .median()
        .rename("peer_group_median_experience")
    )
    df = df.merge(peer_median, on=["peer_group", "quarter"], how="left")
    df["peer_gap"] = (df["experience_index"] - df["peer_group_median_experience"]).round(1)
    df["peer_gap_pct"] = (100 * df["peer_gap"] / df["peer_group_median_experience"]).round(1)
    return df


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def score_zone_quarters(df: pd.DataFrame) -> pd.DataFrame:
    """Runs every step in order: effective latency -> quarters observed -> Experience Index ->
    Confidence Score -> three-tier evidence gate -> Peer Gap. Returns a new DataFrame; the
    input `df` is never modified in place.

    `evidence_tier` ('insufficient' / 'low' / 'full') is the new authoritative eligibility
    field -- `insufficient_evidence` is kept alongside it as a derived boolean (True only for
    the 'insufficient' tier) purely so existing call sites that filter on "has an Experience
    Index at all" (peer_gap below, the dashboard, the copilot, most tests) don't need to change;
    they already mean "not insufficient," which is still correct for both 'low' and 'full'.
    Anything that specifically needs shortlist-eligible zones must filter on
    `evidence_tier == "full"` instead (see `src/priority.py`)."""
    df = add_effective_latency(df)
    df = add_quarters_observed(df)
    df["experience_index"] = experience_index(df)
    df["confidence_score"] = confidence_score(df)
    df["evidence_tier"] = compute_evidence_tier(df)
    df["insufficient_evidence"] = df["evidence_tier"] == "insufficient"
    df.loc[df["insufficient_evidence"], "experience_index"] = np.nan
    df = peer_gap(df)
    return df
