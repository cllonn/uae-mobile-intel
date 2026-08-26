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

# Below this many tests in a quarter, the honest output is "insufficient public evidence" --
# no Experience/Priority classification at all, not just a low score. This is the ONE
# authoritative eligibility gate for the whole project: every downstream module (Peer Gap,
# trends, anomaly detection, priority, the dashboard, the copilot) reads the `insufficient_
# evidence` flag this constant produces rather than re-deriving its own threshold -- change
# it here and it propagates everywhere automatically. Set to 30 to match
# notebooks/04_h3_resolution_choice.ipynb and notebooks/06_first_uae_map.ipynb's own evidence
# bar (both independently chose 30 as "sufficiently sampled"), not the earlier placeholder of
# 5 (which flagged only ~half of zone-quarters as insufficient and let the median-adjacent
# tail of very thinly-tested zones get a full classification anyway). Confidence Score is a
# separate concept from this gate -- it still differentiates strength of evidence *among*
# zones that clear this bar (a 30-test zone scores lower confidence than a 500-test zone),
# it just no longer decides whether a zone is classified at all.
MIN_TESTS_FOR_RELIABLE_EVIDENCE = 30

# Test-count saturation point for the Confidence formula: zones at or above this many
# tests/quarter get full marks on the test-volume component. Set at the brief's own cited
# benchmark for a well-measured zone ("top zones exceed 500").
TESTS_SATURATION = 500

# Total quarters in the dataset window (2024 Q3 -> 2026 Q2). Used to turn "how many of these
# quarters does this zone have any data in" into a 0-1 fraction for Confidence.
TOTAL_QUARTERS = 8


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def normalize_minmax(series: pd.Series, low_pct: float = 1, high_pct: float = 99) -> pd.Series:
    """Rescale a column onto 0-1, same method every run (reproducibility).

    Clips to the 1st/99th percentile *before* rescaling: on the real data, a handful of
    zone-quarters with only 1-2 tests produce fluke extreme values (e.g. download_mbps has a
    national median of ~336 but a max of ~2,337). Without clipping, those few outliers would
    single-handedly compress every other zone's score into a narrow band near zero.
    """
    lo, hi = series.quantile(low_pct / 100), series.quantile(high_pct / 100)
    if hi == lo:
        return pd.Series(0.5, index=series.index)
    return ((series - lo) / (hi - lo)).clip(0, 1)


def add_effective_latency(df: pd.DataFrame) -> pd.DataFrame:
    """Loaded latency (`latency_loaded_ms`) where populated -- the brief's preferred metric,
    since it reflects latency under real network load -- falling back to unloaded latency
    (`latency_ms`, always populated) for the ~1% of zone-quarters missing it."""
    df = df.copy()
    df["latency_effective_ms"] = df["latency_loaded_ms"].fillna(df["latency_ms"])
    return df


def add_quarters_observed(df: pd.DataFrame, total_quarters: int = TOTAL_QUARTERS) -> pd.DataFrame:
    """How many of the dataset's quarters each zone has any measurement in -- the temporal-
    coverage signal Confidence needs. Same value repeated on every quarter-row of a given zone,
    since it describes the zone's overall track record, not just the current quarter."""
    df = df.copy()
    df["quarters_observed"] = df.groupby("h3_cell")["quarter"].transform("nunique")
    return df


# ---------------------------------------------------------------------------
# The two deterministic scores
# ---------------------------------------------------------------------------

def experience_index(df: pd.DataFrame, weights: dict = EXPERIENCE_WEIGHTS) -> pd.Series:
    """Experience = w_d*D + w_u*U + w_l*L, each normalized 0-1 then combined, rescaled to 0-100.
    Latency is inverted first (lower ms = better) so higher always means better, for all three.
    Requires `add_effective_latency` to have already been run."""
    assert abs(sum(weights.values()) - 1.0) < 1e-6, "weights must sum to 1"
    d = normalize_minmax(df["download_mbps"])
    u = normalize_minmax(df["upload_mbps"])
    l = normalize_minmax(-df["latency_effective_ms"])
    return (100 * (weights["download"] * d + weights["upload"] * u + weights["latency"] * l)).round(1)


def confidence_score(df: pd.DataFrame, weights: dict = CONFIDENCE_WEIGHTS,
                      min_tests: int = MIN_TESTS_FOR_RELIABLE_EVIDENCE,
                      tests_cap: int = TESTS_SATURATION,
                      total_quarters: int = TOTAL_QUARTERS):
    """Confidence = f(tests, devices, quarters observed). Returns (score 0-100, insufficient
    flag). 'Insufficient' zones get no Experience/Priority classification at all -- a status,
    not just a low number, per the brief's Case A / Case B example: 500 tests/300 devices/8-of-8
    quarters = trust it; 2 tests/1 device = insufficient evidence, full stop.
    Requires `add_quarters_observed` to have already been run."""
    # Log scale: a handful of zones have thousands of tests while the median has ~4 -- a linear
    # scale would let the busiest zones swamp the comparison.
    tests_component = np.clip(np.log1p(df["tests"]) / np.log1p(tests_cap), 0, 1)
    # devices/tests near 1 = mostly distinct testers (good); near 0 = a few phones testing
    # repeatedly (risk of single-user skew) -- the brief's own "500 tests from 3 phones is not
    # the same as 500 tests from 300 phones" example.
    device_ratio = (df["devices"] / df["tests"]).clip(0, 1)
    quarters_component = (df["quarters_observed"] / total_quarters).clip(0, 1)

    raw = 100 * (weights["tests"] * tests_component + weights["devices"] * device_ratio
                 + weights["quarters"] * quarters_component)
    insufficient = df["tests"] < min_tests
    return raw.round(1), insufficient


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
    Confidence Score -> insufficient-evidence rule -> Peer Gap. Returns a new DataFrame; the
    input `df` is never modified in place."""
    df = add_effective_latency(df)
    df = add_quarters_observed(df)
    df["experience_index"] = experience_index(df)
    df["confidence_score"], df["insufficient_evidence"] = confidence_score(df)
    df.loc[df["insufficient_evidence"], "experience_index"] = np.nan
    df = peer_gap(df)
    return df
