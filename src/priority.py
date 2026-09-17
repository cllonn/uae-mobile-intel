"""
Geographic prioritization engine: Priority = f(PeerGap, Deterioration, TemporalAnomaly,
PopulationExposure, Confidence).

Formula and weights below are mentor-specified (2026-09-01), sensitivity-tested per the brief's
mandatory T5 (perturb weights +-10-20%, report whether the ranking reshuffles). The one rule
that isn't a judgment call, fixed by the brief itself: a low-confidence observation must never
automatically become a high-priority recommendation. Implemented as a confidence **gate**, not
a linear multiplier: `confidence_gate` returns 0 outright below a minimum confidence, and a
compressed 0.7-1.0 range above it, so confidence tunes the ranking without dominating it once a
zone has cleared the bar -- see `confidence_gate` below for why that's not just "multiply by
confidence/100."

Input: the output of `trends.add_trend` + `anomaly_detection.add_peer_gap_ml` +
`anomaly_detection.add_temporal_anomaly_ml`.
"""
import numpy as np
import pandas as pd

from src.compute_scores import percentile_rank

PRIORITY_WEIGHTS = {
    "peer_gap": 0.35,          # how much worse than peers, right now -- the most direct evidence something is wrong
    "deterioration": 0.25,     # persistent decline vs. peer trend -- a widening gap is more actionable than a static one
    "temporal_anomaly": 0.20,  # unusual vs. the zone's OWN history (Isolation Forest) -- catches what a simple slope misses
    "population": 0.20,        # how many people are exposed to it -- tie-breaker between comparably weak zones
}

# Confidence gate G(C) (mentor-specified, 2026-09-01): 0 below this confidence -- a zone we
# barely measured can never appear in the shortlist at all, however bad it looks.
CONFIDENCE_GATE_THRESHOLD = 40

# Top this fraction of classified zones, by priority score (within each quarter), get flagged
# "investigate first."
PRIORITY_ZONE_FRACTION = 0.10


def confidence_gate(confidence: pd.Series, threshold: float = CONFIDENCE_GATE_THRESHOLD) -> pd.Series:
    """G(C) = 0 if C < 40, else 0.5 + 0.5*(C/100) -- mentor-specified, 2026-09-01.

    Why the gate sits at 40 and the multiplier isn't linear: below 40, a zone we barely measured
    can never appear in the shortlist at all, however bad it looks -- a hard cutoff, not a small
    number. Above 40 the multiplier runs 0.7 to 1.0 rather than 0 to 1, so confidence *tunes*
    the ranking without dominating it -- past the gate, what's being ranked is severity, not
    evidence volume. A linear 0-to-1 multiplier would push every merely-adequate zone far down
    the list purely for being less measured, which is not the rule the brief actually wants."""
    gated = 0.5 + 0.5 * (confidence / 100)
    return gated.where(confidence >= threshold, 0.0)


def _factor_band(series: pd.Series) -> pd.Series:
    """Buckets a 0-100 factor into Low/Medium/High for the 'why this priority' display --
    fixed tercile cut points, not a per-zone judgment call baked into the number."""
    return pd.cut(series, bins=[-0.01, 33, 66, 100], labels=["Low", "Medium", "High"])


def _deterioration_factor(magnitude: pd.Series) -> pd.Series:
    """0 (Low band, via `_factor_band`'s cut points) for every zone with zero deterioration
    magnitude -- assigned directly, never percentile-ranked into whatever the tied-zero
    cluster's average rank happens to be. Positive magnitudes (zones that cleared the
    persistence gate) are percentile-ranked ONLY among themselves, so severity spreads across
    the full 0-100 range based on how their relative decline compares to other currently-
    deteriorating zones, rather than every one of them tying at the same value."""
    factor = pd.Series(0.0, index=magnitude.index)
    positive = magnitude > 0
    if positive.any():
        factor.loc[positive] = 100 * percentile_rank(magnitude.loc[positive])
    return factor


def add_priority(df: pd.DataFrame, weights: dict = PRIORITY_WEIGHTS) -> pd.DataFrame:
    """Adds `priority_score` (0-100), `priority_zone` (bool, top decile per quarter), the four
    underlying `factor_*` columns (0-100 each, percentile rank x100), and their Low/Medium/High
    bands. Requires `peer_gap`, `deterioration_magnitude`, `population`, `confidence_score`,
    `temporal_anomaly_ml_score` and `evidence_tier` already on `df`. Returns a new DataFrame.

    Shortlist eligibility (mentor-specified, 2026-09-01): only `evidence_tier == "full"` zones
    are candidates here -- NOT every zone with an Experience Index. A 'low' tier zone (10-29
    tests) has a real, displayed Experience Index but is explicitly "unsafe as a recommendation"
    per the brief, so it gets no Priority Score at all (NaN, not a low one) and can never be
    `priority_zone`, however bad its raw numbers look -- the same "no score is better than a
    fabricated one" rule the evidence gate already applies to Experience Index, just enforced
    one gate later."""
    assert abs(sum(weights.values()) - 1.0) < 1e-6, "weights must sum to 1"
    df = df.copy()
    classified = df.loc[df["evidence_tier"] == "full"].copy()

    # Every factor is percentile-ranked (0-1, mentor-specified 2026-09-01) *within its own
    # quarter*, so a zone is only ever judged against its own quarter's distribution, never
    # mixed across time, then rescaled to 0-100 to match the band cut points below.

    # PeerGap = max(0, peer-group median Experience - zone Experience). `peer_gap` already on
    # `df` is (experience_index - peer_group_median_experience), i.e. the opposite sign.
    classified["peer_gap_raw"] = (-classified["peer_gap"]).clip(lower=0)
    classified["factor_peer_gap"] = 100 * classified.groupby("quarter")["peer_gap_raw"].transform(percentile_rank)

    # Deterioration (mentor methodology, 2026-09-15): a continuous relative-to-national-trend
    # severity, zero unless the decline has persisted `trends.CONSECUTIVE_DECLINES_REQUIRED`
    # consecutive quarters -- `src/trends.py::deterioration_magnitude` already IS this value (0
    # when the persistence gate isn't met, otherwise the mean per-quarter national-relative
    # decline size across the zone's current streak). Percentile-ranking the raw boolean here
    # used to collapse every deteriorating zone to one tied top value regardless of how severe
    # its decline actually was, and tied the entire non-deteriorating majority to one middling
    # rank instead of 0 (a 90%-tied-at-zero cluster's *average* rank lands near the middle of
    # its own span, not at the bottom) -- landing the whole non-deteriorating population in
    # "Medium" nationally, not "Low". `_deterioration_factor` below fixes both: zero magnitude
    # is assigned factor 0 directly (never percentile-ranked into the tied cluster), and only
    # the positive (actually-deteriorating) values are percentile-ranked among themselves, so
    # severity differentiates a barely-there 2-quarter dip from a steep 4-quarter slide instead
    # of tying them at the same "High" value.
    classified["factor_deterioration"] = classified.groupby("quarter")["deterioration_magnitude"].transform(
        _deterioration_factor
    )

    # TemporalAnomaly is now its own distinct factor (previously averaged together with the
    # Peer Gap ML score into a combined 'ml_anomaly' factor) -- the brief requires it as a
    # distinct output, and PeerGap above already carries the deterministic peer-comparison
    # signal, so folding the Peer Gap ML score in here too would double-count the same idea.
    # `peer_gap_ml_score`/`peer_gap_ml_anomaly` still exist on `df` (computed by
    # anomaly_detection.py, reported elsewhere -- T2, the copilot) but no longer feed Priority.
    # A handful of zones land alone in their (peer_group, quarter) cell, so Isolation Forest
    # can't score them at all -- missing ML evidence means "no anomaly signal available," not
    # "this zone can't have a Priority Score": default to 0 (least anomalous) rather than
    # letting a NaN factor null out the other three, otherwise-valid, deterministic factors.
    classified["factor_temporal_anomaly"] = 100 * classified.groupby("quarter")["temporal_anomaly_ml_score"].transform(percentile_rank)
    classified["factor_temporal_anomaly"] = classified["factor_temporal_anomaly"].fillna(0)

    # PopulationExposure = log(1 + population) (mentor-specified, 2026-09-01), then percentile
    # ranked. Note this log is monotonic, and percentile rank only depends on order -- so the
    # ranked factor here is numerically identical to ranking raw population directly. The log
    # still matters if this factor is ever combined a different way (e.g. min-max, or averaged
    # before ranking); kept as specified for fidelity to the mentor's formula and in case the
    # normalization approach changes later.
    classified["population_exposure_raw"] = np.log1p(classified["population"])
    classified["factor_population"] = 100 * classified.groupby("quarter")["population_exposure_raw"].transform(percentile_rank)

    weighted_factors = (
        weights["peer_gap"] * classified["factor_peer_gap"]
        + weights["deterioration"] * classified["factor_deterioration"]
        + weights["temporal_anomaly"] * classified["factor_temporal_anomaly"]
        + weights["population"] * classified["factor_population"]
    ) / 100  # factors are 0-100 above (for the band cut points); the formula below wants 0-1

    # Priority = 100 * G(Confidence) * weighted_factors (mentor-specified, 2026-09-01) -- see
    # `confidence_gate` for why this is a gate, not a linear multiplier.
    gate = confidence_gate(classified["confidence_score"])
    classified["priority_score"] = (100 * gate * weighted_factors).round(1)

    threshold = classified.groupby("quarter")["priority_score"].transform(
        lambda s: s.quantile(1 - PRIORITY_ZONE_FRACTION)
    )
    classified["priority_zone"] = classified["priority_score"] >= threshold

    factor_cols = ["factor_peer_gap", "factor_deterioration", "factor_temporal_anomaly", "factor_population"]
    band_cols = []
    for factor in factor_cols:
        band_col = factor.replace("factor_", "") + "_band"
        classified[band_col] = _factor_band(classified[factor])
        band_cols.append(band_col)

    keep_cols = ["h3_cell", "quarter", "priority_score", "priority_zone"] + factor_cols + band_cols
    out = df.merge(classified[keep_cols], on=["h3_cell", "quarter"], how="left")
    # 'low'/'insufficient' tier rows have no match in `classified` above, so the merge leaves
    # `priority_zone` as NaN (float) rather than False for them -- pandas can't hold NaN in a
    # bool column. Left as NaN, downstream `bool(nan)` evaluates True, which would wrongly tag
    # a shortlist-ineligible zone as a priority zone. `priority_score` and the factor/band
    # columns are left as real NaN on purpose -- "no score computed," not zero.
    out["priority_zone"] = out["priority_zone"].fillna(False).astype(bool)
    return out
