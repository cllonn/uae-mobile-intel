"""
Geographic prioritization engine: Priority = f(PeerGap, TemporalAnomaly, Deterioration,
PopulationExposure, Confidence).

Weights below are a first, explainable pass -- not yet sensitivity-tested (that's the brief's
mandatory T5: perturb weights +-10-20% and report whether the ranking reshuffles). The one
rule that isn't a judgment call, fixed by the brief itself: a low-confidence observation must
never automatically become a high-priority recommendation. Implemented here by scaling the
raw priority by the confidence fraction rather than treating confidence as one more additive
factor -- a zone that scores badly on every other factor still can't reach a high final score
without enough evidence behind it.

Input: the output of `trends.add_trend` + `anomaly_detection.add_peer_gap_ml` +
`anomaly_detection.add_temporal_anomaly_ml`.
"""
import pandas as pd

from src.compute_scores import normalize_minmax

PRIORITY_WEIGHTS = {
    "peer_gap": 0.35,       # how much worse than peers, right now
    "ml_anomaly": 0.20,     # how unusual, by either ML lens (peer or temporal)
    "deterioration": 0.20,  # persistent decline vs. peer trend
    "population": 0.25,     # how many people are exposed to it
}

# Top this fraction of classified zones, by priority score (within each quarter), get flagged
# "investigate first."
PRIORITY_ZONE_FRACTION = 0.10


def _factor_band(series: pd.Series) -> pd.Series:
    """Buckets a 0-100 factor into Low/Medium/High for the 'why this priority' display --
    fixed tercile cut points, not a per-zone judgment call baked into the number."""
    return pd.cut(series, bins=[-0.01, 33, 66, 100], labels=["Low", "Medium", "High"])


def add_priority(df: pd.DataFrame, weights: dict = PRIORITY_WEIGHTS) -> pd.DataFrame:
    """Adds `priority_score` (0-100), `priority_zone` (bool, top decile per quarter), the four
    underlying `factor_*` columns (0-100 each), and their Low/Medium/High bands. Requires
    `peer_gap`, `deteriorating`, `population`, `confidence_score`, `peer_gap_ml_score` and
    `temporal_anomaly_ml_score` already on `df`. Returns a new DataFrame."""
    assert abs(sum(weights.values()) - 1.0) < 1e-6, "weights must sum to 1"
    df = df.copy()
    classified = df.loc[~df["insufficient_evidence"]].copy()

    # Every factor is rescaled 0-100 *within its own quarter*, so a zone is only ever judged
    # against its own quarter's distribution, never mixed across time.
    classified["factor_peer_gap"] = classified.groupby("quarter")["peer_gap"].transform(
        lambda s: 100 * normalize_minmax(-s)
    )
    classified["ml_anomaly_raw"] = classified[["peer_gap_ml_score", "temporal_anomaly_ml_score"]].mean(axis=1)
    classified["factor_ml_anomaly"] = classified.groupby("quarter")["ml_anomaly_raw"].transform(
        lambda s: 100 * normalize_minmax(s)
    )
    # A handful of classified zones land alone in their (peer_group, quarter) cell (group size
    # 1 -> z-score std is undefined), so neither Isolation Forest can score them at all -- both
    # ml scores are NaN, not just one. Missing ML evidence means "no anomaly signal available,"
    # not "this zone can't have a Priority Score" -- default to 0 (least anomalous) rather than
    # letting a NaN factor null out the other three, otherwise-valid, deterministic factors.
    classified["factor_ml_anomaly"] = classified["factor_ml_anomaly"].fillna(0)
    classified["factor_deterioration"] = classified["deteriorating"].astype(float) * 100
    classified["factor_population"] = classified.groupby("quarter")["population"].transform(
        lambda s: 100 * normalize_minmax(s)
    )

    raw_priority = (
        weights["peer_gap"] * classified["factor_peer_gap"]
        + weights["ml_anomaly"] * classified["factor_ml_anomaly"]
        + weights["deterioration"] * classified["factor_deterioration"]
        + weights["population"] * classified["factor_population"]
    )
    # The confidence guardrail: multiplying (not adding) means a zone at 30/100 confidence can
    # score at most 30% of its raw priority, however bad everything else looks.
    classified["priority_score"] = (raw_priority * classified["confidence_score"] / 100).round(1)

    threshold = classified.groupby("quarter")["priority_score"].transform(
        lambda s: s.quantile(1 - PRIORITY_ZONE_FRACTION)
    )
    classified["priority_zone"] = classified["priority_score"] >= threshold

    factor_cols = ["factor_peer_gap", "factor_ml_anomaly", "factor_deterioration", "factor_population"]
    band_cols = []
    for factor in factor_cols:
        band_col = factor.replace("factor_", "") + "_band"
        classified[band_col] = _factor_band(classified[factor])
        band_cols.append(band_col)

    keep_cols = ["h3_cell", "quarter", "priority_score", "priority_zone"] + factor_cols + band_cols
    return df.merge(classified[keep_cols], on=["h3_cell", "quarter"], how="left")
