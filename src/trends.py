"""
Deterioration / trend intelligence: is a zone falling behind the national trend over time?

Per the brief, change must be measured relative to a reference trend, never in raw Mbps -- a
zone can improve in absolute terms and still be falling behind if the reference is improving
faster (the brief's own example: national median rose ~13% over the window, which hides zones
that are falling behind while still improving slightly in absolute terms). Reference is the
NATIONAL median Experience Index per quarter (mentor methodology, 2026-09-15) -- not the zone's
own peer group -- so a zone is never penalized just because the whole country moved similarly;
`national_gap` (zone Experience Index minus the national median, same quarter) is this file's
own relative-standing measure, tracked quarter to quarter exactly the way `peer_gap` used to be
before this methodology change.

Deterioration is only flagged once decline has persisted for 2 consecutive calendar quarters --
single-quarter noise is large on data this sparse (median 4 tests/zone/quarter).

`deterioration_magnitude` (2026-09-15) is the continuous severity companion to the `deteriorating`
boolean: 0 whenever the persistence gate isn't met (no contribution at all, matching "set to zero
unless the decline persists"), otherwise the mean per-quarter size of the national-relative
decline across the zone's current streak -- so a barely-there 2-quarter dip and a steep 4-quarter
slide are no longer indistinguishable. `src/priority.py::add_priority` percentile-ranks this
(never the boolean) to build the Deterioration factor.

Input: the output of `compute_scores.score_zone_quarters` (needs `h3_cell`, `quarter`,
`experience_index`, `evidence_tier`).
"""
import numpy as np
import pandas as pd

QUARTER_ORDER = ["2024Q3", "2024Q4", "2025Q1", "2025Q2", "2025Q3", "2025Q4", "2026Q1", "2026Q2"]
QUARTER_INDEX = {q: i for i, q in enumerate(QUARTER_ORDER)}

# Consecutive quarter-over-quarter declines in national_gap required before flagging a zone
# (mentor methodology, 2026-09-15: "persists for at least 2 consecutive quarters"). Previously 3
# (calibrated against peer_gap-based streaks, which this file no longer computes) -- kept here as
# the one place this number is defined; `tests/test_t3_trends.py` and `src/copilot_tools.py`'s
# methodology text both reference this constant rather than a second hardcoded copy.
CONSECUTIVE_DECLINES_REQUIRED = 2

# How many of the most recent quarters to fit a slope over, for the "points per quarter" trend
# stat shown in the UI (`trend_pts_per_qtr`) -- unchanged by the 2026-09-15 methodology update:
# this remains the zone's own raw Experience Index slope, a separate stat from the
# national-relative deterioration severity below, and still drives the map's Trend layer/legend.
TREND_WINDOW_QUARTERS = 4


def _zone_trend(group: pd.DataFrame, national: pd.Series) -> pd.DataFrame:
    """Runs the deterioration/slope logic for one zone's 8-quarter timeline (called per zone via
    groupby.apply). Written as a plain loop over quarters rather than a vectorized one-liner --
    at 8 quarters per zone the performance cost is negligible, and the loop reads exactly like
    the rule it implements: 'if this quarter's national gap is worse than last quarter's, and
    that was also true the quarter before, the zone is deteriorating.'"""
    h3_cell = group.name  # pandas groupby.apply excludes the grouping column from `group` itself
    group = group.set_index("quarter_idx").reindex(range(len(QUARTER_ORDER)))
    quarters = [QUARTER_ORDER[i] for i in group.index]
    national_gap = group["experience_index"] - national.reindex(quarters).to_numpy(dtype=float)

    declining_streak = 0
    streak_deltas: list[float] = []  # this quarter's decline size, for every step of the CURRENT
    # streak -- cleared the instant the streak breaks, so it never bleeds across a reset
    deteriorating = [False] * len(QUARTER_ORDER)
    magnitude = [0.0] * len(QUARTER_ORDER)
    for i in range(1, len(QUARTER_ORDER)):
        prev_gap, this_gap = national_gap.iloc[i - 1], national_gap.iloc[i]
        # Both quarters must actually have a score -- a gap in the data (insufficient evidence,
        # or no measurement at all) breaks the streak rather than silently skipping past it.
        if pd.notna(prev_gap) and pd.notna(this_gap) and this_gap < prev_gap:
            declining_streak += 1
            streak_deltas.append(prev_gap - this_gap)  # positive = this step's decline size
        else:
            declining_streak = 0
            streak_deltas = []
        deteriorating[i] = declining_streak >= CONSECUTIVE_DECLINES_REQUIRED
        magnitude[i] = round(float(np.mean(streak_deltas)), 2) if deteriorating[i] else 0.0

    group["deteriorating"] = deteriorating
    group["deterioration_magnitude"] = magnitude

    recent = group["experience_index"].tail(TREND_WINDOW_QUARTERS).dropna()
    if len(recent) >= 2:
        slope = np.polyfit(recent.index.to_numpy(), recent.to_numpy(), 1)[0]
    else:
        slope = np.nan
    group["trend_pts_per_qtr"] = round(slope, 2) if pd.notna(slope) else np.nan
    group["h3_cell"] = h3_cell

    return group.reset_index()


def add_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Adds `deteriorating` (bool -- True once national-relative decline has persisted
    CONSECUTIVE_DECLINES_REQUIRED consecutive quarters), `deterioration_magnitude` (float, 0
    unless `deteriorating` is True, otherwise the mean per-quarter national-relative decline size
    across the current streak), and `trend_pts_per_qtr` (float, the zone's own raw Experience
    Index slope -- same value repeated on every row of a zone) to `df`. Requires
    `experience_index` and `evidence_tier` (i.e. `compute_scores.score_zone_quarters` has already
    run). Returns a new DataFrame; `df` is not modified in place."""
    df = df.copy()
    df["quarter_idx"] = df["quarter"].map(QUARTER_INDEX)
    # National reference for the deterioration streak/magnitude -- median Experience Index per
    # quarter across 'full'-tier zones (mirrors `anomaly_detection.py`'s own national series;
    # kept self-contained here rather than imported, so this file has no dependency on that
    # module). Computed once, over the whole df, before the per-zone loop below.
    national = df.loc[df["evidence_tier"] == "full"].groupby("quarter")["experience_index"].median()
    trended = df.groupby("h3_cell", group_keys=False).apply(_zone_trend, national=national)
    # Reindexing added placeholder rows for quarters this zone was never actually measured in
    # (needed so the streak logic sees real gaps) -- drop them now that the logic is done.
    trended = trended.dropna(subset=["quarter"])
    # The reindex step mixed real True/False values with NaN placeholders, which silently
    # upcasts a bool column to `object` -- and `~` on an object-dtype column of Python bools
    # does bitwise-NOT (~True == -2), not logical negation. No NaNs survive the dropna above,
    # so it's safe (and necessary) to cast this back to a real bool column here.
    trended["insufficient_evidence"] = trended["insufficient_evidence"].astype(bool)
    # Each zone's own `_zone_trend` call resets its local index to 0..7, so the concatenated
    # result has heavily duplicated index values (up to ~1,800x on this data) -- harmless for
    # most pandas operations, but catastrophic for anything doing an index-based join
    # downstream (a join between two frames with duplicate indices is a cross-join on those
    # duplicates, which silently explodes into billions of rows). Reset to a clean index here.
    return trended.drop(columns=["quarter_idx"]).reset_index(drop=True)
