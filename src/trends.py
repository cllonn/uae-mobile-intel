"""
Deterioration / trend intelligence: is a zone falling behind its peers over time?

Per the brief, change must be measured relative to the peer-group trend, never in raw Mbps --
a zone can improve in absolute terms and still be falling behind if its peers are improving
faster (the brief's own example: national median rose ~13% over the window, which hides zones
that are falling behind while still improving slightly in absolute terms). `peer_gap` (from
compute_scores.py) already measures "how far above/below peers, this quarter" -- tracking how
peer_gap moves quarter to quarter *is* the relative-trend measure the brief describes.

Deterioration is only flagged once decline has persisted for 2-3 consecutive calendar
quarters -- single-quarter noise is large on data this sparse (median 4 tests/zone/quarter).

Input: the output of `compute_scores.score_zone_quarters` (needs `h3_cell`, `quarter`,
`peer_gap`, `experience_index`).
"""
import numpy as np
import pandas as pd

QUARTER_ORDER = ["2024Q3", "2024Q4", "2025Q1", "2025Q2", "2025Q3", "2025Q4", "2026Q1", "2026Q2"]
QUARTER_INDEX = {q: i for i, q in enumerate(QUARTER_ORDER)}

# Consecutive quarter-over-quarter declines in peer_gap required before flagging a zone.
# The brief allows "two or three quarters" -- tried at 2 first (3 quarters of evidence), which
# flagged 413 zones nationally at some point across the window; on data this sparse, that's
# picking up sampling noise, not genuine deterioration (the brief's own comparable numbers are
# 71-99 zones). 3 consecutive declines (4 quarters of evidence) brings that down to 107,
# in line with the brief's own cited scale.
CONSECUTIVE_DECLINES_REQUIRED = 3

# How many of the most recent quarters to fit a slope over, for the "points per quarter"
# trend stat shown in the UI.
TREND_WINDOW_QUARTERS = 4


def _zone_trend(group: pd.DataFrame) -> pd.DataFrame:
    """Runs the deterioration/slope logic for one zone's 8-quarter timeline (called per zone
    via groupby.apply). Written as a plain loop over quarters rather than a vectorized
    one-liner -- at 8 quarters per zone the performance cost is negligible, and the loop reads
    exactly like the rule it implements: 'if this quarter's peer gap is worse than last
    quarter's, and that was also true the quarter before, the zone is deteriorating.'"""
    h3_cell = group.name  # pandas groupby.apply excludes the grouping column from `group` itself
    group = group.set_index("quarter_idx").reindex(range(len(QUARTER_ORDER)))
    peer_gap = group["peer_gap"]

    declining_streak = 0
    deteriorating = [False] * len(QUARTER_ORDER)
    for i in range(1, len(QUARTER_ORDER)):
        prev_gap, this_gap = peer_gap.iloc[i - 1], peer_gap.iloc[i]
        # Both quarters must actually have a score -- a gap in the data (insufficient evidence,
        # or no measurement at all) breaks the streak rather than silently skipping past it.
        if pd.notna(prev_gap) and pd.notna(this_gap) and this_gap < prev_gap:
            declining_streak += 1
        else:
            declining_streak = 0
        deteriorating[i] = declining_streak >= CONSECUTIVE_DECLINES_REQUIRED

    group["deteriorating"] = deteriorating

    recent = group["experience_index"].tail(TREND_WINDOW_QUARTERS).dropna()
    if len(recent) >= 2:
        slope = np.polyfit(recent.index.to_numpy(), recent.to_numpy(), 1)[0]
    else:
        slope = np.nan
    group["trend_pts_per_qtr"] = round(slope, 2) if pd.notna(slope) else np.nan
    group["h3_cell"] = h3_cell

    return group.reset_index()


def add_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Adds `deteriorating` (bool -- True once decline has persisted 2+ consecutive quarters)
    and `trend_pts_per_qtr` (float, same value repeated on every row of a zone) to `df`.
    Requires `peer_gap` and `experience_index` (i.e. `compute_scores.score_zone_quarters` has
    already run). Returns a new DataFrame; `df` is not modified in place."""
    df = df.copy()
    df["quarter_idx"] = df["quarter"].map(QUARTER_INDEX)
    trended = df.groupby("h3_cell", group_keys=False).apply(_zone_trend)
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
