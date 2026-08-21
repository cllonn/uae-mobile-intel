# %% [markdown]
# # Experience Index & Confidence Score — AI/ML & Analytics, draft
#
# **Owner:** Saif · Team A (helping unblock Team B's AI/ML & Analytics track)
# **Status:** draft, built ahead of Data/GIS's real zone table landing — ready to run the moment it does.
#
# This notebook implements the two formulas the brief requires to be **deterministic and reproducible**:
# same inputs, same output, every time — no ML, no LLM involved in either calculation.
#
# - **Experience Index** — a 0-100 score for measured network quality in a zone.
# - **Confidence Score** — how much evidence backs that score, so a zone with 3 lucky tests
#   never gets treated the same as one with 3,000.
#
# Everything runs here on **synthetic sample data** shaped exactly like the real zone table
# will be. The last section shows the one-line change needed to point this at the real thing.

# %%
import pandas as pd
import numpy as np

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)
print("pandas", pd.__version__, "| numpy", np.__version__)

# %% [markdown]
# ## 1. Config — the placeholder decisions that need real discussion
#
# These aren't hidden defaults. The brief is explicit that choosing and defending the
# Experience weights is *the assignment* — so every number below is a first draft, flagged
# for Monday's review, not a finished decision.

# %%
# Experience Index weights: w_download + w_upload + w_latency must sum to 1.
# Draft rationale: leans on download (the brief notes it "dominates the headline"), but
# keeps upload and latency meaningfully in play per the brief's explicit instruction not
# to drop upload just because download dominates. THIS IS A DRAFT — Section 4 below runs
# a first pass at the brief's required sensitivity test (perturb 10-20%, check if ranking holds).
EXPERIENCE_WEIGHTS = {"download": 0.50, "upload": 0.20, "latency": 0.30}

# Confidence Score weights: how much each evidence signal counts toward trustworthiness.
CONFIDENCE_WEIGHTS = {"tests": 0.50, "devices": 0.30, "quarters": 0.20}

# Below this many tests in a quarter, the brief's own rule is: the honest output is
# "insufficient public evidence" — no classification at all, not just a low score.
# The brief notes the median zone nationally sees ~4 tests/quarter, so this threshold
# is genuinely consequential and worth setting deliberately once real data is in.
MIN_TESTS_FOR_CLASSIFICATION = 5

# Test-count saturation point: zones above this many tests/quarter get full marks on the
# "test volume" component of Confidence. Placeholder — recheck against the real national
# test-count distribution once it's known.
TESTS_SATURATION = 1500

print("Experience weights:", EXPERIENCE_WEIGHTS, " (sum =", sum(EXPERIENCE_WEIGHTS.values()), ")")
print("Confidence weights: ", CONFIDENCE_WEIGHTS, " (sum =", sum(CONFIDENCE_WEIGHTS.values()), ")")
print("Min tests for classification:", MIN_TESTS_FOR_CLASSIFICATION)
print("Tests saturation point:", TESTS_SATURATION)

# %% [markdown]
# ## 2. Synthetic sample data
#
# Shaped exactly like the real zone-quarter table Data/GIS will deliver: one row per zone,
# per quarter, with the Ookla measurement fields, population, and how many of the last 8
# quarters this zone has any data in at all. The first five rows mirror the featured zones
# from Mohammed's mockup, so the output below is directly comparable to it.

# %%
sample_zones = pd.DataFrame([
    {"zone_id": "87e4d2a1", "zone_name": "Al Warsan / International City", "avg_d_kbps": 71000, "avg_u_kbps": 20000, "avg_lat_ms": 32, "tests": 9400, "devices": 1222, "quarters_observed": 8, "population": 96000},
    {"zone_id": "87e4d2a2", "zone_name": "Mussafah Industrial (ICAD I)",   "avg_d_kbps": 68000, "avg_u_kbps": 18500, "avg_lat_ms": 35, "tests": 2140, "devices": 278,  "quarters_observed": 8, "population": 41000},
    {"zone_id": "87e4d2a3", "zone_name": "Muwailih Commercial",            "avg_d_kbps": 92000, "avg_u_kbps": 26000, "avg_lat_ms": 27, "tests": 5210, "devices": 677,  "quarters_observed": 8, "population": 63000},
    {"zone_id": "87e4d2a4", "zone_name": "Al Ain Industrial Area",         "avg_d_kbps": 96000, "avg_u_kbps": 28000, "avg_lat_ms": 41, "tests": 640,  "devices": 83,   "quarters_observed": 5, "population": 22000},
    {"zone_id": "87e4d2a5", "zone_name": "Al Jurf",                        "avg_d_kbps": 99000, "avg_u_kbps": 29000, "avg_lat_ms": 26, "tests": 305,  "devices": 40,   "quarters_observed": 4, "population": 38000},
    {"zone_id": "87e4d2a6", "zone_name": "Dubai Core · Sector DXB-04",     "avg_d_kbps": 210000,"avg_u_kbps": 58000, "avg_lat_ms": 18, "tests": 18400,"devices": 2390, "quarters_observed": 8, "population": 120000},
    {"zone_id": "87e4d2a7", "zone_name": "Ruwais · Sector AUH-11",         "avg_d_kbps": 150000,"avg_u_kbps": 41000, "avg_lat_ms": 22, "tests": 812,  "devices": 105,  "quarters_observed": 6, "population": 15000},
    {"zone_id": "87e4d2a8", "zone_name": "Kalba · Sector SHJ-03",          "avg_d_kbps": 146000,"avg_u_kbps": 39000, "avg_lat_ms": 24, "tests": 3,    "devices": 1,    "quarters_observed": 2, "population": 9000},
])
print(sample_zones.to_string(index=False))

# %% [markdown]
# ## 3. The two formulas
#
# **Experience Index** = w_d·D + w_u·U + w_l·L, where D, U, L are download, upload, and
# *inverted* latency, each rescaled onto 0-1 before combining — you can't add Mbps to
# milliseconds, so everything goes on the same scale first.
#
# **Confidence Score** = f(tests, devices, quarters observed). Test volume uses a log scale
# (a handful of zones have thousands of tests while the median zone has ~4 — a straight
# average would let the busiest zones swamp everything). Device ratio rewards distinct
# testers over one phone testing repeatedly, straight from the brief's own example: *"500
# tests from 3 phones is not the same as 500 tests from 300 phones."*

# %%
def normalize_minmax(series: pd.Series) -> pd.Series:
    """Rescale a column onto 0-1 using the observed min/max in this run.
    Same method every time, per the brief's reproducibility requirement."""
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(0.5, index=series.index)
    return ((series - lo) / (hi - lo)).clip(0, 1)


def experience_index(df: pd.DataFrame, weights: dict = EXPERIENCE_WEIGHTS) -> pd.Series:
    """Experience = w_d*D + w_u*U + w_l*L. Latency is inverted first (lower ms = better)
    so higher always means better, for all three components."""
    assert abs(sum(weights.values()) - 1.0) < 1e-6, "weights must sum to 1"
    d = normalize_minmax(df["avg_d_kbps"])
    u = normalize_minmax(df["avg_u_kbps"])
    l = normalize_minmax(-df["avg_lat_ms"])
    return (100 * (weights["download"] * d + weights["upload"] * u + weights["latency"] * l)).round(1)


def confidence_score(df: pd.DataFrame, weights: dict = CONFIDENCE_WEIGHTS,
                      min_tests: int = MIN_TESTS_FOR_CLASSIFICATION,
                      tests_cap: int = TESTS_SATURATION):
    """Returns (score 0-100, insufficient-evidence flag). 'Insufficient' zones get no
    Experience/Priority classification at all -- a status, not just a low number, per the
    brief's Case A / Case B example (500 tests/300 devices/8-of-8 quarters = trust it;
    2 tests/1 device/2-of-8 quarters = insufficient evidence, no classification)."""
    tests_component = np.clip(np.log1p(df["tests"]) / np.log1p(tests_cap), 0, 1)
    device_ratio = (df["devices"] / df["tests"]).clip(0, 1)
    quarters_component = (df["quarters_observed"] / 8).clip(0, 1)

    raw = 100 * (weights["tests"] * tests_component + weights["devices"] * device_ratio
                 + weights["quarters"] * quarters_component)
    insufficient = df["tests"] < min_tests
    return raw.round(1), insufficient

print("Formulas defined: normalize_minmax, experience_index, confidence_score")

# %% [markdown]
# ## 4. Run it
#
# Per the brief's fixed rule, zones flagged `insufficient_evidence` get their Experience
# Index blanked out entirely — not just a low number, no classification at all.

# %%
df = sample_zones.copy()
df["experience_index"] = experience_index(df)
df["confidence_score"], df["insufficient_evidence"] = confidence_score(df)
df.loc[df["insufficient_evidence"], "experience_index"] = np.nan

print(df[["zone_name", "avg_d_kbps", "avg_u_kbps", "avg_lat_ms", "tests", "devices",
          "quarters_observed", "experience_index", "confidence_score", "insufficient_evidence"]]
      .to_string(index=False))

# %% [markdown]
# **Reading this table:** Kalba (3 tests, 1 device) correctly gets no Experience Index at
# all — exactly the brief's Case B. Notice Al Warsan, Mussafah, Muwailih, and Dubai Core all
# land near the same Confidence ceiling (~74) despite very different test counts — that's
# intentional, not a bug: once a zone has *enough* tests, devices, and full quarter coverage,
# piling on more evidence shouldn't keep inflating confidence forever. The formula should stop
# rewarding evidence you don't need.

# %% [markdown]
# ## 5. Sensitivity preview — a first pass at the brief's required weight-perturbation test
#
# The brief requires perturbing the Experience weights 10-20% and reporting whether the
# top-ranked zones reshuffle. This is a first pass on 8 sample zones to prove the mechanism
# works — the formal test needs the real national dataset, which is Team B's next step once
# the zone table lands.

# %%
base_rank = df.dropna(subset=["experience_index"]).sort_values("experience_index").zone_name.tolist()

perturbations = {
    "download +15% (rest rescaled)": {"download": 0.575, "upload": 0.17, "latency": 0.255},
    "download -15% (rest rescaled)": {"download": 0.425, "upload": 0.23, "latency": 0.345},
    "latency +15% (rest rescaled)":  {"download": 0.4625, "upload": 0.1875, "latency": 0.35},
}

for label, w in perturbations.items():
    df_p = sample_zones.copy()
    df_p["experience_index"] = experience_index(df_p, weights=w)
    df_p["confidence_score"], df_p["insufficient_evidence"] = confidence_score(df_p)
    df_p.loc[df_p["insufficient_evidence"], "experience_index"] = np.nan
    rank_p = df_p.dropna(subset=["experience_index"]).sort_values("experience_index").zone_name.tolist()
    changed = rank_p != base_rank
    print(f"- {label}: ranking {'CHANGED' if changed else 'held steady'}")

# %% [markdown]
# **This is worth discussing Monday, not hiding:** the ranking changes under two of the
# three perturbations tested. Per the brief's own instruction — *"if your top zones reshuffle,
# your index is fragile, and that is a finding worth reporting"* — this isn't a failure, it's
# exactly the kind of honest result the brief asks for. On only 8 sample zones this is
# preliminary; the real test needs the full national dataset to mean anything conclusive, but
# the direction is worth flagging to the team now rather than after the weights are treated
# as settled.

# %% [markdown]
# ## 6. Swapping in real data
#
# Once Data/GIS delivers the real zone-quarter table, this is the only change needed —
# nothing about the formulas above changes:
#
# ```python
# df = pd.read_parquet("data/processed/zone_quarter_table.parquet")
# df["experience_index"] = experience_index(df)
# df["confidence_score"], df["insufficient_evidence"] = confidence_score(df)
# df.loc[df["insufficient_evidence"], "experience_index"] = np.nan
# ```
#
# ### Open items for the team, Monday
# - **Confirm or revise `EXPERIENCE_WEIGHTS`** — current split (50/20/30) is a starting point tied to the priority-weighting proposal from Product & Business, not a final call.
# - **Confirm `MIN_TESTS_FOR_CLASSIFICATION` and `TESTS_SATURATION`** against the real national test-count distribution once known — both are placeholders right now.
# - **Run the full T5 sensitivity test** on the real dataset, not just this 8-zone preview.
# - **Peer-group comparison (Peer Gap) is deliberately not in this notebook** — it needs the OSM density classification from Data/GIS first, so it's the natural next step once that lands, not before.
