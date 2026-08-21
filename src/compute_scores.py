"""
AI/ML & Analytics — Experience Index & Confidence Score
Draft implementation, built ahead of real zone data landing from Data/GIS.

Schema this expects, once Data/GIS delivers the real zone table (one row per zone per quarter):
    zone_id            H3 cell id
    zone_name          human-readable label
    quarter             e.g. "2026Q2"
    avg_d_kbps          average download speed, from Ookla
    avg_u_kbps          average upload speed, from Ookla
    avg_lat_ms          average latency, from Ookla
    tests               test count this quarter, from Ookla
    devices             distinct device count this quarter, from Ookla
    quarters_observed   how many of the last 8 quarters this zone has any data in
    population          estimated residents, from WorldPop

Everything below runs on synthetic sample data shaped like this schema. Swapping in the
real zone table once it exists is a one-line change (see bottom of file) — nothing about
the formulas themselves needs to change.
"""
import pandas as pd
import numpy as np

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)

# ---------------------------------------------------------------------------
# 1. CONFIG — the placeholder decisions that need real discussion, not hidden defaults
# ---------------------------------------------------------------------------

# Experience Index weights: w_download + w_upload + w_latency must sum to 1.
# Starting point below leans on download (matches how the brief frames it — "download dominates
# the headline") but keeps upload and latency meaningfully in play, per the brief's explicit
# instruction not to drop upload just because download dominates. THIS IS A DRAFT, NOT A DECISION —
# the brief requires perturbing these 10-20% and reporting whether the ranking is stable (section 4
# below does a first pass at that).
EXPERIENCE_WEIGHTS = {"download": 0.50, "upload": 0.20, "latency": 0.30}

# Confidence Score weights: how much each evidence signal counts.
CONFIDENCE_WEIGHTS = {"tests": 0.50, "devices": 0.30, "quarters": 0.20}

# Below this many tests in a quarter, the brief says the honest output is "insufficient public
# evidence" — no Experience/Priority classification at all, not just a low score. 5 is a placeholder;
# the brief notes the median zone nationally sees ~4 tests/quarter, so this threshold is genuinely
# consequential and worth setting deliberately once real data is in.
MIN_TESTS_FOR_CLASSIFICATION = 5

# Test-count saturation point for the confidence formula: zones above this many tests/quarter
# get full marks on the "test volume" component. Placeholder — should be re-checked against the
# real national test-count distribution once it's known.
TESTS_SATURATION = 1500


# ---------------------------------------------------------------------------
# 2. SYNTHETIC SAMPLE DATA — shaped like the real zone table, standing in until it exists
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 3. THE TWO FORMULAS
# ---------------------------------------------------------------------------

def normalize_minmax(series: pd.Series) -> pd.Series:
    """Rescale a column onto 0-1 using the observed min/max in this run. Same method every
    time, per the brief's requirement that the Experience Index be fully reproducible."""
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(0.5, index=series.index)
    return ((series - lo) / (hi - lo)).clip(0, 1)


def experience_index(df: pd.DataFrame, weights: dict = EXPERIENCE_WEIGHTS) -> pd.Series:
    """Experience = w_d*D + w_u*U + w_l*L, each normalized 0-1, rescaled to 0-100.
    Latency is inverted first (lower ms = better) so higher always means better, for all three."""
    assert abs(sum(weights.values()) - 1.0) < 1e-6, "weights must sum to 1"
    d = normalize_minmax(df["avg_d_kbps"])
    u = normalize_minmax(df["avg_u_kbps"])
    l = normalize_minmax(-df["avg_lat_ms"])
    return (100 * (weights["download"] * d + weights["upload"] * u + weights["latency"] * l)).round(1)


def confidence_score(df: pd.DataFrame, weights: dict = CONFIDENCE_WEIGHTS,
                      min_tests: int = MIN_TESTS_FOR_CLASSIFICATION,
                      tests_cap: int = TESTS_SATURATION):
    """Confidence = f(tests, devices, quarters observed). Returns (score 0-100, insufficient flag).
    'insufficient' zones get no Experience/Priority classification at all — this is a status,
    not just a low number, per the brief's Case A / Case B example."""
    tests_component = np.clip(np.log1p(df["tests"]) / np.log1p(tests_cap), 0, 1)
    # devices/tests close to 1 = mostly distinct testers (good); close to 0 = a few phones
    # doing most of the testing (risk of single-user skew) — directly the brief's "500 tests
    # from 3 phones is not the same as 500 tests from 300 phones" example.
    device_ratio = (df["devices"] / df["tests"]).clip(0, 1)
    quarters_component = (df["quarters_observed"] / 8).clip(0, 1)

    raw = 100 * (weights["tests"] * tests_component + weights["devices"] * device_ratio
                 + weights["quarters"] * quarters_component)
    insufficient = df["tests"] < min_tests
    return raw.round(1), insufficient


# ---------------------------------------------------------------------------
# 4. RUN IT
# ---------------------------------------------------------------------------

df = sample_zones.copy()
df["experience_index"] = experience_index(df)
df["confidence_score"], df["insufficient_evidence"] = confidence_score(df)

# Per the brief's fixed rule: insufficient-evidence zones get no classification at all.
df.loc[df["insufficient_evidence"], "experience_index"] = np.nan

print("=" * 100)
print("EXPERIENCE INDEX + CONFIDENCE SCORE — sample output")
print("=" * 100)
print(df[["zone_name", "avg_d_kbps", "avg_u_kbps", "avg_lat_ms", "tests", "devices",
          "quarters_observed", "experience_index", "confidence_score", "insufficient_evidence"]]
      .to_string(index=False))

# ---------------------------------------------------------------------------
# 5. SENSITIVITY PREVIEW — a first pass at the brief's required weight-perturbation test
# ---------------------------------------------------------------------------
print()
print("=" * 100)
print("SENSITIVITY PREVIEW — perturbing Experience weights +/-15%, does the ranking hold?")
print("=" * 100)

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

print()
print("This is a first pass on 8 sample zones, not the formal T5 sensitivity test (that needs")
print("the real national dataset). But the mechanism — reweight, recompute, compare rankings —")
print("is exactly what the real test will run.")

# ---------------------------------------------------------------------------
# 6. SWAPPING IN REAL DATA — the only change needed once Data/GIS delivers the zone table
# ---------------------------------------------------------------------------
# df = pd.read_parquet("data/processed/zone_quarter_table.parquet")   # <- replace sample_zones with this
# df["experience_index"] = experience_index(df)
# df["confidence_score"], df["insufficient_evidence"] = confidence_score(df)
# df.loc[df["insufficient_evidence"], "experience_index"] = np.nan
