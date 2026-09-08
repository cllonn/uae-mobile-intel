"""
T7 -- Copilot paraphrase routing: a decision-maker will never type the exact canonical wording
back at the Copilot. This asserts that realistic paraphrases of every canonical intent still
land on the same deterministic tool (and the same parameters) as the canonical phrasing --
routing stays pure keyword matching (src/copilot.py::route, no LLM involved in tool selection),
so this is what actually proves paraphrase robustness, not a hope.

Special focus (explicitly requested): "Which zone has the lowest experience?" -- this used to be
unmatched entirely (no "weak/weakest" keyword), and even after adding "lowest"/"worst" as
synonyms, the singular phrasing ("zone", not "zones") needs to resolve to exactly one result
(n=1), not the usual top-10 list, so the map highlights exactly the one zone asked about.

Run: python -m tests.test_t7_copilot_paraphrase   (or `pytest tests/` if pytest is installed)
"""
from src import copilot

ZONE_ID = "test_zone_id"  # route() only checks truthiness/pattern match, never resolves it

# (intent label, question, zone_id, expected_tool, expected_args)
CASES = [
    # --- get_top_priority_zones ------------------------------------------------------------
    ("top priority (canonical)", "Which 5 areas should we investigate first?", None,
     "get_top_priority_zones", {"n": 5, "emirate": None}),
    ("top priority (paraphrase: 'top N priority')", "What are the top 3 priority areas?", None,
     "get_top_priority_zones", {"n": 3, "emirate": None}),
    ("top priority (paraphrase: no explicit N -> default 5)", "Which areas need the highest priority?", None,
     "get_top_priority_zones", {"n": 5, "emirate": None}),
    ("top priority (paraphrase: emirate-scoped)", "Show the highest-priority zones in Sharjah.", None,
     "get_top_priority_zones", {"n": 5, "emirate": "Sharjah"}),

    # --- get_weakest_zones, plural ("Where is experience weakest?") ------------------------
    ("weakest, plural (canonical)", "Where is experience weakest?", None,
     "get_weakest_zones", {"n": 10, "emirate": None}),
    ("weakest, plural (paraphrase: emirate-scoped)", "Show the weakest zones in Dubai.", None,
     "get_weakest_zones", {"n": 10, "emirate": "Dubai"}),
    ("weakest, plural (paraphrase: 'worst')", "Which areas have the worst mobile experience?", None,
     "get_weakest_zones", {"n": 10, "emirate": None}),

    # --- get_weakest_zones, singular -- the headline case for this test file ---------------
    ("weakest, singular (paraphrase: 'which zone... lowest')", "Which zone has the lowest experience?", None,
     "get_weakest_zones", {"n": 1, "emirate": None}),
    ("weakest, singular (paraphrase: 'which area... worst')", "Which area performs the worst?", None,
     "get_weakest_zones", {"n": 1, "emirate": None}),
    ("weakest, singular (paraphrase: 'the single weakest')", "What is the single weakest area?", None,
     "get_weakest_zones", {"n": 1, "emirate": None}),
    ("weakest, explicit N=3 overrides singular default", "Which 3 zones have the lowest experience?", None,
     "get_weakest_zones", {"n": 3, "emirate": None}),
    ("weakest, explicit N=5 overrides singular default", "Which 5 zones have the lowest experience?", None,
     "get_weakest_zones", {"n": 5, "emirate": None}),
    ("weakest, singular (paraphrase: 'the worst zone', no question mark)", "What is the worst zone", None,
     "get_weakest_zones", {"n": 1, "emirate": None}),
    ("weakest, singular (paraphrase: 'the weakest area')", "Show me the weakest area.", None,
     "get_weakest_zones", {"n": 1, "emirate": None}),

    # --- get_strongest_zones -- Experience-ranking intent, 'highest' direction (new) ----------
    # Same intent as get_weakest_zones above (Experience Index ranking), opposite direction --
    # one resolver (_resolve_experience_query) handles both, never a route per sentence.
    ("strongest, plural (canonical)", "where is experience the highest", None,
     "get_strongest_zones", {"n": 10, "emirate": None}),
    ("strongest, singular ('which zone... highest')", "which zone has the highest experience", None,
     "get_strongest_zones", {"n": 1, "emirate": None}),
    ("strongest, explicit N=5 ('top N ... best experience')", "show me top 5 best experience areas", None,
     "get_strongest_zones", {"n": 5, "emirate": None}),
    ("strongest, emirate-scoped, metric typo ('experince')", "highest experince in rak", None,
     "get_strongest_zones", {"n": 10, "emirate": "Ras Al Khaimah"}),
    ("strongest, emirate-scoped, metric typo ('experiance')", "best experiance in auh", None,
     "get_strongest_zones", {"n": 10, "emirate": "Abu Dhabi"}),
    ("strongest, paraphrase: 'strongest experience'", "which areas have the strongest experience?", None,
     "get_strongest_zones", {"n": 10, "emirate": None}),
    ("strongest, paraphrase: 'top experience'", "show the top experience areas", None,
     "get_strongest_zones", {"n": 10, "emirate": None}),

    # --- get_weakest_zones -- same Experience-ranking intent, explicit 'lowest' direction words
    # not already covered above (weakest/lowest/worst were already covered without the word
    # 'experience' present; these add the explicit-'experience' phrasing the same resolver must
    # also still send to the unchanged get_weakest_zones tool). ------------------------------
    ("weakest, plural, rephrased ('the weakest')", "where is experience the weakest", None,
     "get_weakest_zones", {"n": 10, "emirate": None}),
    ("weakest, emirate-scoped, explicit 'experience'", "worst experience in shj", None,
     "get_weakest_zones", {"n": 10, "emirate": "Sharjah"}),
    ("weakest, paraphrase: 'poorest experience'", "which areas have the poorest experience?", None,
     "get_weakest_zones", {"n": 10, "emirate": None}),

    # --- get_metric_threshold_zones -- one resolver/tool for above/below x download/upload/
    # latency (previously get_above_median_download_zones: download-only, 'above'-only). -------
    ("metric-threshold, above download (exact requested wording)", "Which zones have greater than the median download", None,
     "get_metric_threshold_zones", {"metric": "download_mbps", "comparison": "above", "emirate": None}),
    ("metric-threshold, above download (paraphrase: 'download speed')", "Show zones above the median download speed.", None,
     "get_metric_threshold_zones", {"metric": "download_mbps", "comparison": "above", "emirate": None}),
    ("metric-threshold, above download (paraphrase, emirate-scoped)", "Which areas in Dubai have above-median download?", None,
     "get_metric_threshold_zones", {"metric": "download_mbps", "comparison": "above", "emirate": "Dubai"}),
    ("metric-threshold, below download (exact requested wording)", "show me the areas that scored lower than the median download", None,
     "get_metric_threshold_zones", {"metric": "download_mbps", "comparison": "below", "emirate": None}),
    ("metric-threshold, above upload", "areas above median upload", None,
     "get_metric_threshold_zones", {"metric": "upload_mbps", "comparison": "above", "emirate": None}),
    ("metric-threshold, below upload", "areas below median upload", None,
     "get_metric_threshold_zones", {"metric": "upload_mbps", "comparison": "below", "emirate": None}),
    ("metric-threshold, above latency", "areas above median latency", None,
     "get_metric_threshold_zones", {"metric": "latency_ms", "comparison": "above", "emirate": None}),
    ("metric-threshold, below latency", "areas below median latency", None,
     "get_metric_threshold_zones", {"metric": "latency_ms", "comparison": "below", "emirate": None}),
    ("metric-threshold, typo ('downlod')", "below median downlod", None,
     "get_metric_threshold_zones", {"metric": "download_mbps", "comparison": "below", "emirate": None}),
    ("metric-threshold, paraphrase ('under')", "areas under median upload", None,
     "get_metric_threshold_zones", {"metric": "upload_mbps", "comparison": "below", "emirate": None}),
    ("metric-threshold, typo ('medain')", "latency greater than the medain", None,
     "get_metric_threshold_zones", {"metric": "latency_ms", "comparison": "above", "emirate": None}),

    # --- get_deteriorating_zones -------------------------------------------------------------
    ("deteriorating (canonical, no N -> unchanged 'all flagged' default)", "Which areas are deteriorating?", None,
     "get_deteriorating_zones", {"emirate": None}),
    ("deteriorating (paraphrase: 'getting worse over time', no N)", "Which areas are getting worse over time?", None,
     "get_deteriorating_zones", {"emirate": None}),
    ("deteriorating, explicit top 5 (exact requested wording)", "show me the top 5 most deterioration zones", None,
     "get_deteriorating_zones", {"emirate": None, "n": 5}),
    ("deteriorating, explicit top 3", "top 3 deteriorating zones", None,
     "get_deteriorating_zones", {"emirate": None, "n": 3}),
    ("deteriorating, explicit 'worst 10'", "show me the worst 10 deteriorating areas", None,
     "get_deteriorating_zones", {"emirate": None, "n": 10}),
    ("deteriorating, singular 'the most' -> n=1", "which area deteriorated the most?", None,
     "get_deteriorating_zones", {"emirate": None, "n": 1}),

    # --- get_anomalous_zones, both kinds ------------------------------------------------------
    ("anomalous, peer_gap (canonical)", "Which areas are anomalous?", None,
     "get_anomalous_zones", {"kind": "peer_gap", "emirate": None}),
    ("anomalous, peer_gap (canonical, explicit 'relative to peers')",
     "Which areas are anomalous relative to their peers?", None,
     "get_anomalous_zones", {"kind": "peer_gap", "emirate": None}),
    ("anomalous, peer_gap (paraphrase: 'unusual ... similar zones')",
     "Which areas are unusual compared to similar zones?", None,
     "get_anomalous_zones", {"kind": "peer_gap", "emirate": None}),
    ("anomalous, temporal (paraphrase: 'own history')",
     "Which areas are anomalous compared to their own history?", None,
     "get_anomalous_zones", {"kind": "temporal", "emirate": None}),
    ("anomalous, temporal (paraphrase: 'unusual changes over time')",
     "Which areas show unusual changes over time?", None,
     "get_anomalous_zones", {"kind": "temporal", "emirate": None}),

    # --- get_high_population_weak_zones -------------------------------------------------------
    ("weak + high population (canonical)", "Where do weak experience and high population occur together?", None,
     "get_high_population_weak_zones", {"emirate": None}),
    ("weak + high population (paraphrase)", "Which weak areas have a large population?", None,
     "get_high_population_weak_zones", {"emirate": None}),

    # --- get_coverage_summary -----------------------------------------------------------------
    ("coverage (paraphrase: 'how much ... represented')", "How much of the population is represented?", None,
     "get_coverage_summary", {"emirate": None}),

    # --- zone-scoped: get_zone_details ("why priority") --------------------------------------
    ("why priority (canonical, needs selected zone)", "Why was this zone given high priority?", ZONE_ID,
     "get_zone_details", {"zone_id": ZONE_ID}),
    ("why priority (paraphrase: 'this area')", "Why is this area high priority?", ZONE_ID,
     "get_zone_details", {"zone_id": ZONE_ID}),

    # --- zone-scoped: get_zone_details ("confidence") -----------------------------------------
    ("confidence (canonical, needs selected zone)", "How confident are we in this zone?", ZONE_ID,
     "get_zone_details", {"zone_id": ZONE_ID}),
    ("confidence (paraphrase: implicit selected zone)", "How confident are we?", ZONE_ID,
     "get_zone_details", {"zone_id": ZONE_ID}),

    # --- zone-scoped: get_zone_trend ("what changed") -----------------------------------------
    ("what changed (canonical, needs selected zone)", "What changed since the previous quarter?", ZONE_ID,
     "get_zone_trend", {"zone_id": ZONE_ID}),
    ("what changed (paraphrase: 'since last quarter')", "How has this changed since last quarter?", ZONE_ID,
     "get_zone_trend", {"zone_id": ZONE_ID}),

    # --- zone-scoped: get_zone_details ("measurements") ---------------------------------------
    ("measurements (canonical, needs selected zone)", "Which measurements support this recommendation?", ZONE_ID,
     "get_zone_details", {"zone_id": ZONE_ID}),
    ("measurements (paraphrase: 'behind this recommendation')",
     "What measurements are behind this recommendation?", ZONE_ID,
     "get_zone_details", {"zone_id": ZONE_ID}),

    # --- zone-scoped: get_zone_peer_comparison --------------------------------------------------
    ("peer comparison (paraphrase, needs selected zone)", "How does this zone compare with its peer group?", ZONE_ID,
     "get_zone_peer_comparison", {"zone_id": ZONE_ID}),
]


def test_every_canonical_intent_paraphrase_routes_correctly():
    rows = []
    failures = []
    for label, question, zone_id, expected_tool, expected_args in CASES:
        tool, args = copilot.route(question, zone_id=zone_id)
        rows.append((label, question, tool, args))
        if (tool, args) != (expected_tool, expected_args):
            failures.append(
                f"[{label}] {question!r} (zone_id={zone_id!r}) -> got ({tool!r}, {args!r}), "
                f"expected ({expected_tool!r}, {expected_args!r})"
            )

    print(f"\n{'intent':52} {'question':52} tool -> args")
    print("-" * 160)
    for label, question, tool, args in rows:
        print(f"{label:52} {question:52} {tool} -> {args}")

    assert not failures, "\n" + "\n".join(failures)


# --- Non-tool intents: refusal / fixed-fact checks stay routed to no tool at all, and every
# canonical + paraphrased geographic answer above stays free of the internal field/methodology
# vocabulary a decision-maker was never meant to see (that translation is narration-layer only
# -- src/copilot_tools.py, src/priority.py, src/compute_scores.py compute the exact same numbers
# either way; only the words describing them changed). ------------------------------------------

REFUSAL_CASES = [
    ("data source question -> fixed fact, no tool", "Is this e& network data?", "fixed_fact"),
    ("operator-attribution question -> refusal, no tool", "Which e& site is causing this poor experience?", "refusal"),
]

_FORBIDDEN_JARGON = [
    "peer underperformance", "temporal anomaly", "population exposure",
    "evidence tier", "percentile rank", "isolation forest",
]


def test_refusal_and_fixed_fact_questions_route_to_no_tool():
    for label, question, expected_mode in REFUSAL_CASES:
        result = copilot.answer_question(question)
        assert result["tool"] is None, f"[{label}] {question!r} unexpectedly selected a tool: {result['tool']}"
        assert result["mode"] == expected_mode, f"[{label}] {question!r} -> mode {result['mode']!r}, expected {expected_mode!r}"
        print(f"[{label}] {question!r} -> tool=None, mode={result['mode']!r}")


def test_geographic_answers_avoid_internal_jargon():
    zone_id = copilot.ct.get_top_priority_zones(1)[0]["zone_id"]
    questions = [(q, z) for _, q, z, _, _ in CASES if z is None or z == ZONE_ID]
    bad = []
    for question, _ in questions:
        result = copilot.answer_question(question, zone_id=zone_id)
        answer = (result["answer"] or "").lower()
        for term in _FORBIDDEN_JARGON:
            if term in answer:
                bad.append((question, term, result["answer"]))
    assert not bad, "\n" + "\n".join(f"{q!r} used forbidden term {t!r}: {a!r}" for q, t, a in bad)
    print(f"Checked {len(questions)} answers for internal jargon: none found.")


# --- End-to-end regression tests for the three exact questions this fix targets: correct tool,
# correct returned count, and the H3 ids the frontend would highlight (extractZoneRecords /
# setCopilotHighlight in src/build_dashboard.py just map tool_result -> [z.zone_id for z in
# tool_result], never anything parsed from the narrated text) exactly equal what an independent
# recomputation over the raw parquet data says they should be -- not just "some list of the
# right length." -----------------------------------------------------------------------------
import pandas as pd

ZONE_PRIORITY_PATH = "data/processed/zone_priority.parquet"


def _highlighted_ids(tool_result: list[dict]) -> list[str]:
    """Mirrors the frontend's extractZoneRecords()/setCopilotHighlight() exactly: a listing
    tool's result is a list of zone dicts, and the map highlights precisely their zone_id
    values, in order -- this is what "the map must never parse the LLM's prose" means in
    practice, and this helper is that same rule expressed as a Python-side assertion."""
    return [z["zone_id"] for z in tool_result]


def _latest_classified() -> pd.DataFrame:
    df = pd.read_parquet(ZONE_PRIORITY_PATH)
    latest = sorted(df["quarter"].unique())[-1]
    return df[(df["quarter"] == latest) & (~df["insufficient_evidence"])]


def test_median_download_question_matches_independent_recomputation():
    """'Which zones have greater than the median download' -- previously unmatched entirely."""
    result = copilot.answer_question("Which zones have greater than the median download")
    assert result["tool"] == "get_metric_threshold_zones", result["tool"]
    assert result["tool_args"] == {"metric": "download_mbps", "comparison": "above"}, result["tool_args"]
    tool_result = result["tool_result"]
    highlighted = _highlighted_ids(tool_result)
    assert len(highlighted) == len(set(highlighted)), "duplicate zone_id in tool_result"

    classified = _latest_classified()
    median_download = classified["download_mbps"].median()
    expected_ids = set(classified.loc[classified["download_mbps"] > median_download, "h3_cell"])

    assert set(highlighted) == expected_ids, (
        f"{len(set(highlighted) ^ expected_ids)} zone(s) differ from the independently "
        f"recomputed 'above median download' set"
    )
    print(f"'Which zones have greater than the median download' -> {len(tool_result)} zones "
          f"(median {median_download:.2f} Mbps), exactly matches independent recomputation.")


# --- Metric-threshold intent, both directions x all 3 raw metrics (new capability): "show me
# the areas that scored higher/lower than the median download/upload/latency" -- previously only
# 'download'+'above' was wired up at all ('below' silently fell through to the median-VALUE
# question and answered only the number). One resolver/tool now covers both directions for all
# three raw metrics -- these prove each combination routes correctly, is scoped/filtered by the
# deterministic backend (never the LLM), and matches an independent recomputation of the median
# and the filtered set from the raw parquet data. --------------------------------------------

METRIC_THRESHOLD_EXAMPLE_QUESTIONS = [
    # (question, metric, comparison, dataframe column the metric actually reads)
    ("show me the areas that scored higher than the median download", "download_mbps", "above", "download_mbps"),
    ("show me the areas that scored lower than the median download", "download_mbps", "below", "download_mbps"),
    ("areas above median upload", "upload_mbps", "above", "upload_mbps"),
    ("areas below median upload", "upload_mbps", "below", "upload_mbps"),
    ("areas above median latency", "latency_ms", "above", "latency_effective_ms"),
    ("areas below median latency", "latency_ms", "below", "latency_effective_ms"),
]


def test_metric_threshold_examples_route_and_match_independent_recomputation():
    """The exact regression set requested for this fix. Each of the 6 (metric x direction)
    combinations checked end to end: correct tool/metric/comparison resolved from text alone,
    the returned zones exactly equal an independent recomputation of 'value > median' or
    'value < median' over the raw parquet data (never just 'some zones of the right shape'),
    and the map highlight (extractZoneRecords/setCopilotHighlight in build_dashboard.py) is
    exactly the tool's own zone_id set, in order."""
    classified = _latest_classified()
    for question, metric, comparison, col in METRIC_THRESHOLD_EXAMPLE_QUESTIONS:
        result = copilot.answer_question(question)
        assert result["tool"] == "get_metric_threshold_zones", f"{question!r}: tool {result['tool']!r}"
        assert result["tool_args"].get("metric") == metric, (
            f"{question!r}: metric {result['tool_args'].get('metric')!r}, expected {metric!r}"
        )
        assert result["tool_args"].get("comparison") == comparison, (
            f"{question!r}: comparison {result['tool_args'].get('comparison')!r}, expected {comparison!r}"
        )
        tool_result = result["tool_result"]
        assert tool_result, f"{question!r}: expected at least one zone"
        highlighted = _highlighted_ids(tool_result)
        assert len(highlighted) == len(set(highlighted)), f"{question!r}: duplicate zone_id in tool_result"
        assert highlighted == [z["zone_id"] for z in tool_result], (
            f"{question!r}: highlighted ids must exactly equal tool_result's own zone_id order"
        )

        median_value = round(float(classified[col].median()), 2)
        if comparison == "above":
            expected_ids = set(classified.loc[classified[col] > median_value, "h3_cell"])
        else:
            expected_ids = set(classified.loc[classified[col] < median_value, "h3_cell"])
        assert set(highlighted) == expected_ids, (
            f"{question!r}: {len(set(highlighted) ^ expected_ids)} zone(s) differ from the "
            f"independently recomputed '{comparison} median {metric}' set"
        )
        print(f"{question!r} -> {result['tool']}, {len(tool_result)} zone(s) (median {median_value}), "
              f"exactly matches independent recomputation.")


def test_above_and_below_median_download_return_disjoint_opposite_sets():
    """'higher than the median download' and 'lower than the median download' must return
    disjoint sets that, together with any zones exactly AT the median, account for the entire
    classified population -- the exact regression this fix targets: 'below' used to silently
    fall through to the median-VALUE question and never return a zone list at all."""
    classified = _latest_classified()
    above = copilot.answer_question("show me the areas that scored higher than the median download")
    below = copilot.answer_question("show me the areas that scored lower than the median download")
    assert above["tool"] == "get_metric_threshold_zones", above["tool"]
    assert below["tool"] == "get_metric_threshold_zones", below["tool"]
    assert above["tool_args"]["comparison"] == "above", above["tool_args"]
    assert below["tool_args"]["comparison"] == "below", below["tool_args"]

    above_ids = set(_highlighted_ids(above["tool_result"]))
    below_ids = set(_highlighted_ids(below["tool_result"]))
    assert above_ids, "expected at least one zone above the median"
    assert below_ids, "expected at least one zone below the median"
    assert not (above_ids & below_ids), f"above/below sets must never overlap: {above_ids & below_ids}"

    median_download = round(float(classified["download_mbps"].median()), 2)
    at_median = int((classified["download_mbps"] == median_download).sum())
    assert len(above_ids) + len(below_ids) + at_median == len(classified), (
        "above + below + exactly-at-median must account for every classified zone"
    )
    print(f"'higher than median download' -> {len(above_ids)} zones, 'lower than median download' -> "
          f"{len(below_ids)} zones, {at_median} exactly at the median ({median_download} Mbps) -- "
          f"disjoint, sums to the full classified population ({len(classified)} zones).")


def test_below_median_narration_matches_exact_requested_wording():
    """Regression guard for the exact short-answer wording requested: 'These are the areas with
    download speeds below the median for <quarter>. I highlighted them on the map.'"""
    result = copilot.answer_question("show me the areas that scored lower than the median download")
    quarter = result["tool_result"][0]["quarter"]
    assert result["answer"] == (
        f"These are the areas with download speeds below the median for {quarter}. "
        f"I highlighted them on the map."
    ), result["answer"]
    print(f"Narration matches exactly: {result['answer']!r}")


def test_median_value_question_never_highlights_zones():
    """'What is the median download?' must return the number only -- get_metric_extreme with
    zone_id=None (the backend equivalent of 'nothing highlighted': build_dashboard.py's
    GEO_TOOLS/extractZoneRecords only ever highlight a real zone_id, and get_metric_extreme's
    own docstring is explicit that 'median' has no single zone that IS the median)."""
    result = copilot.answer_question("what is the median download")
    assert result["tool"] == "get_metric_extreme", result["tool"]
    assert result["tool_args"] == {"metric": "download_mbps", "operation": "median"}, result["tool_args"]
    assert result["tool_result"]["zone_id"] is None, "a median-VALUE question must never name a zone to highlight"
    expected_median = round(float(_latest_classified()["download_mbps"].median()), 2)
    assert result["tool_result"]["value"] == expected_median
    print(f"'what is the median download' -> value {result['tool_result']['value']} Mbps, "
          f"zone_id=None (nothing highlighted), matches independent recomputation.")


def test_lowest_experience_question_returns_exactly_one_zone():
    """'Which zone has the lowest experience' -- previously unmatched entirely (no 'weak'
    keyword). Must return, and highlight, exactly the one true minimum -- not a top-10 list."""
    result = copilot.answer_question("Which zone has the lowest experience")
    assert result["tool"] == "get_weakest_zones", result["tool"]
    assert result["tool_args"] == {"n": 1}, result["tool_args"]
    tool_result = result["tool_result"]
    assert len(tool_result) == 1, f"expected exactly 1 zone, got {len(tool_result)}"
    highlighted = _highlighted_ids(tool_result)

    classified = _latest_classified()
    expected_id = classified.loc[classified["experience_index"].idxmin(), "h3_cell"]
    assert highlighted == [expected_id], f"got {highlighted}, expected the true minimum [{expected_id!r}]"
    print("'Which zone has the lowest experience' -> exactly 1 zone highlighted, "
          "matches the true minimum Experience Index in the data.")


def test_top_n_deterioration_question_returns_exactly_n_strongest():
    """'show me the top 5 most deterioration zones' used to return all 17 flagged zones,
    ignoring the requested N entirely. Also covers top 3 and the singular 'the most' -> n=1."""
    for n, question in [
        (5, "show me the top 5 most deterioration zones"),
        (3, "top 3 deteriorating zones"),
        (1, "which area deteriorated the most?"),
    ]:
        result = copilot.answer_question(question)
        assert result["tool"] == "get_deteriorating_zones", (question, result["tool"])
        assert result["tool_args"] == {"n": n}, (question, result["tool_args"])
        tool_result = result["tool_result"]
        assert len(tool_result) == n, f"[{question!r}] expected exactly {n} zones, got {len(tool_result)}"
        highlighted = _highlighted_ids(tool_result)

        classified = _latest_classified()
        flagged = classified[classified["deteriorating"]].sort_values("trend_pts_per_qtr", ascending=True)
        expected_ids = flagged.head(n)["h3_cell"].tolist()

        assert highlighted == expected_ids, (
            f"[{question!r}] highlighted {highlighted} != independently recomputed top-{n} {expected_ids}"
        )
        print(f"{question!r} -> exactly {n} zone(s), matches the {n} strongest "
              f"independently-recomputed deterioration cases.")

    # Regression guard: the bug this fix corrects was ignoring N and returning every flagged
    # zone -- this proves the unquantified canonical question ("no number supplied") still gets
    # the original, unchanged "all flagged zones" behavior, not an accidental n=1/n=10 default.
    result = copilot.answer_question("Which areas are deteriorating?")
    assert result["tool_args"] == {}, result["tool_args"]
    expected_total = int(_latest_classified()["deteriorating"].sum())
    assert len(result["tool_result"]) == expected_total
    print(f"'Which areas are deteriorating?' (no N) -> all {expected_total} flagged zones, default unchanged.")


# --- Experience-ranking, 'highest' direction (new capability): "where is experience the
# highest" used to be entirely unmatched (only 'lowest'/'weakest'/'worst' routed anywhere), even
# though "where is experience the weakest" already worked. These prove both directions now share
# one structured intent (metric=experience, direction=highest|lowest), the returned zones are
# actually sorted by Experience Index in the requested direction (never just "some N zones of
# the right shape"), and the map highlights exactly the tool's own zone_id set. -----------------

def test_highest_experience_question_returns_exactly_one_zone():
    """'Which zone has the highest experience' -- mirrors the existing lowest/singular
    regression test above, opposite direction. Must return, and highlight, exactly the one true
    maximum -- not a top-10 list."""
    result = copilot.answer_question("Which zone has the highest experience")
    assert result["tool"] == "get_strongest_zones", result["tool"]
    assert result["tool_args"] == {"n": 1}, result["tool_args"]
    tool_result = result["tool_result"]
    assert len(tool_result) == 1, f"expected exactly 1 zone, got {len(tool_result)}"
    highlighted = _highlighted_ids(tool_result)

    classified = _latest_classified()
    expected_id = classified.loc[classified["experience_index"].idxmax(), "h3_cell"]
    assert highlighted == [expected_id], f"got {highlighted}, expected the true maximum [{expected_id!r}]"
    print("'Which zone has the highest experience' -> exactly 1 zone highlighted, "
          "matches the true maximum Experience Index in the data.")


def test_experience_ranking_both_directions_sorted_and_highlighted_correctly():
    """'where is experience the highest' / 'where is experience the weakest' -- the map must
    highlight exactly the tool's own zone_id set, in the tool's own order (never reparsed from
    narration), and that order must be true Experience Index rank, independently recomputed from
    the raw data. Also proves the two directions don't secretly share a sort order (they'd
    overlap entirely at n=10 if 'highest' silently reused 'lowest''s ascending sort)."""
    classified = _latest_classified()

    highest = copilot.answer_question("where is experience the highest")
    assert highest["tool"] == "get_strongest_zones", highest["tool"]
    assert len(highest["tool_result"]) == 10
    highlighted_highest = _highlighted_ids(highest["tool_result"])
    expected_highest = classified.sort_values("experience_index", ascending=False).head(10)["h3_cell"].tolist()
    assert highlighted_highest == expected_highest, (highlighted_highest, expected_highest)
    highest_values = [z["experience_index"] for z in highest["tool_result"]]
    assert highest_values == sorted(highest_values, reverse=True), "highest-direction result not sorted descending"

    weakest = copilot.answer_question("where is experience the weakest")
    assert weakest["tool"] == "get_weakest_zones", weakest["tool"]
    assert len(weakest["tool_result"]) == 10
    highlighted_weakest = _highlighted_ids(weakest["tool_result"])
    expected_weakest = classified.sort_values("experience_index", ascending=True).head(10)["h3_cell"].tolist()
    assert highlighted_weakest == expected_weakest, (highlighted_weakest, expected_weakest)
    weakest_values = [z["experience_index"] for z in weakest["tool_result"]]
    assert weakest_values == sorted(weakest_values), "lowest-direction result not sorted ascending"

    assert not (set(highlighted_highest) & set(highlighted_weakest)), (
        "highest and weakest result sets overlap -- direction not actually applied"
    )
    print(f"'where is experience the highest' -> {len(highest['tool_result'])} zones, sorted "
          f"descending, matches independent recomputation. 'where is experience the weakest' -> "
          f"{len(weakest['tool_result'])} zones, sorted ascending, matches independent "
          f"recomputation. No overlap between the two extremes.")


def test_strongest_listing_narration_matches_exact_requested_wording():
    """Regression guard for the exact short-answer wording requested: 10 areas, no emirate ->
    'These are the 10 areas with the highest measured mobile experience in <quarter>. I
    highlighted them on the map.'"""
    result = copilot.answer_question("where is experience the highest")
    quarter = result["tool_result"][0]["quarter"]
    assert result["answer"] == (
        f"These are the 10 areas with the highest measured mobile experience in {quarter}. "
        f"I highlighted them on the map."
    ), result["answer"]
    print(f"Narration matches exactly: {result['answer']!r}")


EXPERIENCE_DIRECTION_EXAMPLE_QUESTIONS = [
    # (question, expected_direction, expected_emirate) -- the exact regression set requested,
    # covering both directions, a metric typo, and emirate scoping.
    ("where is experience the highest", "highest", None),
    ("which zone has the highest experience", "highest", None),
    ("show me top 5 best experience areas", "highest", None),
    ("highest experince in rak", "highest", "Ras Al Khaimah"),
    ("best experiance in auh", "highest", "Abu Dhabi"),
    ("where is experience the weakest", "lowest", None),
    ("worst experience in shj", "lowest", "Sharjah"),
]


def test_experience_direction_examples_route_scope_and_sort_correctly():
    """The exact 7 regression questions requested for this fix, checked end to end: correct
    tool, correct emirate confinement (deterministic filtering, never LLM-narrated around an
    unfiltered result), Experience Index actually sorted in the requested direction, and the map
    highlight exactly equal to the tool's own returned zone_id order."""
    for question, direction, expected_emirate in EXPERIENCE_DIRECTION_EXAMPLE_QUESTIONS:
        expected_tool = "get_strongest_zones" if direction == "highest" else "get_weakest_zones"
        result = copilot.answer_question(question)
        assert result["tool"] == expected_tool, f"{question!r}: tool {result['tool']!r}, expected {expected_tool!r}"
        assert result["tool_args"].get("emirate") == expected_emirate, (
            f"{question!r}: emirate {result['tool_args'].get('emirate')!r}, expected {expected_emirate!r}"
        )
        tool_result = result["tool_result"]
        assert tool_result, f"{question!r}: expected at least one zone"
        if expected_emirate:
            wrong_emirate = [z for z in tool_result if z["emirate"] != expected_emirate]
            assert not wrong_emirate, f"{question!r}: zone(s) outside {expected_emirate}: {wrong_emirate}"

        exp_values = [z["experience_index"] for z in tool_result]
        expected_sorted = sorted(exp_values, reverse=(direction == "highest"))
        assert exp_values == expected_sorted, f"{question!r}: not sorted {direction} ({exp_values})"

        highlighted = _highlighted_ids(tool_result)
        assert highlighted == [z["zone_id"] for z in tool_result], (
            f"{question!r}: highlighted ids must exactly equal tool_result's own zone_id order"
        )
        print(f"{question!r} -> {expected_tool}, direction={direction}, emirate={expected_emirate}, "
              f"{len(tool_result)} zone(s), sorted correctly, map highlight matches exactly.")


def test_experience_question_without_direction_asks_for_clarification():
    """'experience' alone, with no direction word at all, must ask -- never silently default to
    one direction. Same clarification contract test_low_confidence_metric_typo_... already
    proves for the raw-metric resolver, applied here to the Experience-ranking resolver."""
    result = copilot.answer_question("tell me about experience")
    assert result["tool"] is None, result["tool"]
    assert result["mode"] == "clarification", result["mode"]
    assert "experience" in result["answer"].lower()
    print(f"No-direction experience question -> mode=clarification, answer={result['answer']!r}")


# --- Emirate alias resolution: "which zones has worst experience in rak" used to be answered
# nationally (RAK never recognized as anything, so `emirate` stayed None) -- these prove the
# whole EMIRATE_ALIASES table resolves to the correct canonical name, is applied BEFORE the tool
# runs (never inferred/filtered by the LLM afterward), and that the returned zones are actually
# confined to that emirate, not just that the routing dict looks right. ------------------------

# The brief's own required alias list -- kept literally here (not "whatever EMIRATE_ALIASES
# happens to contain") so a future edit that silently drops a required alias fails this test
# instead of just quietly losing coverage.
REQUIRED_EMIRATE_ALIASES = {
    "Abu Dhabi": ["abu dhabi", "abudhabi", "auh", "ad"],
    "Dubai": ["dubai", "dxb"],
    "Sharjah": ["sharjah", "shj"],
    "Ajman": ["ajman", "ajm"],
    "Umm Al Quwain": ["umm al quwain", "umm al-quwain", "uaq"],
    "Ras Al Khaimah": ["ras al khaimah", "ras al-khaimah", "rak"],
    "Fujairah": ["fujairah", "fuj", "fjr"],
}

# The exact 8 example questions from the request (RAK headline case + the 7 "also test" ones).
EMIRATE_EXAMPLE_QUESTIONS = [
    ("which zones has worst experience in rak", "Ras Al Khaimah", "get_weakest_zones"),
    ("worst zones in AUH", "Abu Dhabi", "get_weakest_zones"),
    ("lowest experience in AD", "Abu Dhabi", "get_weakest_zones"),
    ("weakest areas in SHJ", "Sharjah", "get_weakest_zones"),
    ("top priority in DXB", "Dubai", "get_top_priority_zones"),
    ("deteriorating zones in RAK", "Ras Al Khaimah", "get_deteriorating_zones"),
    ("weak experience in UAQ", "Umm Al Quwain", "get_weakest_zones"),
    ("worst area in FUJ", "Fujairah", "get_weakest_zones"),
]

# Short codes must never match as a substring of an unrelated word -- 'ad' inside 'advance',
# 'shj'/'uaq'/etc. never occurring in plain English at all, but 'ad' is the real risk case.
EMIRATE_FALSE_POSITIVE_GUARD = [
    "advance notice needed for the deployment",
    "load balancing across the network",
    "additional measurements would help",
]


def test_emirate_alias_table_matches_the_required_list():
    assert copilot.EMIRATE_ALIASES == REQUIRED_EMIRATE_ALIASES, (
        "EMIRATE_ALIASES no longer matches the brief's required alias table -- "
        f"got {copilot.EMIRATE_ALIASES}"
    )


def test_every_required_alias_resolves_case_and_hyphen_insensitively():
    failures = []
    checked = 0
    for canonical, aliases in REQUIRED_EMIRATE_ALIASES.items():
        for alias in aliases:
            for phrasing in (
                f"weakest zones in {alias}",
                f"WEAKEST ZONES IN {alias.upper()}",
                f"weakest zones in {alias.replace(' ', '-')}",  # hyphenated form, even for aliases already given with spaces
            ):
                checked += 1
                got = copilot._extract_emirate(phrasing)
                if got != canonical:
                    failures.append(f"{phrasing!r} -> {got!r}, expected {canonical!r}")
    assert not failures, "\n" + "\n".join(failures)
    print(f"Checked {checked} alias/casing/hyphenation combinations across all 7 emirates: all resolved correctly.")


def test_short_emirate_codes_do_not_false_positive_inside_unrelated_words():
    bad = [(q, copilot._extract_emirate(q)) for q in EMIRATE_FALSE_POSITIVE_GUARD if copilot._extract_emirate(q) is not None]
    assert not bad, f"short alias code false-matched inside an unrelated word: {bad}"
    print(f"Checked {len(EMIRATE_FALSE_POSITIVE_GUARD)} unrelated-word cases: no false-positive emirate match.")


def test_emirate_examples_resolve_before_routing_and_scope_the_tool_result():
    """The 8 example questions: emirate must be resolved to its canonical name and land in
    `route()`'s args (resolved BEFORE the tool runs), and every zone the tool actually returns
    must belong to that exact emirate -- proving the deterministic tool did the filtering, not
    the LLM narrating around an unfiltered national result."""
    for question, expected_emirate, expected_tool in EMIRATE_EXAMPLE_QUESTIONS:
        tool, args = copilot.route(question)
        assert tool == expected_tool, f"{question!r}: tool {tool!r}, expected {expected_tool!r}"
        assert args.get("emirate") == expected_emirate, (
            f"{question!r}: route() resolved emirate={args.get('emirate')!r}, expected {expected_emirate!r}"
        )

        result = copilot.answer_question(question)
        assert result["tool_args"].get("emirate") == expected_emirate, (
            f"{question!r}: answer_question tool_args={result['tool_args']}"
        )
        tool_result = result["tool_result"]
        assert tool_result, f"{question!r}: expected at least one zone in {expected_emirate}, got none"
        wrong_emirate = [z for z in tool_result if z["emirate"] != expected_emirate]
        assert not wrong_emirate, (
            f"{question!r}: {len(wrong_emirate)} returned zone(s) not in {expected_emirate}: "
            f"{[(z['zone_id'], z['emirate']) for z in wrong_emirate]}"
        )
        highlighted = _highlighted_ids(tool_result)
        assert highlighted == [z["zone_id"] for z in tool_result], "highlighted ids must exactly equal tool_result's own zone_id order"
        assert expected_emirate.lower() in result["answer"].lower(), (
            f"{question!r}: answer does not name the resolved emirate: {result['answer']!r}"
        )
        print(f"{question!r} -> {expected_tool}, emirate={expected_emirate}, "
              f"{len(tool_result)} zone(s), all confined to {expected_emirate}.")


# --- Metric-extreme (raw measurement) questions: "lowest recorded download speed" used to be
# misrouted entirely into get_weakest_zones -- an EXPERIENCE INDEX question, not a raw download/
# upload/latency question. These prove the router now tells them apart, resolves
# intent/metric/operation/emirate from TEXT ALONE before any tool runs (never inferred by the
# LLM), and that the returned value is the true, independently-recomputed min/max/median from
# the raw data -- not just "some number of the right shape." ------------------------------------

# (question, expected metric, expected operation, expected emirate)
METRIC_EXAMPLE_QUESTIONS = [
    ("what is the lowest recorded download speed", "download_mbps", "min", None),
    ("what is the lowest recorded upload speed", "upload_mbps", "min", None),
    ("what is the highest download speed", "download_mbps", "max", None),
    ("what is the highest latency", "latency_ms", "max", None),
    ("lowest downlod in rak", "download_mbps", "min", "Ras Al Khaimah"),
    ("higest uplod in shj", "upload_mbps", "max", "Sharjah"),
]

# metric key (as used everywhere in the copilot layer) -> the actual dataframe column it reads.
# latency_ms deliberately reads latency_effective_ms -- see get_metric_extreme's own docstring.
_METRIC_RAW_COLUMN = {"download_mbps": "download_mbps", "upload_mbps": "upload_mbps", "latency_ms": "latency_effective_ms"}


def test_metric_extreme_examples_route_correctly_and_return_the_true_value():
    print(f"\n{'normalized text':45} {'intent':15} {'metric':13} {'op':6} {'emirate':16} tool -> value")
    print("-" * 145)
    for question, expected_metric, expected_op, expected_emirate in METRIC_EXAMPLE_QUESTIONS:
        normalized = copilot._normalize_geo_text(question)
        result = copilot.answer_question(question)

        assert result["tool"] == "get_metric_extreme", f"{question!r} -> tool {result['tool']!r}"
        args = result["tool_args"]
        assert args.get("metric") == expected_metric, f"{question!r}: metric {args.get('metric')!r}"
        assert args.get("operation") == expected_op, f"{question!r}: operation {args.get('operation')!r}"
        assert args.get("emirate") == expected_emirate, f"{question!r}: emirate {args.get('emirate')!r}"

        tool_result = result["tool_result"]
        classified = _latest_classified()
        if expected_emirate:
            classified = classified[classified["emirate"] == expected_emirate]
        col = _METRIC_RAW_COLUMN[expected_metric]
        expected_value = round(float(classified[col].min() if expected_op == "min" else classified[col].max()), 2)
        assert tool_result["value"] == expected_value, (
            f"{question!r}: value {tool_result['value']}, expected {expected_value} (independently recomputed)"
        )
        expected_zone = classified.loc[
            classified[col].idxmin() if expected_op == "min" else classified[col].idxmax(), "h3_cell"
        ]
        assert tool_result["zone_id"] == expected_zone, (
            f"{question!r}: zone_id {tool_result['zone_id']!r}, expected {expected_zone!r}"
        )
        assert result["answer"].strip(), f"{question!r}: empty narration"

        print(f"{normalized:45} {'metric_extreme':15} {expected_metric:13} {expected_op:6} "
              f"{str(expected_emirate):16} {result['tool']} -> {tool_result['value']} {tool_result['unit']}")
    print(f"\nAll {len(METRIC_EXAMPLE_QUESTIONS)} metric-extreme questions matched their true, "
          f"independently-recomputed min/max value and zone.")


def test_metric_examples_never_route_to_experience_weakest():
    """The exact regression this fix targets: these must NOT land on get_weakest_zones (an
    Experience Index tool) just because they contain 'lowest'/'highest'."""
    for question, _, _, _ in METRIC_EXAMPLE_QUESTIONS:
        tool, _ = copilot.route(question)
        assert tool != "get_weakest_zones", f"{question!r} still misrouted to get_weakest_zones"
        assert tool != "get_top_priority_zones", f"{question!r} still misrouted to get_top_priority_zones"
    print("Confirmed: none of the metric-extreme examples route to Experience-Index tools.")


def test_metric_extreme_structured_request_shape():
    """The exact structured-request shape requested: intent=metric_extreme, metric, operation,
    quarter, emirate -- resolved before the tool runs, never left for the LLM to infer."""
    result = copilot.answer_question("what is the lowest recorded download speed")
    structured = {
        "intent": "metric_extreme",  # route() -> tool_name "get_metric_extreme" IS this intent
        "metric": result["tool_args"]["metric"],
        "operation": result["tool_args"]["operation"],
        "quarter": result["tool_result"]["quarter"],
        "emirate": result["tool_args"].get("emirate"),
    }
    assert structured == {
        "intent": "metric_extreme", "metric": "download_mbps", "operation": "min",
        "quarter": result["tool_result"]["quarter"], "emirate": None,
    }
    print(f"Structured request: {structured}")


def test_median_download_listing_and_median_value_questions_stay_distinguished():
    """Regression guard: 'which zones have greater than the median download' (a LISTING of
    zones) and 'what is the median download speed' (a single VALUE) both contain 'median'
    and 'download' -- the comparison word ('greater than') is what tells them apart, and this
    must keep working after adding metric-extreme median support."""
    listing = copilot.answer_question("Which zones have greater than the median download")
    assert listing["tool"] == "get_metric_threshold_zones", listing["tool"]
    assert isinstance(listing["tool_result"], list) and len(listing["tool_result"]) > 1

    value = copilot.answer_question("what is the median download speed")
    assert value["tool"] == "get_metric_extreme", value["tool"]
    assert value["tool_args"] == {"metric": "download_mbps", "operation": "median"}
    assert isinstance(value["tool_result"], dict) and value["tool_result"]["zone_id"] is None
    expected_median = round(float(_latest_classified()["download_mbps"].median()), 2)
    assert value["tool_result"]["value"] == expected_median
    print(f"Listing question -> {len(listing['tool_result'])} zones (get_metric_threshold_zones). "
          f"Value question -> single value {value['tool_result']['value']} Mbps (get_metric_extreme), "
          f"no zone_id -- correctly distinguished.")


def test_low_confidence_metric_typo_asks_for_clarification_instead_of_guessing():
    """A typo real enough to suggest SOMETHING but not confident enough to auto-run must ask,
    never silently guess and run a tool. A question with no metric signal at all stays
    'unmatched' as before -- the fuzzy layer must not start firing on unrelated questions."""
    result = copilot.answer_question("tell me the dnwlod thing")
    assert result["tool"] is None, result["tool"]
    assert result["mode"] == "clarification", result["mode"]
    assert "download" in result["answer"].lower()
    assert "?" in result["answer"]
    print(f"Low-confidence typo -> mode=clarification, answer={result['answer']!r}")

    unrelated = copilot.answer_question("asdkjasjdaksjd")
    assert unrelated["tool"] is None and unrelated["mode"] == "unmatched", unrelated["mode"]
    print("Unrelated gibberish -> mode=unmatched (fuzzy layer did not false-trigger).")


if __name__ == "__main__":
    test_every_canonical_intent_paraphrase_routes_correctly()
    test_refusal_and_fixed_fact_questions_route_to_no_tool()
    test_geographic_answers_avoid_internal_jargon()
    test_median_download_question_matches_independent_recomputation()
    test_metric_threshold_examples_route_and_match_independent_recomputation()
    test_above_and_below_median_download_return_disjoint_opposite_sets()
    test_below_median_narration_matches_exact_requested_wording()
    test_median_value_question_never_highlights_zones()
    test_lowest_experience_question_returns_exactly_one_zone()
    test_top_n_deterioration_question_returns_exactly_n_strongest()
    test_highest_experience_question_returns_exactly_one_zone()
    test_experience_ranking_both_directions_sorted_and_highlighted_correctly()
    test_strongest_listing_narration_matches_exact_requested_wording()
    test_experience_direction_examples_route_scope_and_sort_correctly()
    test_experience_question_without_direction_asks_for_clarification()
    test_emirate_alias_table_matches_the_required_list()
    test_every_required_alias_resolves_case_and_hyphen_insensitively()
    test_short_emirate_codes_do_not_false_positive_inside_unrelated_words()
    test_emirate_examples_resolve_before_routing_and_scope_the_tool_result()
    test_metric_extreme_examples_route_correctly_and_return_the_true_value()
    test_metric_examples_never_route_to_experience_weakest()
    test_metric_extreme_structured_request_shape()
    test_median_download_listing_and_median_value_questions_stay_distinguished()
    test_low_confidence_metric_typo_asks_for_clarification_instead_of_guessing()
    print("T7 PASSED.")
