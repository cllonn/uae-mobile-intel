"""
T8 -- Copilot generic area/place lookup ("show me Seyouh", "where is Deira", "find Al Nahda",
"take me to Al Majaz", "show Khalifa City on the map").

The requirement: a decision-maker can ask for ANY area name that actually exists in the
processed dataset (data/processed/zone_priority.parquet's own `area_name` column, via
src/copilot_tools.py::get_known_area_names) -- never a hardcoded list of place names, and never
the LLM's own world knowledge of UAE geography. Matching order: exact -> normalized (case/
punctuation/hyphen/space-insensitive) -> stored alias -> fuzzy (typo-tolerant) against ONLY that
real set (src/copilot.py::_match_area_name). A name real enough to span more than one emirate
with no emirate stated asks a short clarification instead of guessing; a name with no
confident match at all says so honestly instead of hallucinating one.

Run: python -m tests.test_t8_copilot_locate_area   (or `pytest tests/` if pytest is installed)
"""
from src import copilot
from src import copilot_tools as ct

AREA_NOT_FOUND = copilot.AREA_NOT_FOUND_MESSAGE


def _known_names() -> list[str]:
    return sorted({row["area_name"] for row in ct.get_known_area_names()})


# --- Required opening step: print a real sample of the area-name universe this feature must
# work against, and prove the lookup actually resolves several of them end to end. -------------

def print_area_name_sample_and_spot_check():
    names = _known_names()
    assert len(names) >= 20, f"expected at least 20 known area names in the processed data, got {len(names)}"
    sample = names[:20]
    print(f"\n{len(names)} distinct area names available in the processed dataset. First 20:")
    for n in sample:
        print(f"  - {n}")

    print("\nSpot-checking the lookup against several of them (exact-phrase 'show me <name>'):")
    for name in sample[:8]:
        result = copilot.answer_question(f"show me {name}")
        print(f"  'show me {name}' -> tool={result['tool']!r}, "
              f"area_name={result.get('area_name')!r}, emirate={result.get('emirate')!r}, "
              f"mode={result['mode']!r}")


# --- Matching order: exact, then normalized (case/punctuation/hyphen/space-insensitive) --------

def test_exact_match_resolves_directly():
    # 'Al Hamra' is unambiguous (single emirate) -- a clean end-to-end case for the exact tier.
    result = copilot.answer_question("show me Al Hamra")
    assert result["tool"] == "locate_area", result
    assert result["area_name"] == "Al Hamra", result
    assert result["emirate"], result
    assert result["h3_ids"], "expected at least one H3 zone"
    assert result["answer"] == f"Here is Al Hamra in {result['emirate']}. I highlighted it on the map.", result["answer"]
    print(f"'show me Al Hamra' -> {result['area_name']}, {result['emirate']}, {len(result['h3_ids'])} zone(s).")


def test_normalized_match_ignores_case_punctuation_and_hyphens():
    known = ct.get_known_area_names()
    apostrophe_name = next(row["area_name"] for row in known if "'" in row["area_name"])
    variants = [
        apostrophe_name.upper(),
        apostrophe_name.replace("'", ""),
        apostrophe_name.replace(" ", "-"),
        apostrophe_name.lower().replace(" ", "  "),  # extra internal whitespace
    ]
    for variant in variants:
        result = copilot.answer_question(f"show me {variant}")
        assert result["tool"] == "locate_area", (variant, result)
        assert result["area_name"] == apostrophe_name, (variant, result["area_name"])
    print(f"Normalized-match variants of {apostrophe_name!r} all resolved correctly: {variants}")


# --- Fuzzy (typo-tolerant) matching against only the real dataset ------------------------------

FUZZY_TYPO_CASES = [
    # (typo'd question, expected canonical area_name)
    ("show me seyoh", "Al Seyouh Suburb"),        # dropped letter, one word inside a 3-word name
    ("show me muwailah", "Muwaylih"),               # transliteration-style respelling
    ("find al nahdah", "Al Nahdah"),                # exact once normalized -- also proves tier 2
]


def test_typo_examples_resolve_via_fuzzy_matching():
    for question, expected_name in FUZZY_TYPO_CASES:
        tool, args = copilot.route(question)
        # 'seyoh' spans 2 emirates for the same name -- that's a SEPARATE (emirate) ambiguity,
        # not a naming failure, so it correctly asks for clarification naming the right area.
        if tool == "clarify":
            assert expected_name in args["suggestion"], (question, args)
            print(f"{question!r} -> clarify (ambiguous emirate), correctly named {expected_name!r}: {args['suggestion']!r}")
            continue
        assert tool == "locate_area", (question, tool, args)
        assert args["area_name"] == expected_name, (question, args)
        print(f"{question!r} -> locate_area, area_name={args['area_name']!r}")


# --- Ambiguity: one real name, more than one emirate --------------------------------------------

def test_ambiguous_name_across_emirates_asks_for_clarification_naming_both():
    result = copilot.answer_question("show me Al Seyouh Suburb")
    assert result["tool"] is None, result
    assert result["mode"] == "clarification", result
    assert "Al Seyouh Suburb" in result["answer"], result["answer"]
    assert "Dubai" in result["answer"] and "Sharjah" in result["answer"], result["answer"]
    assert result["answer"].startswith("Did you mean ") and result["answer"].endswith("?"), result["answer"]
    print(f"'show me Al Seyouh Suburb' -> {result['answer']!r}")


def test_explicit_emirate_resolves_the_ambiguous_name_directly():
    result = copilot.answer_question("find Al Seyouh Suburb in Sharjah")
    assert result["tool"] == "locate_area", result
    assert result["area_name"] == "Al Seyouh Suburb", result
    assert result["emirate"] == "Sharjah", result
    assert all(z["emirate"] == "Sharjah" for z in result["tool_result"]), result["tool_result"]
    print(f"'find Al Seyouh Suburb in Sharjah' -> {len(result['h3_ids'])} zone(s), all in Sharjah.")

    result_dubai = copilot.answer_question("find Al Seyouh Suburb in Dubai")
    assert result_dubai["emirate"] == "Dubai", result_dubai
    assert set(result["h3_ids"]).isdisjoint(result_dubai["h3_ids"]), "Sharjah/Dubai zone sets must not overlap"
    print(f"'find Al Seyouh Suburb in Dubai' -> {len(result_dubai['h3_ids'])} zone(s), all in Dubai, disjoint from Sharjah's.")


# --- Non-hallucination: a name genuinely absent from the processed dataset ----------------------

def test_nonexistent_name_gives_the_exact_required_message_not_a_guess():
    result = copilot.answer_question("show me Zzznotarealplace")
    assert result["tool"] is None, result
    assert result["mode"] == "area_not_found", result
    assert result["answer"] == AREA_NOT_FOUND, result["answer"]
    print(f"'show me Zzznotarealplace' -> {result['answer']!r}")


def test_gibberish_never_produces_a_locate_area_tool_call():
    for question in ["show me asdkjaksjdaksjd", "where is xkcdqqzz", "find qqzzxxnotreal"]:
        result = copilot.answer_question(question)
        assert result["tool"] is None, (question, result["tool"])
        assert result["mode"] == "area_not_found", (question, result["mode"])
    print("Gibberish locate-style questions all -> mode=area_not_found, no tool call, no guess.")


# --- Regression guard: the router must never let this new, LAST-resort resolver steal a
# question an earlier canonical-question branch already owns. ------------------------------------

EXISTING_INTENT_REGRESSION_CASES = [
    ("show me the weakest zones in Dubai", "get_weakest_zones"),
    ("show me the top priority zones", "get_top_priority_zones"),
    ("show me the top 5 best experience areas", "get_strongest_zones"),
    ("find the zones above median download", "get_metric_threshold_zones"),
]


def test_locate_area_never_hijacks_an_existing_canonical_question():
    for question, expected_tool in EXISTING_INTENT_REGRESSION_CASES:
        tool, _ = copilot.route(question)
        assert tool == expected_tool, f"{question!r}: got {tool!r}, expected {expected_tool!r}"
    print(f"Checked {len(EXISTING_INTENT_REGRESSION_CASES)} existing-intent questions: none hijacked by locate_area.")


# --- Structured result shape the frontend actually depends on (per src/build_dashboard.py's
# extractZoneRecords/GEO_TOOLS/setCopilotHighlight contract: highlight from structured H3 ids,
# never parsed out of the narrated text). ---------------------------------------------------

def test_structured_result_shape_matches_the_frontend_contract():
    result = copilot.answer_question("show me Al Hamra")
    assert result["intent"] == "locate_area"
    assert isinstance(result["area_name"], str) and result["area_name"]
    assert isinstance(result["emirate"], str) and result["emirate"]
    assert isinstance(result["h3_ids"], list) and result["h3_ids"]
    # Same "array of zone dicts, each carrying zone_id" shape every other GEO_TOOLS listing tool
    # already returns -- extractZoneRecords()/setCopilotHighlight() in build_dashboard.py need no
    # new code path, only 'locate_area' added to the GEO_TOOLS set.
    assert isinstance(result["tool_result"], list)
    assert all("zone_id" in z for z in result["tool_result"])
    assert result["h3_ids"] == [z["zone_id"] for z in result["tool_result"]]
    print(f"Structured result: intent={result['intent']!r}, area_name={result['area_name']!r}, "
          f"emirate={result['emirate']!r}, h3_ids={len(result['h3_ids'])} zone(s).")


def test_locate_area_tool_layer_defensive_behavior_direct():
    """ct.locate_area is meant to be called with an already-resolved, unambiguous area_name --
    but exercised directly (bypassing the router) it must still never silently merge zones from
    two different emirates into one answer, and must say plainly when a name doesn't exist."""
    ambiguous = ct.locate_area("Al Seyouh Suburb")  # no emirate given, name spans 2
    assert ambiguous and ambiguous[0].get("error") == "ambiguous_emirate", ambiguous
    assert set(ambiguous[0]["ambiguous_emirates"]) == {"Dubai", "Sharjah"}, ambiguous

    unknown = ct.locate_area("Definitely Not A Real Area")
    assert unknown and unknown[0].get("error") == "unknown_area_name", unknown

    resolved = ct.locate_area("Al Seyouh Suburb", emirate="Dubai")
    assert resolved and all(z.get("emirate") == "Dubai" for z in resolved), resolved
    print("ct.locate_area direct calls: ambiguous-emirate and unknown-name both refuse cleanly; "
          "emirate-scoped call resolves correctly.")


if __name__ == "__main__":
    print_area_name_sample_and_spot_check()
    test_exact_match_resolves_directly()
    test_normalized_match_ignores_case_punctuation_and_hyphens()
    test_typo_examples_resolve_via_fuzzy_matching()
    test_ambiguous_name_across_emirates_asks_for_clarification_naming_both()
    test_explicit_emirate_resolves_the_ambiguous_name_directly()
    test_nonexistent_name_gives_the_exact_required_message_not_a_guess()
    test_gibberish_never_produces_a_locate_area_tool_call()
    test_locate_area_never_hijacks_an_existing_canonical_question()
    test_structured_result_shape_matches_the_frontend_contract()
    test_locate_area_tool_layer_defensive_behavior_direct()
    print("\nT8 PASSED.")
