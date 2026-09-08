"""
T6 -- Copilot evaluation: runs the 72-question bank in tests/t6_questions.json through the real,
deployed Copilot path (question -> src/copilot.py::route() -> a src/copilot_tools.py function ->
narrate()) -- never a mock, never a hand-typed answer -- and scores the four dimensions the brief
requires: factual/numerical correctness, evidence grounding, hallucination, and appropriate
uncertainty. Includes the mandatory trap question ("Which e& site is causing this poor
experience?"), whose only correct behaviour is a grounded refusal.

Every "expected" value is recomputed HERE, independently, straight from
data/processed/zone_priority.parquet -- never a number typed into this file by hand -- so this
suite stays correct after the pipeline is re-run and never drifts from the data it is meant to
check. Zone-scoped questions pick a real zone_id at run time by role (the current #1 national
priority zone, a real 'low'-evidence-tier zone, a real insufficient-evidence zone) rather than a
hardcoded id, so the suite adapts to whichever zones the current pipeline output actually has in
each tier.

Run: python -m tests.test_t6_copilot_evaluation
(or `pytest tests/test_t6_copilot_evaluation.py -q` if pytest is installed)

Outputs (every run, win or lose):
  data/processed/t6_copilot_results.csv   -- one row per question: all 4 dimension scores + pass/fail
  data/processed/t6_copilot_results.json  -- same, machine-readable, plus the summary block
"""
import csv
import json
import re
from pathlib import Path

import pandas as pd

from src import copilot as c
from src import copilot_tools as ct

QUESTIONS_PATH = Path("tests/t6_questions.json")
ZONE_PRIORITY_PATH = Path("data/processed/zone_priority.parquet")
RESULTS_CSV_PATH = Path("data/processed/t6_copilot_results.csv")
RESULTS_JSON_PATH = Path("data/processed/t6_copilot_results.json")

_METRIC_COL = {"download_mbps": "download_mbps", "upload_mbps": "upload_mbps", "latency_ms": "latency_effective_ms"}
_METRIC_WORD = {"download_mbps": "download", "upload_mbps": "upload", "latency_ms": "latency"}


# ---------------------------------------------------------------------------
# Data access -- independent of src/copilot_tools.py's own filtering helpers (mirrors the
# "recompute from the raw table" pattern tests/test_t7_copilot_paraphrase.py already uses), so
# this suite isn't just checking the app's own cache of its own logic.
# ---------------------------------------------------------------------------

def load_questions() -> list[dict]:
    return json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))["questions"]


def load_df() -> pd.DataFrame:
    return pd.read_parquet(ZONE_PRIORITY_PATH)


def latest_quarter(df: pd.DataFrame) -> str:
    return sorted(df["quarter"].unique())[-1]


def classified(df: pd.DataFrame, quarter: str) -> pd.DataFrame:
    return df[(df["quarter"] == quarter) & (~df["insufficient_evidence"])]


def full_tier(df: pd.DataFrame, quarter: str) -> pd.DataFrame:
    return df[(df["quarter"] == quarter) & (df["evidence_tier"] == "full")]


def num2(v):
    return None if pd.isna(v) else round(float(v), 2)


def raw_row(df: pd.DataFrame, zone_id: str, quarter: str):
    match = df[(df["h3_cell"] == zone_id) & (df["quarter"] == quarter)]
    return None if match.empty else match.iloc[0]


# ---------------------------------------------------------------------------
# Zone-role resolution -- real zone_ids picked from the CURRENT data, never hardcoded.
# ---------------------------------------------------------------------------

def resolve_zone_roles(df: pd.DataFrame, quarter: str) -> dict:
    full = full_tier(df, quarter)
    top_priority = full.sort_values("priority_score", ascending=False).iloc[0]["h3_cell"]
    low_rows = df[(df["quarter"] == quarter) & (df["evidence_tier"] == "low")]
    insufficient_rows = df[(df["quarter"] == quarter) & (df["insufficient_evidence"])]
    roles = {"top_priority": top_priority}
    roles["low_tier"] = low_rows.iloc[0]["h3_cell"] if not low_rows.empty else None
    roles["insufficient_evidence"] = insufficient_rows.iloc[0]["h3_cell"] if not insufficient_rows.empty else None
    return roles


# ---------------------------------------------------------------------------
# Independent oracles for every listing intent in the question bank -- recomputed straight from
# the raw table with the exact same filtering rules documented in docs/data_dictionary.md /
# src/copilot_tools.py's own docstrings, but written here from scratch rather than calling
# src/copilot_tools.py's functions, so a bug shared between the tool and its own test would still
# be caught.
# ---------------------------------------------------------------------------

def expected_weakest(df, quarter, n):
    return classified(df, quarter).sort_values("experience_index", ascending=True).head(n)["h3_cell"].tolist()


def expected_priority(df, quarter, n):
    return full_tier(df, quarter).sort_values("priority_score", ascending=False).head(n)["h3_cell"].tolist()


def expected_deteriorating(df, quarter, n):
    flagged = classified(df, quarter)
    flagged = flagged[flagged["deteriorating"]]
    if n is not None:
        return flagged.sort_values("trend_pts_per_qtr", ascending=True).head(n)["h3_cell"].tolist()
    return set(flagged["h3_cell"].tolist())  # unranked when no N is requested -- order isn't meaningful


def expected_anomalous(df, quarter, kind):
    col = {"peer_gap": "peer_gap_ml_anomaly", "temporal": "temporal_anomaly_ml_flag"}[kind]
    sub = classified(df, quarter)
    return set(sub.loc[sub[col], "h3_cell"].tolist())  # ML flags aren't ranked by the tool either


def expected_high_pop_weak(df, quarter, n):
    sub = classified(df, quarter)
    if sub.empty:
        return []
    exp_median, pop_median = sub["experience_index"].median(), sub["population"].median()
    candidates = sub[(sub["experience_index"] <= exp_median) & (sub["population"] >= pop_median)]
    return candidates.sort_values("population", ascending=False).head(n)["h3_cell"].tolist()


def expected_metric_threshold(df, quarter, metric, comparison):
    sub = classified(df, quarter)
    col = _METRIC_COL[metric]
    median_value = round(float(sub[col].median()), 2)
    if comparison == "above":
        matched = sub[sub[col] > median_value].sort_values(col, ascending=False)
    else:
        matched = sub[sub[col] < median_value].sort_values(col, ascending=True)
    return matched["h3_cell"].tolist()


def expected_locate_area(df, area_name, emirate):
    real = df[df["area_name"] == area_name].drop_duplicates("h3_cell")
    if emirate:
        real = real[real["emirate"] == emirate]
    return sorted(real["h3_cell"].tolist())


# ---------------------------------------------------------------------------
# Shared dimension checks -- evidence grounding & hallucination, applied identically regardless
# of which tool answered. Deterministic string/number matching, never an LLM judging "does this
# sound right".
# ---------------------------------------------------------------------------

# Requires a digit on BOTH sides of a decimal point -- plain `-?\d+\.?\d*` also swallows a
# sentence-ending period right after a bare integer (e.g. "...in 2026Q2." -> a spurious "2."
# token from the quarter code's trailing digit + the sentence's own full stop), which is never
# really a second number in the text.
_NUMBER_RE = re.compile(r"-?\d+\.\d+|-?\d+")
_H3_ID_PATTERN = re.compile(r"\b8[0-9a-f]{14}\b", re.IGNORECASE)  # UAE H3 res-6 cells: 15 hex chars, start '8'
_OPERATOR_NAMES = ["e&", "du", "etisalat", "virgin mobile"]
# Template-narration literals that are never claims about THIS zone's data (the "/100" scale
# suffix, the "/8 quarters" denominator) -- structural constants, not numbers that need grounding.
_STRUCTURAL_NUMBERS = {"100", "8"}


def _numeric_strings(v) -> set[str]:
    variants = {str(v)}
    if isinstance(v, float):
        variants |= {f"{v:.1f}", f"{v:.2f}", f"{v:+.1f}", f"{abs(v):.1f}", f"{abs(v):.2f}"}
        if v == int(v):
            variants.add(str(int(v)))
    if isinstance(v, int):
        variants.add(f"{v:+d}")
    return variants


def _flatten_grounded_tokens(obj) -> set[str]:
    tokens: set[str] = set()

    def rec(o):
        if isinstance(o, dict):
            for v in o.values():
                rec(v)
        elif isinstance(o, list):
            for v in o:
                rec(v)
        elif isinstance(o, bool):
            pass
        elif isinstance(o, (int, float)):
            tokens.update(_numeric_strings(o))
        elif isinstance(o, str):
            tokens.update(_NUMBER_RE.findall(o))

    rec(obj)
    return tokens


def check_grounding(answer: str, tool_result, tool_args, extra_allowed: tuple = ()) -> tuple[bool, list[str]]:
    """Every number in `answer` must trace back to `tool_result` (or `tool_args`, or a small
    structural-constant/allowed set) -- the same "never calculate, estimate, or round a new
    number" rule src/copilot.py's own SYSTEM_PROMPT states for an LLM narrator, checked here
    mechanically against whichever narrator actually ran (template today; LLM if connected)."""
    if tool_result is None:
        return True, []  # no structured evidence to ground against -- refusal/clarify/unmatched
    grounded = _flatten_grounded_tokens(tool_result) | _flatten_grounded_tokens(tool_args) | _STRUCTURAL_NUMBERS
    grounded |= {str(x) for x in extra_allowed}
    found = _NUMBER_RE.findall(answer)
    ungrounded = [n for n in found if n not in grounded]
    return (len(ungrounded) == 0), ungrounded


def check_hallucination(answer: str, mode: str) -> tuple[bool, list[str]]:
    """No raw H3 cell id ever surfaced (SYSTEM_PROMPT rule 2), and no operator name asserted
    outside the two modes whose entire job is stating the data has none (refusal/fixed_fact --
    both draw from fixed constants that name operators only to disclaim them, never to attribute
    a result to one)."""
    reasons = []
    if _H3_ID_PATTERN.search(answer):
        reasons.append("raw H3 cell id present in narration")
    if mode not in ("refusal", "fixed_fact"):
        lower = answer.lower()
        for name in _OPERATOR_NAMES:
            if re.search(rf"\b{re.escape(name)}\b", lower):
                reasons.append(f"operator name {name!r} mentioned outside a refusal/data-source answer")
    return (len(reasons) == 0), reasons


_REFUSAL_MEANING_GROUPS = [
    ["public", "crowdsourced", "outside-in"],
    ["no operator attribution", "cannot determine", "does not have"],
]
_DATA_SOURCE_MEANING_GROUPS = [
    ["public", "crowdsourced"],
    ["no e&", "no.", "not e&", "no internal"],
]


def _meaning_present(answer: str, groups: list[list[str]]) -> bool:
    """True if the answer contains at least one phrase from EVERY group -- a meaning check, not
    an exact-sentence check (the brief explicitly asks not to require one exact wording)."""
    lower = answer.lower()
    return all(any(phrase in lower for phrase in group) for group in groups)


# ---------------------------------------------------------------------------
# Per-question evaluation -- dispatches on `kind`, computes the 4 dimension scores + an overall
# pass/fail, returns one flat result record.
# ---------------------------------------------------------------------------

def _listing_result(qdef, df, quarter, expected_ids, expected_tool, ordered: bool, extra_arg_checks=None):
    result = c.answer_question(qdef["question"])
    tool_result = result["tool_result"] or []
    actual_ids = [z["zone_id"] for z in tool_result]

    tool_ok = result["tool"] == expected_tool
    args_ok = True
    if extra_arg_checks:
        for key, expected_val in extra_arg_checks.items():
            args_ok = args_ok and (result["tool_args"] or {}).get(key) == expected_val
    if ordered:
        ids_ok = actual_ids == expected_ids
    else:
        ids_ok = set(actual_ids) == set(expected_ids)
    factual = tool_ok and args_ok and ids_ok

    grounding_ok, ungrounded = check_grounding(result["answer"], tool_result, result["tool_args"], extra_allowed=(len(tool_result),))
    halluc_ok, halluc_reasons = check_hallucination(result["answer"], result["mode"])
    uncertainty_ok = True  # a plain listing carries no uncertainty claim to get wrong

    reasons = []
    if not tool_ok:
        reasons.append(f"tool={result['tool']!r}, expected {expected_tool!r}")
    if not args_ok:
        reasons.append(f"tool_args={result['tool_args']!r} missing/mismatched {extra_arg_checks!r}")
    if not ids_ok:
        reasons.append(f"{len(set(actual_ids) ^ set(expected_ids))} zone(s) differ from independent recomputation")
    if not grounding_ok:
        reasons.append(f"ungrounded number(s) in answer: {ungrounded}")
    if not halluc_ok:
        reasons.append("; ".join(halluc_reasons))

    return _record(qdef, result, factual, grounding_ok, halluc_ok, uncertainty_ok, reasons)


def _record(qdef, result, factual, grounding, halluc, uncertainty, reasons) -> dict:
    overall = factual and grounding and halluc and uncertainty
    return {
        "id": qdef["id"], "category": qdef["category"], "kind": qdef["kind"], "question": qdef["question"],
        "tool": result["tool"], "mode": result["mode"], "answer": result["answer"],
        "factual_correctness": int(bool(factual)), "evidence_grounding": int(bool(grounding)),
        "no_hallucination": int(bool(halluc)), "appropriate_uncertainty": int(bool(uncertainty)),
        "overall_pass": int(bool(overall)), "reason": " | ".join(reasons),
    }


def evaluate_question(qdef, df, quarter, zone_roles) -> dict:
    kind = qdef["kind"]
    params = qdef.get("params", {}) or {}

    if kind == "listing_weakest":
        expected = expected_weakest(df, quarter, params["n"])
        return _listing_result(qdef, df, quarter, expected, "get_weakest_zones", ordered=True)

    if kind == "listing_priority":
        expected = expected_priority(df, quarter, params["n"])
        return _listing_result(qdef, df, quarter, expected, "get_top_priority_zones", ordered=True)

    if kind == "listing_deteriorating":
        expected = expected_deteriorating(df, quarter, params["n"])
        return _listing_result(qdef, df, quarter, expected, "get_deteriorating_zones", ordered=(params["n"] is not None))

    if kind == "listing_anomalous":
        expected = expected_anomalous(df, quarter, params["kind"])
        return _listing_result(qdef, df, quarter, expected, "get_anomalous_zones", ordered=False,
                                extra_arg_checks={"kind": params["kind"]})

    if kind == "listing_high_pop_weak":
        expected = expected_high_pop_weak(df, quarter, params["n"])
        return _listing_result(qdef, df, quarter, expected, "get_high_population_weak_zones", ordered=True)

    if kind == "listing_metric_threshold":
        expected = expected_metric_threshold(df, quarter, params["metric"], params["comparison"])
        return _listing_result(qdef, df, quarter, expected, "get_metric_threshold_zones", ordered=True,
                                extra_arg_checks={"metric": params["metric"], "comparison": params["comparison"]})

    if kind in ("zone_why_priority", "zone_confidence", "zone_evidence", "zone_measurements", "zone_raw_metrics"):
        zone_id = zone_roles[qdef["zone_role"]]
        result = c.answer_question(qdef["question"], zone_id=zone_id)
        row = raw_row(df, zone_id, quarter)
        tool_result = result["tool_result"] or {}
        tool_ok = result["tool"] == "get_zone_details"

        checks = {
            "experience_index": num2(row["experience_index"]), "confidence_score": num2(row["confidence_score"]),
            "priority_score": num2(row["priority_score"]), "tests": int(row["tests"]), "devices": int(row["devices"]),
            "quarters_observed": int(row["quarters_observed"]),
            "download_mbps": num2(row["download_mbps"]), "upload_mbps": num2(row["upload_mbps"]),
            "latency_ms": num2(row["latency_effective_ms"]),
        }
        mismatches = [f"{k}: got {tool_result.get(k)!r}, expected {v!r}" for k, v in checks.items()
                      if k in tool_result and tool_result.get(k) != v]
        factual = tool_ok and not mismatches

        uncertainty_ok = True
        if kind == "zone_evidence":
            insufficient = bool(row["insufficient_evidence"])
            says_insufficient = bool(re.search(r"not enough|insufficient", result["answer"], re.IGNORECASE))
            uncertainty_ok = says_insufficient if insufficient else True
            factual = factual and (tool_result.get("evidence_status") == "insufficient_evidence") == insufficient

        grounding_ok, ungrounded = check_grounding(result["answer"], tool_result, result["tool_args"])
        halluc_ok, halluc_reasons = check_hallucination(result["answer"], result["mode"])

        reasons = []
        if not tool_ok:
            reasons.append(f"tool={result['tool']!r}, expected get_zone_details")
        if mismatches:
            reasons.append("; ".join(mismatches))
        if not uncertainty_ok:
            reasons.append("evidence-insufficiency not communicated in narration")
        if not grounding_ok:
            reasons.append(f"ungrounded number(s): {ungrounded}")
        if not halluc_ok:
            reasons.append("; ".join(halluc_reasons))
        return _record(qdef, result, factual, grounding_ok, halluc_ok, uncertainty_ok, reasons)

    if kind == "zone_trend":
        zone_id = zone_roles[qdef["zone_role"]]
        result = c.answer_question(qdef["question"], zone_id=zone_id)
        row = raw_row(df, zone_id, quarter)
        tool_result = result["tool_result"] or {}
        tool_ok = result["tool"] == "get_zone_trend"
        checks_ok = (tool_result.get("trend_pts_per_qtr") == num2(row["trend_pts_per_qtr"])
                     and tool_result.get("quarters_observed") == int(row["quarters_observed"])
                     and tool_result.get("currently_deteriorating") == bool(row["deteriorating"]))
        factual = tool_ok and checks_ok
        grounding_ok, ungrounded = check_grounding(result["answer"], tool_result, result["tool_args"])
        halluc_ok, halluc_reasons = check_hallucination(result["answer"], result["mode"])
        reasons = []
        if not factual:
            reasons.append(f"trend fields mismatch: tool={result['tool']!r}, tool_result={tool_result}")
        if not grounding_ok:
            reasons.append(f"ungrounded number(s): {ungrounded}")
        if not halluc_ok:
            reasons.append("; ".join(halluc_reasons))
        return _record(qdef, result, factual, grounding_ok, halluc_ok, True, reasons)

    if kind == "zone_trend_metric":
        zone_id = zone_roles[qdef["zone_role"]]
        metric = params["metric"]
        result = c.answer_question(qdef["question"], zone_id=zone_id)
        tool_result = result["tool_result"] or {}
        tool_ok = result["tool"] == "get_zone_trend"

        scored = [h for h in (tool_result.get("history") or []) if h.get("evidence_status") == "scored"]
        col = _METRIC_COL[metric]
        history_col = {"download_mbps": "download_mbps", "upload_mbps": "upload_mbps", "latency_ms": "latency_ms"}[metric]
        expected_change = None
        rows = df[df["h3_cell"] == zone_id].sort_values("quarter")
        scored_rows = rows[~rows["insufficient_evidence"]]
        if len(scored_rows) >= 2:
            prev_val, cur_val = scored_rows.iloc[-2][col], scored_rows.iloc[-1][col]
            if pd.notna(prev_val) and pd.notna(cur_val):
                expected_change = round(float(cur_val) - float(prev_val), 2)
        change_key = f"{metric}_change_qoq"
        factual = tool_ok and tool_result.get(change_key) == expected_change

        # Narration must name the right metric and the right improve/worsen direction (latency:
        # lower is better, so a negative change is an "improvement" -- the same valence flip
        # applied in src/copilot.py's own narration).
        metric_word = _METRIC_WORD[metric]
        direction_ok = True
        if expected_change is not None and expected_change != 0:
            improved = (expected_change < 0) if metric == "latency_ms" else (expected_change > 0)
            expected_word = "improved" if improved else "worsened"
            direction_ok = metric_word in result["answer"].lower() and expected_word in result["answer"].lower()
        factual = factual and direction_ok

        grounding_ok, ungrounded = check_grounding(result["answer"], tool_result, result["tool_args"])
        halluc_ok, halluc_reasons = check_hallucination(result["answer"], result["mode"])
        reasons = []
        if not factual:
            reasons.append(f"expected {change_key}={expected_change}, got {tool_result.get(change_key)!r} "
                            f"(direction_ok={direction_ok})")
        if not grounding_ok:
            reasons.append(f"ungrounded number(s): {ungrounded}")
        if not halluc_ok:
            reasons.append("; ".join(halluc_reasons))
        return _record(qdef, result, factual, grounding_ok, halluc_ok, True, reasons)

    if kind == "locate_area":
        result = c.answer_question(qdef["question"])
        expected_name, expected_emirate = params["expect_area_name"], params["expect_emirate"]
        expected_ids = expected_locate_area(df, expected_name, expected_emirate)
        factual = (result["tool"] == "locate_area" and result.get("area_name") == expected_name
                   and result.get("emirate") == expected_emirate
                   and sorted(result.get("h3_ids") or []) == expected_ids)
        grounding_ok, ungrounded = check_grounding(result["answer"], result["tool_result"], result["tool_args"],
                                                     extra_allowed=(len(result.get("h3_ids") or []),))
        halluc_ok, halluc_reasons = check_hallucination(result["answer"], result["mode"])
        reasons = []
        if not factual:
            reasons.append(f"got area_name={result.get('area_name')!r} emirate={result.get('emirate')!r} "
                            f"mode={result['mode']!r}, expected {expected_name!r} in {expected_emirate!r}")
        if not grounding_ok:
            reasons.append(f"ungrounded number(s): {ungrounded}")
        if not halluc_ok:
            reasons.append("; ".join(halluc_reasons))
        return _record(qdef, result, factual, grounding_ok, halluc_ok, True, reasons)

    if kind == "locate_area_ambiguous":
        result = c.answer_question(qdef["question"])
        name, emirates = params["expect_area_name"], params["expect_emirates"]
        factual = (result["tool"] is None and result["mode"] == "clarification" and name in result["answer"]
                   and all(e in result["answer"] for e in emirates))
        halluc_ok, halluc_reasons = check_hallucination(result["answer"], result["mode"])
        reasons = [] if factual else [f"got mode={result['mode']!r} answer={result['answer']!r}"]
        if not halluc_ok:
            reasons.append("; ".join(halluc_reasons))
        return _record(qdef, result, factual, True, halluc_ok, True, reasons)

    if kind == "refusal_trap":
        zone_id = zone_roles.get(qdef["zone_role"]) if qdef.get("zone_role") else None
        result = c.answer_question(qdef["question"], zone_id=zone_id)
        meaning_ok = _meaning_present(result["answer"], _REFUSAL_MEANING_GROUPS)
        factual = result["tool"] is None and result["mode"] == "refusal" and meaning_ok
        halluc_ok, halluc_reasons = check_hallucination(result["answer"], result["mode"])
        reasons = []
        if not factual:
            reasons.append(f"got tool={result['tool']!r} mode={result['mode']!r} meaning_ok={meaning_ok} "
                            f"answer={result['answer']!r}")
        if not halluc_ok:
            reasons.append("; ".join(halluc_reasons))
        return _record(qdef, result, factual, True, halluc_ok, factual, reasons)

    if kind == "data_source_fact":
        result = c.answer_question(qdef["question"])
        meaning_ok = _meaning_present(result["answer"], _DATA_SOURCE_MEANING_GROUPS)
        factual = result["tool"] is None and result["mode"] == "fixed_fact" and meaning_ok
        halluc_ok, halluc_reasons = check_hallucination(result["answer"], result["mode"])
        reasons = [] if factual else [f"got mode={result['mode']!r} meaning_ok={meaning_ok} answer={result['answer']!r}"]
        if not halluc_ok:
            reasons.append("; ".join(halluc_reasons))
        return _record(qdef, result, factual, True, halluc_ok, factual, reasons)

    if kind == "unsupported_no_hallucination":
        result = c.answer_question(qdef["question"], zone_id=zone_roles.get("top_priority"))
        factual = result["tool"] is None  # never a confidently-invented answer to an out-of-scope question
        halluc_ok, halluc_reasons = check_hallucination(result["answer"], result["mode"])
        reasons = [] if factual else [f"tool={result['tool']!r} -- an out-of-scope question must never select a tool"]
        if not halluc_ok:
            reasons.append("; ".join(halluc_reasons))
        return _record(qdef, result, factual, True, halluc_ok, factual, reasons)

    raise ValueError(f"unknown question kind: {kind!r} (id={qdef['id']})")


# ---------------------------------------------------------------------------
# Suite runner, report, and outputs
# ---------------------------------------------------------------------------

def run_suite() -> list[dict]:
    df = load_df()
    quarter = latest_quarter(df)
    zone_roles = resolve_zone_roles(df, quarter)
    questions = load_questions()
    return [evaluate_question(q, df, quarter, zone_roles) for q in questions]


def write_outputs(rows: list[dict]) -> None:
    RESULTS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["id", "category", "kind", "question", "tool", "mode", "factual_correctness",
                  "evidence_grounding", "no_hallucination", "appropriate_uncertainty", "overall_pass",
                  "reason", "answer"]
    with open(RESULTS_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    RESULTS_JSON_PATH.write_text(
        json.dumps({"summary": summary, "results": rows}, indent=2, ensure_ascii=False), encoding="utf-8")


def summarize(rows: list[dict]) -> dict:
    total = len(rows)
    passed = sum(r["overall_pass"] for r in rows)
    by_category = {}
    for r in rows:
        cat = by_category.setdefault(r["category"], {"total": 0, "passed": 0})
        cat["total"] += 1
        cat["passed"] += r["overall_pass"]
    dims = {}
    for dim in ("factual_correctness", "evidence_grounding", "no_hallucination", "appropriate_uncertainty"):
        dims[dim] = {"passed": sum(r[dim] for r in rows), "total": total}
    return {
        "total_questions": total, "passed": passed, "failed": total - passed,
        "pass_rate": round(passed / total, 4) if total else None,
        "by_category": by_category, "by_dimension": dims,
    }


def print_report(rows: list[dict]) -> None:
    s = summarize(rows)
    print("=" * 100)
    print(f"T6 COPILOT EVALUATION -- {s['total_questions']} questions")
    print("=" * 100)
    print(f"Passed: {s['passed']}/{s['total_questions']}  ({s['pass_rate']:.1%})")
    print()
    print("By category:")
    for cat, v in sorted(s["by_category"].items()):
        print(f"  {cat:22} {v['passed']:3}/{v['total']:3}")
    print()
    print("By dimension:")
    for dim, v in s["by_dimension"].items():
        print(f"  {dim:24} {v['passed']:3}/{v['total']:3}")
    failures = [r for r in rows if not r["overall_pass"]]
    print()
    if failures:
        print(f"FAILURES ({len(failures)}):")
        for r in failures:
            failed_dims = [d for d in ("factual_correctness", "evidence_grounding", "no_hallucination",
                                        "appropriate_uncertainty") if not r[d]]
            print(f"  [{r['id']}] {r['question']!r}")
            print(f"       failed dimensions: {failed_dims}")
            print(f"       reason: {r['reason']}")
    else:
        print("No failures.")
    print("=" * 100)
    print(f"Results written to {RESULTS_CSV_PATH} and {RESULTS_JSON_PATH}")


# ---------------------------------------------------------------------------
# pytest-discoverable entry points
# ---------------------------------------------------------------------------

def test_t6_full_suite():
    rows = run_suite()
    write_outputs(rows)
    print_report(rows)
    s = summarize(rows)
    failures = [r for r in rows if not r["overall_pass"]]
    assert not failures, (
        f"{len(failures)}/{s['total_questions']} T6 question(s) failed -- see "
        f"{RESULTS_CSV_PATH} for details:\n" + "\n".join(f"  [{r['id']}] {r['question']!r}: {r['reason']}" for r in failures)
    )


def test_mandatory_operator_trap_refuses():
    """The one hard requirement the brief calls out by name: 'Which e& site is causing this poor
    experience?' MUST be a grounded refusal -- no tool call, no operator-specific answer, no
    invented site/tower/root cause. Enforced on its own, separate from the aggregate pass rate,
    so this specific requirement can never be silently averaged away by other passing questions."""
    df = load_df()
    quarter = latest_quarter(df)
    zone_roles = resolve_zone_roles(df, quarter)
    questions = {q["id"]: q for q in load_questions()}
    trap = questions["L1"]
    assert trap["question"] == "Which e& site is causing this poor experience?", trap
    record = evaluate_question(trap, df, quarter, zone_roles)
    assert record["tool"] is None, f"trap question selected a tool: {record['tool']!r} -- must be None"
    assert record["mode"] == "refusal", f"trap question mode={record['mode']!r} -- must be 'refusal'"
    assert record["overall_pass"], f"trap question failed: {record['reason']}"
    print(f"Mandatory trap question -> mode=refusal, no tool call. Answer: {record['answer']!r}")


def test_area_name_resolution_regression():
    """The specific reported bug: 'show me Al Mafraq, Abu Dhabi' must resolve to Al Mafraq, never
    collapse to the emirate name 'Abu Dhabi' -- checked across every query form the brief asked
    for (comma, omitted comma, case, 'what is happening in ...')."""
    df = load_df()
    quarter = latest_quarter(df)
    zone_roles = resolve_zone_roles(df, quarter)
    questions = {q["id"]: q for q in load_questions()}
    failures = []
    for qid in ("J1", "J2", "J3", "J4", "J5"):
        record = evaluate_question(questions[qid], df, quarter, zone_roles)
        if not record["overall_pass"]:
            failures.append(f"[{qid}] {questions[qid]['question']!r}: {record['reason']}")
        print(f"[{qid}] {questions[qid]['question']!r} -> pass={bool(record['overall_pass'])}")
    assert not failures, "\n" + "\n".join(failures)


if __name__ == "__main__":
    _rows = run_suite()
    write_outputs(_rows)
    print_report(_rows)
    _failed = [r for r in _rows if not r["overall_pass"]]
    if _failed:
        raise SystemExit(f"T6 FAILED: {len(_failed)}/{len(_rows)} question(s) failed.")
    print("\nT6 PASSED.")
