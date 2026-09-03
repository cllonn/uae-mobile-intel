"""
The grounded copilot: question -> deterministic tool -> structured evidence -> narration.

    User question -> [refusal check] -> choose tool -> execute tool -> LLM explains the
    structured result -> grounded answer

The LLM (when connected -- see `_llm_narrate`) is only ever shown the JSON a `copilot_tools`
function already returned. It is never asked to compute a score, and the system prompt says
so explicitly. If no LLM is connected, `_template_narrate` produces the same structure from
plain string formatting -- the tool layer and routing are fully testable without any API key.

No RAG, no embeddings, no vector store, no agent framework: the "tool selection" step below
is plain keyword routing over the ten canonical questions plus the two zone-scoped filters
this challenge asks for (Dubai/Sharjah-style emirate mentions). That is deliberately as far
as this goes today -- free-form question answering is an explicit stretch goal in the brief,
not a requirement.
"""
import os
import re

from src import copilot_tools as ct

EMIRATES = ["Abu Dhabi", "Dubai", "Sharjah", "Ajman", "Umm Al Quwain", "Ras Al Khaimah", "Fujairah"]

# ---------------------------------------------------------------------------
# Hard grounding / refusal -- deterministic, checked BEFORE any tool routing.
# Seeded from the T6 trap-question bank's phrasings (operator/site attribution, root-cause
# diagnosis, indoor coverage, customer counts) -- this is a keyword safeguard, not a model
# judgment call, so it can't be talked out of refusing by clever phrasing of the same intent.
# ---------------------------------------------------------------------------
OPERATOR_ATTRIBUTION_PATTERNS = [
    r"\be&?\s*site\b", r"\be&?\s*tower\b", r"\be&?\s*cell\b", r"\bwhich (operator|carrier)\b",
    r"\bfault(y)?\b.*(tower|site|cell)", r"\broot cause\b", r"\bindoor coverage\b",
    r"\bcustomer count\b", r"\bnumber of customers\b", r"\bnetwork element\b",
    r"\be&\s*network performance\b", r"\bdiagnose\b.*(network|e&)",
]
# NOTE: "is this e& (network) data" is deliberately NOT in this list -- that question gets the
# factual DATA_SOURCE_MESSAGE ("No, it's public data") below, not the operator-attribution
# refusal. Both end up saying this isn't e& data, but they're different questions: one asks
# what the data *is*, the other asks the system to attribute a cause it structurally cannot.
REFUSAL_MESSAGE = (
    "This system uses public, outside-in mobile experience measurements (crowdsourced "
    "Speedtest data) with no operator attribution. It cannot determine which e&, or any "
    "other operator's, site, tower, or internal network element caused an observed result, "
    "and it does not have customer, root-cause, or indoor-coverage data of any kind."
)

DATA_SOURCE_PATTERNS = [r"\bis this e&", r"\be&\s*(internal\s+)?network data\b", r"\bwhose data is this\b"]
DATA_SOURCE_MESSAGE = (
    "No. This system is built entirely from public, crowdsourced data: Ookla Speedtest Open "
    "Data, WorldPop population estimates, and OpenStreetMap. It contains no e& internal "
    "network data, counters, alarms, or customer records at any stage."
)


def check_refusal(question: str) -> str | None:
    q = question.lower()
    for pattern in OPERATOR_ATTRIBUTION_PATTERNS:
        if re.search(pattern, q):
            return REFUSAL_MESSAGE
    return None


def check_data_source_question(question: str) -> str | None:
    q = question.lower()
    for pattern in DATA_SOURCE_PATTERNS:
        if re.search(pattern, q):
            return DATA_SOURCE_MESSAGE
    return None


# ---------------------------------------------------------------------------
# Tool routing -- plain keyword matching over the canonical questions.
# ---------------------------------------------------------------------------

_NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "ten": 10}


def _extract_n(question: str, default: int) -> int:
    q = question.lower()
    m = re.search(r"\b(\d+)\b", q)
    if m:
        return int(m.group(1))
    for word, n in _NUMBER_WORDS.items():
        if re.search(rf"\b{word}\b", q):
            return n
    return default


def _extract_emirate(question: str) -> str | None:
    for emirate in EMIRATES:
        if emirate.lower() in question.lower():
            return emirate
    return None


def route(question: str, zone_id: str | None = None) -> tuple[str, dict]:
    """Returns (tool_name, tool_args) for the first canonical pattern the question matches.
    Zone-scoped questions (why/confidence/trend/measurements) require a `zone_id` to already
    be in context -- exactly like the dashboard's drill-down panel, where a question is asked
    about the currently-selected zone, not typed from nothing."""
    q = question.lower()
    emirate = _extract_emirate(question)

    if zone_id and re.search(r"\bwhy\b.*(priorit|invest)", q):
        return "get_zone_details", {"zone_id": zone_id}
    if zone_id and re.search(r"\bconfiden", q):
        return "get_zone_details", {"zone_id": zone_id}
    if zone_id and re.search(r"\bchanged?\b.*(quarter|since)", q):
        return "get_zone_trend", {"zone_id": zone_id}
    if zone_id and re.search(r"\bmeasurements?\b.*(support|behind|recommend)", q):
        return "get_zone_details", {"zone_id": zone_id}
    if zone_id and re.search(r"\bpeer\b", q):
        return "get_zone_peer_comparison", {"zone_id": zone_id}

    if re.search(r"\b(deteriorat\w*|declin\w*|worse over time)\b", q):
        return "get_deteriorating_zones", {"emirate": emirate}
    if re.search(r"\banomal|unusual\b", q):
        kind = "temporal" if re.search(r"\bhistory|own past|over time\b", q) else "peer_gap"
        return "get_anomalous_zones", {"kind": kind, "emirate": emirate}
    if re.search(r"\bpopulation\b", q) and re.search(r"\bweak\b", q):
        return "get_high_population_weak_zones", {"emirate": emirate}
    if re.search(r"\b(priorit\w*|investigate\w*)\b", q):
        return "get_top_priority_zones", {"n": _extract_n(q, 5), "emirate": emirate}
    if re.search(r"\bweak(est)?\b", q):
        return "get_weakest_zones", {"n": _extract_n(q, 10), "emirate": emirate}
    if re.search(r"\bcoverage|represent|how much of\b", q):
        return "get_coverage_summary", {"emirate": emirate}

    return "unmatched", {}


_TOOLS = {
    "get_zone_details": ct.get_zone_details,
    "get_zone_peer_comparison": ct.get_zone_peer_comparison,
    "get_zone_trend": ct.get_zone_trend,
    "get_top_priority_zones": ct.get_top_priority_zones,
    "get_priority_zones": ct.get_priority_zones,
    "get_weakest_zones": ct.get_weakest_zones,
    "get_deteriorating_zones": ct.get_deteriorating_zones,
    "get_anomalous_zones": ct.get_anomalous_zones,
    "get_high_population_weak_zones": ct.get_high_population_weak_zones,
    "get_coverage_summary": ct.get_coverage_summary,
    "get_methodology": ct.get_methodology,
}
# get_priority_zones is deliberately NOT reachable from route() -- "which N areas first" (a
# canonical question, caller-chosen N) and "every zone actually flagged" (a fixed, data-
# determined set, used only by generate_all_priority_zone_briefs below) are different
# questions. It's registered in _TOOLS so answer_question() can still dispatch to it directly
# if a caller passes the tool name explicitly.


# ---------------------------------------------------------------------------
# Narration -- LLM if available, deterministic template otherwise. Same contract either way:
# explain the structured result, invent nothing.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You explain UAE mobile network experience data to a decision-maker. You will be given "
    "a user question and a JSON object that was already computed by a deterministic pipeline "
    "(Experience Index, Confidence Score, Peer Gap, trend, ML anomaly flags, Priority Score). "
    "Explain the JSON in plain, concise language. "
    "Rules, no exceptions: "
    "1) Use only numbers that appear in the JSON -- never calculate, estimate, or round a new "
    "number of your own. "
    "2) If a field is missing or null, say the data is not available -- never guess. "
    "3) This is public, outside-in mobile measurement data with no operator attribution -- "
    "never attribute a result to e& or any specific operator, site, or tower. "
    "4) Keep the answer under 120 words."
)


def _llm_available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def _llm_narrate(question: str, tool_name: str, tool_result) -> str:
    import anthropic  # imported lazily so the module loads fine without the package installed

    client = anthropic.Anthropic()
    message = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=300,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f"Question: {question}\n\nTool called: {tool_name}\nTool result (JSON):\n{tool_result}",
        }],
    )
    return message.content[0].text


def _template_narrate(question: str, tool_name: str, tool_result) -> str:
    """Deterministic fallback narrator -- no LLM call, plain string formatting over the same
    JSON an LLM would receive. Exists so the full pipeline is demoable and testable with zero
    external dependencies; swap in `_llm_narrate` the moment a key is available."""
    if tool_name in ("get_top_priority_zones", "get_priority_zones", "get_weakest_zones",
                     "get_high_population_weak_zones", "get_deteriorating_zones", "get_anomalous_zones"):
        if not tool_result:
            return "No zones matched this query at the requested evidence threshold."
        lines = [
            f"{i+1}. Zone {z['zone_id'][-6:]} ({z['emirate']}, {z['peer_group']}) -- "
            f"Experience {z.get('experience_index')}, Confidence {z.get('confidence_score')}, "
            + (f"Priority {z['priority_score']}" if z.get("evidence_tier") == "full"
               else "not shortlist-eligible (10-29 tests, low-confidence evidence)")
            for i, z in enumerate(tool_result[:10])
        ]
        return f"Found {len(tool_result)} matching zone(s):\n" + "\n".join(lines)

    if tool_name == "get_zone_details":
        if tool_result.get("evidence_status") == "insufficient_evidence":
            return (f"Zone {tool_result['zone_id'][-6:]}: {tool_result['message']} "
                    f"({tool_result['tests']} tests, {tool_result['devices']} devices this quarter.)")
        if tool_result.get("error"):
            return f"No data found: {tool_result['error']}."

        base = (
            f"Zone {tool_result['zone_id'][-6:]} ({tool_result['emirate']}, {tool_result['peer_group']}), "
            f"{tool_result['quarter']}: Experience Index {tool_result['experience_index']} vs. peer median "
            f"{tool_result['peer_group_median_experience']} ({tool_result['peer_gap']:+.1f} pts). "
            f"Confidence {tool_result['confidence_score']}/100 from {tool_result['tests']} tests / "
            f"{tool_result['devices']} devices across {tool_result['quarters_observed']}/8 quarters."
        )
        # 'low' evidence tier (10-29 tests): a real Experience Index exists above, but
        # src/priority.py never computes a Priority Score for this tier at all -- no Priority
        # Score line to show, and no factor breakdown to invent one from.
        if tool_result.get("evidence_tier") != "full":
            return (
                f"{base} This zone has low-confidence evidence (10-29 tests) -- shown, but "
                f"not eligible for the Priority shortlist; it is informative, not a safe "
                f"recommendation. This reflects public measurement patterns only; it cannot "
                f"identify an operator-specific cause."
            )
        high_factors = [k for k, v in tool_result["priority_factors"].items() if v == "High"]
        return (
            f"{base} Priority Score {tool_result['priority_score']}"
            f"{' (flagged for investigation)' if tool_result['priority_zone'] else ''}, driven mainly by: "
            + (", ".join(high_factors) if high_factors else "no single dominant factor")
            + ". This reflects public measurement patterns only; it cannot identify an operator-specific cause."
        )

    if tool_name == "get_zone_peer_comparison":
        return (
            f"Zone {tool_result['zone_id'][-6:]} vs. its {tool_result['peer_group_size']} "
            f"{tool_result['peer_group']} peers ({tool_result['quarter']}): Experience Index "
            f"{tool_result['experience_index']} vs. peer median {tool_result['peer_group_median_experience']}, "
            f"a gap of {tool_result['peer_gap']:+.1f} points ({tool_result['peer_gap_pct']:+.1f}%)."
        )

    if tool_name == "get_zone_trend":
        cur = "currently deteriorating" if tool_result.get("currently_deteriorating") else "not currently flagged as deteriorating"
        return (
            f"Zone {tool_result['zone_id'][-6:]}: {cur}, trend {tool_result['trend_pts_per_qtr']} "
            f"Experience-Index points/quarter over its {tool_result['quarters_observed']} observed quarters. "
            f"({'Has' if tool_result['ever_deteriorated_in_window'] else 'Has not'} hit the 3-quarter "
            f"decline pattern at some point in the 8-quarter window.)"
        )

    if tool_name == "get_coverage_summary":
        return (
            f"{tool_result['emirate']}, {tool_result['quarter']}: {tool_result['classified_zones']} of "
            f"{tool_result['measured_zones']} measured zones meet the evidence threshold, representing "
            f"{tool_result['population_represented_pct']}% of the population "
            f"({tool_result['population_represented']:,} of {tool_result['population_universe']:,}). "
            f"{tool_result['priority_zones']} zone(s) flagged for investigation."
        )

    if tool_name == "get_methodology":
        return f"{tool_result.get('metric')}: {tool_result.get('formula', tool_result)}"

    return "I could not map this question to a supported tool. Try one of the ten canonical questions."


def narrate(question: str, tool_name: str, tool_result) -> tuple[str, str]:
    """Returns (answer_text, mode) where mode is 'llm' or 'template'."""
    if _llm_available():
        try:
            return _llm_narrate(question, tool_name, tool_result), "llm"
        except Exception as exc:  # network/key/package issues -- fall back rather than crash the demo
            return _template_narrate(question, tool_name, tool_result) + f"\n[LLM call failed, used fallback: {exc}]", "template"
    return _template_narrate(question, tool_name, tool_result), "template"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def answer_question(question: str, zone_id: str | None = None, quarter: str | None = None) -> dict:
    refusal = check_refusal(question)
    if refusal:
        return {"question": question, "tool": None, "tool_args": None, "tool_result": None,
                "answer": refusal, "mode": "refusal"}

    data_source_answer = check_data_source_question(question)
    if data_source_answer:
        return {"question": question, "tool": None, "tool_args": None, "tool_result": None,
                "answer": data_source_answer, "mode": "fixed_fact"}

    tool_name, tool_args = route(question, zone_id)
    if tool_name == "unmatched":
        return {"question": question, "tool": None, "tool_args": None, "tool_result": None,
                "answer": "I could not map this question to a supported tool. Try one of the "
                          "ten canonical questions.", "mode": "unmatched"}

    if quarter and "quarter" in _TOOLS[tool_name].__code__.co_varnames:
        tool_args = {**tool_args, "quarter": quarter}
    tool_args = {k: v for k, v in tool_args.items() if v is not None}

    tool_result = _TOOLS[tool_name](**tool_args)
    answer, mode = narrate(question, tool_name, tool_result)
    return {"question": question, "tool": tool_name, "tool_args": tool_args,
            "tool_result": tool_result, "answer": answer, "mode": mode}


# ---------------------------------------------------------------------------
# Zone brief (capability #7's other mandatory piece)
# ---------------------------------------------------------------------------

def generate_zone_context(zone_id: str, quarter: str | None = None) -> dict:
    """Collects every verified fact an AI zone brief is allowed to state -- composed entirely
    from the tool layer above, nothing recalculated here."""
    details = ct.get_zone_details(zone_id, quarter)
    if details.get("evidence_status") != "scored":
        return details
    peer = ct.get_zone_peer_comparison(zone_id, details["quarter"])
    trend = ct.get_zone_trend(zone_id)
    return {**details, "peer_group_size": peer["peer_group_size"], "trend_history": trend["history"]}


BRIEF_SYSTEM_PROMPT = (
    "Write a 3-4 sentence zone brief for a decision-maker, using ONLY the numbers in the JSON "
    "provided. Cover: how this zone compares to its peer group, its confidence level, its "
    "population exposure, why it received its Priority Score (name the High-banded factors), "
    "and what to investigate next. End with one sentence noting this is public data and cannot "
    "identify an operator-specific root cause. Never invent a number not present in the JSON."
)


def generate_zone_brief(zone_id: str, quarter: str | None = None) -> dict:
    context = generate_zone_context(zone_id, quarter)
    if context.get("evidence_status") != "scored":
        return {"zone_id": zone_id, "brief": context.get("message", "No data available."), "mode": "insufficient_evidence"}

    if _llm_available():
        try:
            import anthropic
            client = anthropic.Anthropic()
            message = client.messages.create(
                model="claude-sonnet-5", max_tokens=300, system=BRIEF_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": str(context)}],
            )
            return {"zone_id": zone_id, "brief": message.content[0].text, "mode": "llm", "evidence": context}
        except Exception as exc:
            pass  # fall through to template

    peer_summary = (
        f"a public mobile experience of {context['experience_index']}/100, "
        f"{abs(context['peer_gap']):.1f} points {'below' if context['peer_gap'] < 0 else 'above'} "
        f"its {context['peer_group']} peer median of {context['peer_group_median_experience']} "
        f"({context['peer_group_size']} peers)."
    )
    confidence_summary = (
        f"Confidence is {context['confidence_score']}/100, based on {context['tests']} tests from "
        f"{context['devices']} devices across {context['quarters_observed']}/8 quarters."
    )
    # 'low' evidence tier: no Priority Score exists for this zone at all -- generate_zone_brief
    # can be called on any zone_id, not only ones that came from get_priority_zones, so this
    # path is real, not hypothetical. Never invent a priority narrative for it.
    if context.get("evidence_tier") != "full":
        brief = (
            f"Zone {zone_id[-6:]} ({context['emirate']}) shows {peer_summary} {confidence_summary} "
            f"Estimated population exposure is {context['population']:,}. This zone has "
            f"low-confidence evidence (10-29 tests) and was never eligible for the Priority "
            f"shortlist -- no Priority Score exists for it; treat this as informative context, "
            f"not an investigation recommendation. This is public, outside-in measurement data; "
            f"it cannot identify an operator-specific root cause."
        )
        return {"zone_id": zone_id, "brief": brief, "mode": "template", "evidence": context}

    high_factors = [k for k, v in context["priority_factors"].items() if v == "High"]
    if len(high_factors) > 1:
        factor_text = ", ".join(high_factors[:-1]) + f" and {high_factors[-1]}"
    else:
        factor_text = high_factors[0] if high_factors else "a combination of moderate factors"
    brief = (
        f"Zone {zone_id[-6:]} ({context['emirate']}) shows {peer_summary} {confidence_summary} "
        f"Estimated population exposure is {context['population']:,}. "
        f"It received a Priority Score of {context['priority_score']}, driven mainly by {factor_text}. "
        f"This is public, outside-in measurement data; it cannot identify an operator-specific root cause."
    )
    return {"zone_id": zone_id, "brief": brief, "mode": "template", "evidence": context}


def generate_all_priority_zone_briefs(quarter: str | None = None, emirate: str | None = None) -> list[dict]:
    """The mandatory "AI zone brief for every Priority zone" capability -- iterates
    `ct.get_priority_zones()` (the real, data-determined shortlist, not a caller-chosen top N)
    and calls `generate_zone_brief` once per zone. No new grounding logic here: correctness is
    entirely inherited from `get_priority_zones` (evidence gating) and `generate_zone_brief`
    (narration contract) above."""
    zones = ct.get_priority_zones(quarter=quarter, emirate=emirate)
    return [generate_zone_brief(z["zone_id"], z["quarter"]) for z in zones]
