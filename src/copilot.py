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
import difflib
import os
import re

from src import copilot_tools as ct

# Central emirate alias table -- canonical name -> every accepted alias/abbreviation, each
# already lowercase (matching is done on normalized, lowercased text; see _normalize_geo_text).
# This is the ONLY place emirate text is resolved: the canonical name it returns is what gets
# passed as the deterministic tool's `emirate` argument, resolved BEFORE that tool ever runs --
# the LLM never sees raw emirate text and never filters a result by emirate itself.
EMIRATE_ALIASES: dict[str, list[str]] = {
    "Abu Dhabi": ["abu dhabi", "abudhabi", "auh", "ad"],
    "Dubai": ["dubai", "dxb"],
    "Sharjah": ["sharjah", "shj"],
    "Ajman": ["ajman", "ajm"],
    "Umm Al Quwain": ["umm al quwain", "umm al-quwain", "uaq"],
    "Ras Al Khaimah": ["ras al khaimah", "ras al-khaimah", "rak"],
    "Fujairah": ["fujairah", "fuj", "fjr"],
}
EMIRATES = list(EMIRATE_ALIASES)  # canonical names only -- kept for anything that just needs the list

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


def _extract_explicit_n(question: str) -> int | None:
    """Same digit/number-word detection as `_extract_n`, but returns None (not a fallback
    default) when no explicit count is present -- lets a caller tell 'no number given' apart
    from 'given N', which matters wherever "no number" and "n=<some number>" mean genuinely
    different things (e.g. get_deteriorating_zones: no number keeps the full flagged list)."""
    q = question.lower()
    m = re.search(r"\b(\d+)\b", q)
    if m:
        return int(m.group(1))
    for word, n in _NUMBER_WORDS.items():
        if re.search(rf"\b{word}\b", q):
            return n
    return None


def _extract_n(question: str, default: int) -> int:
    n = _extract_explicit_n(question)
    return default if n is None else n


def _wants_single_zone(q: str) -> bool:
    """True for genuinely singular phrasing ("which ZONE/AREA...", not "...ZONES/AREAS...") --
    e.g. "Which zone has the lowest experience?" or "which area deteriorated the most?" want
    exactly one answer, not the tool's usual top-N list. \\bzone\\b / \\barea\\b never matches
    inside "zones"/"areas" (no word boundary before the trailing 's'), so this only fires on
    real singular wording, never a plural one."""
    return bool(re.search(r"\b(zone|area|hex(agon)?)\b", q))


def _normalize_geo_text(text: str) -> str:
    """Lowercases, turns hyphens into spaces, and collapses whitespace -- so 'Ras Al-Khaimah',
    'ras al  khaimah', and 'RAS AL KHAIMAH' all normalize to the same string before alias
    matching. Applied only for emirate detection; never changes what reaches the tool/LLM."""
    return re.sub(r"\s+", " ", text.lower().replace("-", " ")).strip()


def _extract_emirate(question: str) -> str | None:
    """Resolves the emirate scope from EMIRATE_ALIASES only -- no fuzzy guessing, so an
    ambiguous-looking short code always resolves the same, explicit way. Matched with word
    boundaries on normalized text, so a two/three-letter code like 'ad' or 'rak' only matches as
    a standalone word/phrase, never as a substring inside an unrelated word."""
    normalized = _normalize_geo_text(question)
    for canonical, aliases in EMIRATE_ALIASES.items():
        for alias in aliases:
            if re.search(rf"\b{re.escape(alias)}\b", normalized):
                return canonical
    return None


# ---------------------------------------------------------------------------
# Metric-extreme detection -- "lowest recorded download speed" etc. This is the fix for the
# router previously mapping ANY "lowest/worst/highest" question to weakest/top-priority
# EXPERIENCE zones: a raw supporting measurement (download_mbps, upload_mbps, latency_ms) is a
# fundamentally different question from the Experience Index, so it must never fall into the
# get_weakest_zones/get_top_priority_zones branches below -- this block runs BEFORE those and
# returns early whenever a metric word is present, with its own min/max/median resolution.
# ---------------------------------------------------------------------------

_METRIC_EXACT_ALIASES: dict[str, list[str]] = {
    "download_mbps": ["download speed", "download", "downlink", "downlod", "downlaod", "down load"],
    "upload_mbps": ["upload speed", "upload", "uplink", "uplod", "uplaod", "up load"],
    "latency_ms": ["latency", "ping", "latancy", "lattency"],
}
_OPERATION_EXACT_ALIASES: dict[str, list[str]] = {
    "min": ["lowest", "minimum", "min"],
    "max": ["highest", "maximum", "max", "higest", "heighest", "highst"],
    "median": ["median"],
}
# "worst"/"best" are metric-RELATIVE, not a fixed min/max: worse download/upload is a LOWER
# number, but worse latency is a HIGHER number (latency: lower is always better) -- resolved
# against the matched metric in _resolve_metric_query below, never assumed to mean "min".
_VALENCE_EXACT_ALIASES: dict[str, list[str]] = {
    "worst": ["worst", "wrost", "worse"],
    "best": ["best", "bset", "better"],
}

# Fuzzy fallback whitelist -- ONLY these words, matched with difflib (deterministic string
# similarity; not an LLM, never asked to interpret meaning) when a typo isn't in the exact-alias
# lists above. _METRIC_FLOOR is the bar to even consider a word "maybe a metric typo" at all
# (below this, an unrelated word must never be treated as a metric question); the higher
# _CONFIDENCE_THRESHOLD is the bar to auto-run the tool without asking first.
_METRIC_FUZZY_WHITELIST = {"download": "download_mbps", "upload": "upload_mbps", "latency": "latency_ms"}
_OPERATION_FUZZY_WHITELIST = {"lowest": "min", "minimum": "min", "highest": "max", "maximum": "max", "median": "median"}
_METRIC_FLOOR = 0.55
_CONFIDENCE_THRESHOLD = 0.72


def _exact_phrase_match(normalized: str, aliases: dict[str, list[str]]) -> str | None:
    for key, terms in aliases.items():
        for term in terms:
            if re.search(rf"\b{re.escape(term)}\b", normalized):
                return key
    return None


def _best_fuzzy_match(normalized: str, whitelist: dict[str, str]) -> tuple[str | None, float]:
    """Best (resolved_value, confidence) across every word in `normalized` against `whitelist`'s
    keys, or (None, 0.0) if no word reaches _METRIC_FLOOR at all."""
    best_value, best_ratio = None, 0.0
    for word in normalized.split():
        for term, value in whitelist.items():
            ratio = difflib.SequenceMatcher(None, word, term).ratio()
            if ratio > best_ratio:
                best_value, best_ratio = value, ratio
    return (best_value, best_ratio) if best_ratio >= _METRIC_FLOOR else (None, best_ratio)


def _resolve_metric_query(question: str) -> dict | None:
    """Returns None if this isn't a metric question at all (no metric word/typo found even
    fuzzily) -- lets route() fall through to its other branches unaffected. Otherwise returns
    {'metric', 'operation', 'confidence'} when confident enough to run, or
    {'clarify': "<best-guess phrase>"} when the match is real but too uncertain to act on
    without asking -- e.g. "Did you mean lowest download speed?" -- never silently guessed."""
    normalized = _normalize_geo_text(question)

    metric = _exact_phrase_match(normalized, _METRIC_EXACT_ALIASES)
    metric_confidence = 1.0 if metric else 0.0
    if metric is None:
        metric, metric_confidence = _best_fuzzy_match(normalized, _METRIC_FUZZY_WHITELIST)
    if metric is None:
        return None  # no metric signal at all -- not this router's question

    operation = _exact_phrase_match(normalized, _OPERATION_EXACT_ALIASES)
    op_confidence = 1.0 if operation else 0.0
    if operation is None:
        valence = _exact_phrase_match(normalized, _VALENCE_EXACT_ALIASES)
        if valence:
            operation, op_confidence = valence, 1.0
    if operation is None:
        operation, op_confidence = _best_fuzzy_match(normalized, _OPERATION_FUZZY_WHITELIST)

    if operation in ("worst", "best"):
        # latency: lower is better, so "worst" = highest latency; download/upload: higher is
        # better, so "worst" = lowest speed. Flip resolved here, once, from the matched metric --
        # never left for the LLM to reason about.
        lower_is_better = metric == "latency_ms"
        wants_bad = operation == "worst"
        operation = ("max" if wants_bad else "min") if lower_is_better else ("min" if wants_bad else "max")

    confidence = min(metric_confidence, op_confidence)
    if operation is None or confidence < _CONFIDENCE_THRESHOLD:
        metric_word = {"download_mbps": "download", "upload_mbps": "upload", "latency_ms": "latency"}[metric]
        op_word = {"min": "lowest", "max": "highest", "median": "median", None: "lowest"}.get(operation, "lowest")
        return {"clarify": f"{op_word} {metric_word} speed" if metric != "latency_ms" else f"{op_word} {metric_word}"}

    return {"metric": metric, "operation": operation, "confidence": confidence}


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
        args = {"emirate": emirate}
        # No explicit count keeps the full flagged list (unchanged default behavior) UNLESS the
        # question is itself singular ("which AREA deteriorated the most" wants exactly one
        # answer, not all 17) -- same singular-phrasing signal get_weakest_zones uses below.
        explicit_n = _extract_explicit_n(q)
        if explicit_n is not None:
            args["n"] = explicit_n
        elif _wants_single_zone(q):
            args["n"] = 1
        return "get_deteriorating_zones", args
    if re.search(r"\banomal|unusual\b", q):
        kind = "temporal" if re.search(r"\bhistory|own past|over time\b", q) else "peer_gap"
        return "get_anomalous_zones", {"kind": kind, "emirate": emirate}
    if re.search(r"\bpopulation\b", q) and re.search(r"\bweak\b", q):
        return "get_high_population_weak_zones", {"emirate": emirate}
    # "which zones have greater than the median download" (a LISTING of zones above the median)
    # is a different question from "what is the median download speed" (a single value) even
    # though both contain 'median'+'download' -- the comparison word ('greater than'/'above'/...)
    # is what makes it a listing; without one, it falls through to the metric-extreme block below.
    if (re.search(r"\bmedian\b", q) and re.search(r"\bdownload\b", q)
            and re.search(r"\b(greater than|above|more than|higher than|over)\b", q)):
        return "get_above_median_download_zones", {"emirate": emirate}

    metric_query = _resolve_metric_query(q)
    if metric_query is not None:
        if "clarify" in metric_query:
            return "clarify", {"suggestion": metric_query["clarify"]}
        return "get_metric_extreme", {
            "metric": metric_query["metric"], "operation": metric_query["operation"], "emirate": emirate,
        }

    if re.search(r"\b(priorit\w*|investigate\w*)\b", q):
        return "get_top_priority_zones", {"n": _extract_n(q, 5), "emirate": emirate}
    # "weakest" is the canonical wording, but a decision-maker is just as likely to ask for the
    # "lowest" or "worst" -- same tool, same ranking (ascending Experience Index), just different
    # words for the same idea.
    if re.search(r"\b(weak(est)?|lowest|worst)\b", q):
        # Singular phrasing means the caller wants exactly one answer, not the usual top-10 list
        # -- e.g. "Which zone has the lowest experience?" should return (and the map should
        # highlight) exactly one zone, not ten.
        default_n = 1 if _wants_single_zone(q) else 10
        return "get_weakest_zones", {"n": _extract_n(q, default_n), "emirate": emirate}
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
    "get_above_median_download_zones": ct.get_above_median_download_zones,
    "get_metric_extreme": ct.get_metric_extreme,
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
    "You explain UAE mobile network experience data to a decision-maker who is not a network "
    "engineer or data scientist. You will be given a user question and a JSON object that was "
    "already computed by a deterministic pipeline (Experience Index, Confidence Score, Peer "
    "Gap, trend, ML anomaly flags, Priority Score). The dashboard map has already highlighted, "
    "directly from that JSON's own zone_id field(s), the exact area(s) it names -- your only "
    "job is to explain in plain language what the highlighted map is showing. "
    "Rules, no exceptions: "
    "1) Keep the answer to 1-3 short sentences, unless the user explicitly asks for more detail. "
    "2) Never state a raw H3 cell ID, and never list the areas one by one -- refer to them as "
    "'these areas' / 'this area' and say they're highlighted on the map. The map is the visual "
    "answer; your text only explains what it shows. "
    "3) Write for a decision-maker, not an engineer: never use the internal terms 'peer "
    "underperformance', 'temporal anomaly', 'population exposure', 'evidence tier', "
    "'percentile rank', 'Isolation Forest', or 'H3' -- say 'worse than similar areas', "
    "'unusual recent changes', 'number of people affected', 'not enough public measurements', "
    "etc. instead. Only use the technical terms if the user's own question already uses them or "
    "explicitly asks for technical/methodology detail. "
    "4) Use only numbers that appear in the JSON -- never calculate, estimate, or round a new "
    "number of your own. "
    "5) If a field is missing or null, say the data is not available -- never guess. "
    "6) This is public, outside-in mobile measurement data with no operator attribution -- "
    "never attribute a result to e& or any specific operator, site, or tower."
)


def _llm_available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def _llm_narrate(question: str, tool_name: str, tool_result, tool_args: dict) -> str:
    import anthropic  # imported lazily so the module loads fine without the package installed

    client = anthropic.Anthropic()
    message = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=200,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": (
                f"Question: {question}\n\nTool called: {tool_name}\n"
                f"Tool arguments (the exact scope already applied, e.g. emirate -- state this "
                f"scope in your answer if one was given; do not infer or apply a different one "
                f"yourself): {tool_args}\n"
                f"Tool result (JSON):\n{tool_result}"
            ),
        }],
    )
    return message.content[0].text


# Plain-language stand-ins for the internal factor/field names -- never shown to a decision-
# maker as raw identifiers (`peer_gap`, `temporal_anomaly`, ...). The underlying data, tools,
# and Priority calculation are untouched; only this narration-facing vocabulary changes.
#   peer underperformance -> worse than similar areas
#   deterioration          -> getting worse over time
#   temporal anomaly       -> unusual recent changes
#   population exposure    -> how many people may be affected
def _join_and(items: list[str]) -> str:
    if len(items) <= 1:
        return items[0] if items else ""
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


_PLAIN_FACTORS = {
    "peer_gap": "how it compares with similar areas",
    "temporal_anomaly": "unusual recent changes",
    "deterioration": "getting worse over time",
    "population": "how many people may be affected",
}

_PRIORITY_CRITERIA_SENTENCE = (
    "They were selected based on how they compare with similar areas, whether performance is "
    "getting worse or changing unusually, and how many people may be affected."
)


def _lead(n: int) -> tuple[str, str]:
    """('This is'/'These are', 'it'/'them') for a result set of size n -- shared by every
    listing narration below so singular vs. plural phrasing never has to be duplicated."""
    return ("This is", "it") if n == 1 else ("These are", "them")


def _scope_suffix(emirate: str | None) -> str:
    """' in {emirate}' (resolved, canonical -- never re-guessed here) or '' -- the fragment every
    listing narration below inserts right after its subject, so a scoped question's answer
    always states the exact scope the tool was actually run against."""
    return f" in {emirate}" if emirate else ""


def _priority_listing_narration(n: int, quarter: str, emirate: str | None) -> str:
    lead, pronoun = _lead(n)
    where = _scope_suffix(emirate)
    subject = "the area that needs the most attention" if n == 1 else f"the {n} areas that need the most attention"
    return f"{lead} {subject}{where} this quarter ({quarter}). {_PRIORITY_CRITERIA_SENTENCE} I highlighted {pronoun} on the map."


def _weakest_listing_narration(n: int, quarter: str, emirate: str | None) -> str:
    lead, pronoun = _lead(n)
    where = f" in {emirate} for {quarter}" if emirate else f" in {quarter}"
    if n == 1:
        return f"This area has the lowest measured mobile experience{where}. I highlighted it in red on the map."
    return f"{lead} the {n} areas with the lowest measured mobile experience{where}. I highlighted {pronoun} on the map."


def _deteriorating_listing_narration(n: int, quarter: str, emirate: str | None) -> str:
    where = _scope_suffix(emirate)
    if n == 1:
        return (f"This area{where} has been getting worse over time compared with similar areas, "
                f"for at least three quarters in a row (as of {quarter}). I highlighted it on the map.")
    lead, pronoun = _lead(n)
    return (f"{lead} {n} areas{where} where performance has been getting worse over time compared "
            f"with similar areas, for at least three quarters in a row (as of {quarter}). "
            f"I highlighted {pronoun} on the map.")


def _anomalous_listing_narration(question: str, n: int, quarter: str, emirate: str | None) -> str:
    lead, pronoun = _lead(n)
    subject = "one area" if n == 1 else f"{n} areas"
    where = _scope_suffix(emirate)
    is_temporal = bool(re.search(r"\bhistory|own past|over time\b", question.lower()))
    clause = "compared with their own recent history" if is_temporal else "compared with similar areas"
    return f"{lead} {subject}{where} showing unusual recent changes {clause}, in {quarter}. I highlighted {pronoun} on the map."


def _high_population_weak_listing_narration(n: int, quarter: str, emirate: str | None) -> str:
    lead, pronoun = _lead(n)
    subject = "one area" if n == 1 else f"{n} areas"
    where = _scope_suffix(emirate)
    return (f"{lead} {subject}{where} with weak mobile experience where a large number of people "
            f"may be affected, in {quarter}. I highlighted {pronoun} on the map.")


def _above_median_download_narration(n: int, quarter: str, emirate: str | None) -> str:
    lead, pronoun = _lead(n)
    where = _scope_suffix(emirate)
    subject = "the area with" if n == 1 else f"{n} areas with"
    return f"{lead} {subject}{where} download speeds above the median for {quarter}. I highlighted {pronoun} on the map."


_LISTING_NARRATORS = {
    "get_top_priority_zones": lambda q, n, quarter, emirate: _priority_listing_narration(n, quarter, emirate),
    "get_priority_zones": lambda q, n, quarter, emirate: _priority_listing_narration(n, quarter, emirate),
    "get_weakest_zones": lambda q, n, quarter, emirate: _weakest_listing_narration(n, quarter, emirate),
    "get_deteriorating_zones": lambda q, n, quarter, emirate: _deteriorating_listing_narration(n, quarter, emirate),
    "get_anomalous_zones": lambda q, n, quarter, emirate: _anomalous_listing_narration(q, n, quarter, emirate),
    "get_high_population_weak_zones": lambda q, n, quarter, emirate: _high_population_weak_listing_narration(n, quarter, emirate),
    "get_above_median_download_zones": lambda q, n, quarter, emirate: _above_median_download_narration(n, quarter, emirate),
}


def _template_narrate(question: str, tool_name: str, tool_result, tool_args: dict | None = None) -> str:
    """Deterministic fallback narrator -- no LLM call, plain string formatting over the same
    JSON an LLM would receive. Exists so the full pipeline is demoable and testable with zero
    external dependencies; swap in `_llm_narrate` the moment a key is available. Every branch
    stays to 1-3 short, plain-language sentences (see _PLAIN_FACTORS) and never states a raw H3
    id or lists areas one by one -- the map (driven straight from this same tool_result, in the
    frontend) is the visual answer; this text only explains what it shows. `tool_args['emirate']`
    (already resolved by route()/EMIRATE_ALIASES before the tool ever ran) is read here only to
    state that scope in the sentence -- never to filter or re-derive it."""
    tool_args = tool_args or {}
    if tool_name in _LISTING_NARRATORS:
        emirate = tool_args.get("emirate")
        if not tool_result:
            where = f" in {emirate}" if emirate else ""
            return f"No areas matched this query{where} at the requested evidence threshold."
        return _LISTING_NARRATORS[tool_name](question, len(tool_result), tool_result[0]["quarter"], emirate)

    if tool_name == "get_zone_details":
        if tool_result.get("evidence_status") == "insufficient_evidence":
            return (f"Not enough public measurements this quarter ({tool_result['tests']} tests, "
                    f"{tool_result['devices']} devices) to score this area reliably.")
        if tool_result.get("error"):
            return "No data found for this area this quarter."

        # 'low' evidence tier (10-29 tests): a real Experience Index exists, but
        # src/priority.py never computes a Priority Score for this tier at all -- no factor
        # breakdown or confidence claim to make beyond "not eligible."
        if tool_result.get("evidence_tier") != "full":
            return (f"This area's Experience Index is {tool_result['experience_index']}/100, but "
                    f"there aren't enough public measurements yet (only {tool_result['tests']} tests) "
                    f"for it to be considered for the Priority list.")

        q = question.lower()
        if re.search(r"\bconfiden", q):
            return (f"Confidence is {tool_result['confidence_score']}/100, based on "
                     f"{tool_result['tests']} tests from {tool_result['devices']} devices across "
                     f"{tool_result['quarters_observed']}/8 quarters.")
        if re.search(r"\bmeasurements?\b.*(support|behind|recommend)", q):
            return (f"This recommendation is based on {tool_result['tests']} tests from "
                    f"{tool_result['devices']} devices across {tool_result['quarters_observed']}/8 "
                    f"quarters ({tool_result['confidence_score']}/100 confidence).")

        # Default: "why is this zone high priority" and any other zone-scoped question that
        # reaches this tool.
        high_factors = [_PLAIN_FACTORS[k] for k, v in tool_result["priority_factors"].items() if v == "High"]
        factor_text = _join_and(high_factors) if high_factors else "no single standout factor"
        return (f"This area scored {tool_result['priority_score']} on Priority, driven mainly by "
                f"{factor_text}. Its Experience Index is {tool_result['experience_index']}/100 vs. "
                f"a typical {tool_result['peer_group_median_experience']} for similar areas.")

    if tool_name == "get_zone_peer_comparison":
        return (f"Compared with {tool_result['peer_group_size']} similar areas this quarter, this "
                f"area's Experience Index is {tool_result['experience_index']} vs. a typical "
                f"{tool_result['peer_group_median_experience']} for that group "
                f"({tool_result['peer_gap']:+.1f} pts).")

    if tool_name == "get_zone_trend":
        cur = "getting worse over time" if tool_result.get("currently_deteriorating") else "not currently getting worse"
        return (f"This area is {cur}, trending {tool_result['trend_pts_per_qtr']} Experience-Index "
                f"points per quarter over {tool_result['quarters_observed']} observed quarters.")

    if tool_name == "get_metric_extreme":
        if tool_result.get("error") or tool_result.get("value") is None:
            where = f" in {tool_result['emirate']}" if tool_result.get("emirate", "All UAE") != "All UAE" else ""
            return f"No measurements are available for this{where} this quarter."
        metric_label = {"download_mbps": "download speed", "upload_mbps": "upload speed",
                         "latency_ms": "latency"}[tool_result["metric"]]
        op_label = {"min": "lowest recorded", "max": "highest recorded", "median": "median"}[tool_result["operation"]]
        where = f" in {tool_result['emirate']}" if tool_result.get("emirate", "All UAE") != "All UAE" else ""
        sentence = (f"The {op_label} {metric_label}{where} in {tool_result['quarter']} is "
                    f"{tool_result['value']} {tool_result['unit']}.")
        if tool_result.get("zone_id"):
            sentence += " I highlighted the matching area on the map."
        return sentence

    if tool_name == "get_coverage_summary":
        return (
            f"{tool_result['emirate']}, {tool_result['quarter']}: {tool_result['classified_zones']} of "
            f"{tool_result['measured_zones']} measured areas have enough public measurements to score, "
            f"representing {tool_result['population_represented_pct']}% of the population "
            f"({tool_result['population_represented']:,} of {tool_result['population_universe']:,}). "
            f"{tool_result['priority_zones']} area(s) flagged for investigation."
        )

    if tool_name == "get_methodology":
        return f"{tool_result.get('metric')}: {tool_result.get('formula', tool_result)}"

    return "I could not map this question to a supported tool. Try one of the ten canonical questions."


def narrate(question: str, tool_name: str, tool_result, tool_args: dict | None = None) -> tuple[str, str]:
    """Returns (answer_text, mode) where mode is 'llm' or 'template'. `tool_args` is the exact,
    already-resolved arguments the tool was called with (e.g. `{'emirate': 'Ras Al Khaimah'}`) --
    passed through so the narration can state the scope it was given, not re-guess it from
    result content (which would be wrong for a legitimately empty result) or leave it out."""
    tool_args = tool_args or {}
    if _llm_available():
        try:
            return _llm_narrate(question, tool_name, tool_result, tool_args), "llm"
        except Exception as exc:  # network/key/package issues -- fall back rather than crash the demo
            return _template_narrate(question, tool_name, tool_result, tool_args) + f"\n[LLM call failed, used fallback: {exc}]", "template"
    return _template_narrate(question, tool_name, tool_result, tool_args), "template"


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
    if tool_name == "clarify":
        # A metric/typo match was found but confidence was below _CONFIDENCE_THRESHOLD -- ask
        # rather than guess. No tool ran, so tool_result stays None; the map is untouched (the
        # frontend only ever moves the map from a tool's own zone_id field(s), and there isn't
        # one here).
        return {"question": question, "tool": None, "tool_args": None, "tool_result": None,
                "answer": f"Did you mean {tool_args['suggestion']}?", "mode": "clarification"}

    if quarter and "quarter" in _TOOLS[tool_name].__code__.co_varnames:
        tool_args = {**tool_args, "quarter": quarter}
    tool_args = {k: v for k, v in tool_args.items() if v is not None}

    tool_result = _TOOLS[tool_name](**tool_args)
    answer, mode = narrate(question, tool_name, tool_result, tool_args)
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
