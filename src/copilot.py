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
from pathlib import Path

from src import copilot_tools as ct
from src.trends import CONSECUTIVE_DECLINES_REQUIRED


def _load_dotenv_if_present() -> None:
    """Minimal, stdlib-only loader for a `.env` file at the repo root (ANTHROPIC_API_KEY, most
    importantly) -- so any entry point that imports this module (`src.serve_dashboard`, a
    script, a notebook) picks up a locally-configured key without whoever launched that process
    having to `export`/`set` it in that exact shell first. Missing that is easy and silent: the
    process just falls back to deterministic-only mode with no error, which is exactly what
    made a real API key look like it "wasn't working" when the server process launching it
    predates the key being configured. Never overrides a variable already set in the real
    environment (an explicit `export`/`set` always wins over the file). Not python-dotenv --
    this repo has zero GenAI dependency without a key configured at all (see requirements.txt),
    so this stays a few stdlib lines rather than adding a package just for this."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip()


_load_dotenv_if_present()

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
    # Site/tower attribution phrased without the word "e&" attached directly (e.g. "tell me the
    # site ID responsible", "which tower serves this zone") -- still an attribution question this
    # public, outside-in dataset structurally cannot answer.
    r"\bsite\s*id\b", r"\bwhich\s+(site|tower|cell)\b", r"\b(site|tower|cell)\b.{0,20}\bresponsible\b",
    # Cross-operator comparison ("is e& worse than du", "does e& have better coverage than du",
    # "which is better, e& or du") -- a different question from "is this e& data" (that one gets
    # DATA_SOURCE_MESSAGE below): this asks the system to rank operators against each other, which
    # it has no operator-attributed data to do at all.
    r"\b(e&|du|etisalat|virgin mobile)\b.{0,40}\b(worse|better|worst|best|compare|beat|outperform)\b",
    r"\b(worse|better|worst|best|compare|beat|outperform)\b.{0,40}\b(e&|du|etisalat|virgin mobile)\b",
    # Customer-level questions ("how many customers use this", "customer complaints") -- broader
    # than the exact "customer count"/"number of customers" phrasings already listed above.
    r"\bcustomers?\b",
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

AREA_NOT_FOUND_MESSAGE = "I couldn't confidently match that to an area in the map. Could you check the name?"

# Genuinely couldn't be understood as any supported analytics request -- deterministic routing
# found nothing AND the semantic classifier (when connected) either agreed or wasn't available.
# Deliberately says nothing about "canonical questions" or any other internal/project
# terminology -- that phrasing is for this repo's own docs and tests, never surfaced to a real
# dashboard user.
UNMATCHED_MESSAGE = (
    "I'm not sure what you'd like to check. You can ask me about areas that need attention, "
    "mobile experience, download or upload speeds, changes over time, or a specific area."
)

# ---------------------------------------------------------------------------
# Basic conversational check -- runs BEFORE refusal/routing/semantic classification, entirely
# deterministic (a fixed, closed set of greeting words, not a judgment call an LLM should make).
# Only matches a message that IS a bare greeting and nothing else (`fullmatch`, allowing trailing
# punctuation/a name) -- "hi, which areas need attention?" still falls through to real routing
# unaffected; only a standalone "hi"/"hello"/"good morning"/etc. gets the canned reply below.
# ---------------------------------------------------------------------------

GREETING_MESSAGE = (
    "Hi! I can help you explore UAE public mobile experience. You can ask which areas need "
    "attention, where experience is strongest or weakest, or what has changed over time."
)

_GREETING_RE = re.compile(
    r"^\s*(hi+|hello+|hey+|hiya|yo|howdy|greetings|sup|what'?s\s*up|"
    r"good\s*(morning|afternoon|evening|day))\s*(there|team|everyone)?\s*[!.,?]*\s*$",
    re.IGNORECASE,
)


def check_greeting(question: str) -> str | None:
    return GREETING_MESSAGE if _GREETING_RE.match(question) else None


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


# ---------------------------------------------------------------------------
# Experience-ranking detection -- "where is experience highest/lowest", "best/worst experience",
# "top experience areas", etc. Same exact-phrase-then-fuzzy, confidence-gated shape as
# _resolve_metric_query above (never a separate hardcoded regex per sentence), but for the
# Experience Index ranking intent specifically -- NOT a raw measurement, so this never touches
# get_metric_extreme/_METRIC_EXACT_ALIASES above. A direction (highest/lowest) is the one thing
# this resolver needs beyond the shared metric/confidence machinery, since Experience Index is
# always "higher is better" (no latency-style flip needed, unlike _resolve_metric_query).
# ---------------------------------------------------------------------------

_EXPERIENCE_METRIC_EXACT_ALIASES: dict[str, list[str]] = {
    "experience_index": ["experience index", "experience score", "mobile experience", "experience"],
}
_EXPERIENCE_METRIC_FUZZY_WHITELIST = {"experience": "experience_index"}

_EXPERIENCE_DIRECTION_EXACT_ALIASES: dict[str, list[str]] = {
    "highest": ["highest", "best", "strongest", "top", "maximum", "max"],
    "lowest": ["lowest", "weakest", "weak", "worst", "poorest", "minimum", "min", "bottom"],
}
_EXPERIENCE_DIRECTION_FUZZY_WHITELIST = {
    "highest": "highest", "best": "highest", "strongest": "highest", "top": "highest",
    "lowest": "lowest", "weakest": "lowest", "worst": "lowest", "poorest": "lowest", "weak": "lowest",
    "bottom": "lowest",
}


def _resolve_experience_query(question: str) -> dict | None:
    """Returns None if this isn't an Experience-ranking question at all (no 'experience' word,
    or a close typo of it, present anywhere) -- lets route()'s other branches (population+weak,
    priority, the bare weak/lowest/worst bucket, get_metric_extreme, etc.) handle the question
    unaffected, exactly like _resolve_metric_query does for its own metric. Once the Experience
    metric IS detected, a direction is required; if none is found confidently, returns
    {'clarify': ...} rather than guessing one -- same contract _resolve_metric_query already
    uses for an ambiguous operation. Otherwise returns {'direction': 'highest' | 'lowest',
    'confidence': ...}."""
    normalized = _normalize_geo_text(question)

    metric = _exact_phrase_match(normalized, _EXPERIENCE_METRIC_EXACT_ALIASES)
    metric_confidence = 1.0 if metric else 0.0
    if metric is None:
        metric, metric_confidence = _best_fuzzy_match(normalized, _EXPERIENCE_METRIC_FUZZY_WHITELIST)
    if metric is None:
        return None  # no Experience signal at all -- not this router's question

    direction = _exact_phrase_match(normalized, _EXPERIENCE_DIRECTION_EXACT_ALIASES)
    direction_confidence = 1.0 if direction else 0.0
    if direction is None:
        direction, direction_confidence = _best_fuzzy_match(normalized, _EXPERIENCE_DIRECTION_FUZZY_WHITELIST)

    confidence = min(metric_confidence, direction_confidence)
    if direction is None or confidence < _CONFIDENCE_THRESHOLD:
        return {"clarify": "highest experience or lowest experience"}

    return {"direction": direction, "confidence": confidence}


# ---------------------------------------------------------------------------
# Metric-threshold detection -- "areas above/below median download/upload/latency". Reuses the
# exact same metric vocabulary (_METRIC_EXACT_ALIASES/_METRIC_FUZZY_WHITELIST) _resolve_metric_
# query already defines above, so download/upload/latency typo tolerance is defined once and
# shared, never duplicated per metric or per comparison direction -- this is the fix for the
# router previously only recognizing this intent for 'download' specifically (get_above_median_
# download_zones), with 'above' hardcoded and no 'below' direction at all.
# ---------------------------------------------------------------------------

_MEDIAN_THRESHOLD_EXACT_ALIASES: dict[str, list[str]] = {"median": ["median"]}
_MEDIAN_THRESHOLD_FUZZY_WHITELIST = {"median": "median"}

# Pure numeric-direction words -- "areas above median latency" really does mean
# latency > the median latency value, full stop, regardless of whether a higher latency is good
# or bad for that metric.
_COMPARISON_EXACT_ALIASES: dict[str, list[str]] = {
    "above": ["higher than", "above", "greater than", "over"],
    "below": ["lower than", "below", "less than", "under"],
}

# Valence words -- "better than median download" / "worse than median latency" -- ARE a judgment
# call, and (like _resolve_metric_query's own 'worst'/'best' flip) need to know latency is "lower
# is better" while download/upload are "higher is better" before they can resolve to a literal
# numeric direction. Checked only when no _COMPARISON_EXACT_ALIASES word matched at all, so a
# question using both ("higher than median, which is better") still resolves on the literal word.
_THRESHOLD_VALENCE_EXACT_ALIASES: dict[str, list[str]] = {
    "better": ["better than", "better"],
    "worse": ["worse than", "worse"],
}


def _resolve_metric_threshold_query(question: str) -> dict | None:
    """'areas above/below median download/upload/latency' -- a LISTING of zones vs. a threshold
    statistic. Structurally different from _resolve_metric_query's 'what is the median download
    speed' (a single value): both mention 'median' + a metric, but only a LISTING question also
    carries an explicit comparison-direction or valence word (above/below/greater than/better
    than/worse than/...). Returns None (not 'clarify') whenever a required signal is completely
    absent -- in particular, 'median' present with NO direction/valence word is not ambiguous, it
    is simply the other (value) question, and route() must let it fall straight through to
    _resolve_metric_query unaffected. Once median + metric + direction are ALL present, low
    combined confidence (e.g. a metric typo too weak to trust) asks for clarification instead of
    guessing, exactly like the other resolvers above."""
    normalized = _normalize_geo_text(question)

    threshold = _exact_phrase_match(normalized, _MEDIAN_THRESHOLD_EXACT_ALIASES)
    threshold_confidence = 1.0 if threshold else 0.0
    if threshold is None:
        threshold, threshold_confidence = _best_fuzzy_match(normalized, _MEDIAN_THRESHOLD_FUZZY_WHITELIST)
    if threshold is None:
        return None  # no threshold word (or typo of one) at all -- not this router's question

    metric = _exact_phrase_match(normalized, _METRIC_EXACT_ALIASES)
    metric_confidence = 1.0 if metric else 0.0
    if metric is None:
        metric, metric_confidence = _best_fuzzy_match(normalized, _METRIC_FUZZY_WHITELIST)
    if metric is None:
        return None  # 'median' + a direction word but no recognizable metric -- not this router's question

    comparison = _exact_phrase_match(normalized, _COMPARISON_EXACT_ALIASES)
    if comparison is None:
        valence = _exact_phrase_match(normalized, _THRESHOLD_VALENCE_EXACT_ALIASES)
        if valence:
            # latency: lower is better, so "worse" = above the median; download/upload: higher is
            # better, so "worse" = below the median -- same flip _resolve_metric_query already
            # applies for 'worst'/'best', resolved here once from the matched metric.
            lower_is_better = metric == "latency_ms"
            wants_worse = valence == "worse"
            comparison = ("above" if wants_worse else "below") if lower_is_better else ("below" if wants_worse else "above")
    if comparison is None:
        return None  # 'median' with no direction/valence word -- the VALUE question, not a listing

    confidence = min(threshold_confidence, metric_confidence)
    if confidence < _CONFIDENCE_THRESHOLD:
        metric_word = {"download_mbps": "download", "upload_mbps": "upload", "latency_ms": "latency"}[metric]
        return {"clarify": f"areas {comparison} median {metric_word}"}

    return {"metric": metric, "comparison": comparison, "confidence": confidence}


def _global_metric_intent_present(question: str) -> bool:
    """True only when a GLOBAL metric question (_resolve_metric_query or
    _resolve_metric_threshold_query) is confident enough to actually run a tool -- i.e. returns a
    real result dict, not its own 'clarify' fallback (metric word present but no recognizable
    operation/comparison word) and not None (no metric word at all). Used by route()'s zone-scoped
    raw-metric branch to tell "areas above median download" (a real global intent, must not be
    hijacked by a merely-selected zone) apart from "what is the download speed" with a zone
    selected (no global operation word at all -- exactly the bare per-zone value question that
    branch exists to answer)."""
    metric_query = _resolve_metric_query(question)
    if metric_query is not None and "clarify" not in metric_query:
        return True
    threshold_query = _resolve_metric_threshold_query(question)
    return threshold_query is not None and "clarify" not in threshold_query


# ---------------------------------------------------------------------------
# Severity modifier resolution (2026-09-17) -- shared by the anomalous-areas and
# deteriorating-areas branches below, and by the semantic classifier's own 'severity' field
# validation, so "high/medium/low" and "highest/strongest/top N" are recognized once,
# consistently, rather than reimplemented per branch. Deliberately narrow: only the exact band
# words resolve to a band filter; superlative words resolve to a RANKING request instead (rank
# by the real factor score, cap at N) -- "areas that ARE high severity" and "the N most severe
# areas" are different questions, never conflated.
# ---------------------------------------------------------------------------

_SEVERITY_BAND_ALIASES: dict[str, list[str]] = {
    "high": ["high", "severe"],
    "medium": ["medium", "moderate"],
    "low": ["low", "mild"],
}

# Superlative words that mean "rank by score, give me the top N" -- never a band filter, even
# though "highest"/"strongest" sound like they could mean 'high' severity. See
# _wants_severity_ranking below.
_SEVERITY_RANKING_WORDS = ["highest", "strongest", "most", "top"]


def _resolve_severity_band(question: str) -> str | None:
    """'low'/'medium'/'high' if the question uses that exact band word (or a light synonym),
    else None. A superlative ('highest', 'strongest', ...) never resolves here -- callers check
    _wants_severity_ranking for that, a different question with different tool arguments."""
    normalized = _normalize_geo_text(question)
    for canonical, aliases in _SEVERITY_BAND_ALIASES.items():
        for alias in aliases:
            if re.search(rf"\b{re.escape(alias)}\b", normalized):
                return canonical
    return None


def _wants_severity_ranking(question: str) -> bool:
    """True for 'highest/strongest/most/top N ...' phrasing -- "rank by the actual factor/
    score and return the top N," never a Low/Medium/High band filter."""
    q = question.lower()
    return bool(re.search(r"\b(" + "|".join(_SEVERITY_RANKING_WORDS) + r")\b", q))


# ---------------------------------------------------------------------------
# Peer-group filter -- "show me industrial areas", "which zones are commercial", "show
# low-density residential areas". A plain categorical filter over the composite peer-group
# classifier's 4 stable groups (never an OSM land-use tag -- see project methodology), matched
# by common synonyms so a caller never has to type the exact stored label. PEER_GROUPS itself
# (the exact strings this resolves TO) lives in copilot_tools.py, the module that actually
# knows what's in the data -- imported here rather than duplicated, so the two can't drift.
# ---------------------------------------------------------------------------

PEER_GROUP_ALIASES: dict[str, list[str]] = {
    "industrial": ["industrial", "industry", "industries"],
    "commercial/urban-core": ["commercial/urban-core", "commercial", "urban core", "urban-core"],
    "low-density residential": [
        "low-density residential", "low density residential", "residential", "low-density", "low density",
    ],
    "rural/edge": ["rural/edge", "rural"],
}


def _resolve_peer_group(question: str) -> str | None:
    normalized = _normalize_geo_text(question)
    for canonical, aliases in PEER_GROUP_ALIASES.items():
        for alias in aliases:
            if re.search(rf"\b{re.escape(alias)}\b", normalized):
                return canonical
    return None


# ---------------------------------------------------------------------------
# Area/place lookup -- "show me Seyouh", "where is Deira", "find Al Nahda", "take me to Al
# Majaz", "show Khalifa City on the map". This is the LAST resolver route() tries, run only on a
# question that every canonical-question branch above has already declined -- so a phrasing that
# happens to start with a locate-style trigger word (e.g. "show me the weakest zones") is claimed
# by its own, earlier keyword branch first and never reaches this one. The valid area-name
# universe is loaded fresh from the processed dataset each call (ct.get_known_area_names(), a
# plain cached read of zone_priority.parquet) -- never a hardcoded list of place names, and the
# LLM's own world knowledge of UAE geography is never consulted to decide whether a name is real.
# ---------------------------------------------------------------------------

_LOCATE_TRIGGER_PATTERNS = [
    re.compile(r"^\s*show\s+me\s+(.+?)\s+on\s+the\s+map\s*$", re.IGNORECASE),
    re.compile(r"^\s*show\s+me\s+(.+?)\s*$", re.IGNORECASE),
    re.compile(r"^\s*show\s+(.+?)\s+on\s+the\s+map\s*$", re.IGNORECASE),
    re.compile(r"^\s*where\s+is\s+(.+?)\s*\??\s*$", re.IGNORECASE),
    re.compile(r"^\s*what(?:'s|\s+is)\s+happening\s+in\s+(.+?)\s*\??\s*$", re.IGNORECASE),
    re.compile(r"^\s*find\s+(.+?)\s*$", re.IGNORECASE),
    re.compile(r"^\s*take\s+me\s+to\s+(.+?)\s*$", re.IGNORECASE),
    re.compile(r"^\s*locate\s+(.+?)\s*$", re.IGNORECASE),
    re.compile(r"^\s*(?:go|zoom|pan)\s+to\s+(.+?)\s*$", re.IGNORECASE),
]

_AREA_FUZZY_THRESHOLD = 0.72  # same bar _CONFIDENCE_THRESHOLD uses elsewhere in this file
_AREA_TIE_MARGIN = 0.03  # names within this much of the best score are treated as a genuine tie

# Stored aliases/spelling variants for area-name matching -- keyed by the NORMALIZED phrase (see
# _normalize_area_text), valued by the canonical area_name it should resolve to. Deliberately
# empty by default: this is the "stored aliases, if available" matching tier the brief asks for,
# not a place to hardcode the examples this feature must already handle through fuzzy matching
# alone (see _area_name_ratio) -- add an entry here only for a real alias fuzzy matching can't
# reach (e.g. a nickname that shares no characters with the OSM-derived name).
AREA_NAME_ALIASES: dict[str, str] = {}


def _extract_locate_phrase(question: str) -> str | None:
    """First locate-style trigger phrase found in `question` -- matched case-insensitively, but
    run against the ORIGINAL (not lowercased) question text so the extracted phrase keeps its
    real casing, which is what lets _match_area_name's tier-1 exact match (case-sensitive,
    against the real 'Al Hamra'-style capitalization stored in the data) actually fire. Also
    strips a trailing 'on the map' and a trailing 'in <emirate>' clause -- the emirate scope
    for a question like 'find Yaraah in Dubai' is already resolved separately, over the whole
    question, by _extract_emirate; it must not also leak into the text being area-name-matched
    here. Returns None if the question doesn't use any locate-style phrasing at all, in which
    case route() must not touch this resolver's result."""
    for pattern in _LOCATE_TRIGGER_PATTERNS:
        m = pattern.match(question)
        if m:
            phrase = re.sub(r"\s+on\s+the\s+map\s*$", "", m.group(1).strip(), flags=re.IGNORECASE).strip()
            phrase = _strip_emirate_clause(phrase)
            return phrase or None
    return None


def _strip_emirate_clause(phrase: str) -> str:
    """Strips a trailing emirate clause -- ' in <emirate alias>' (e.g. 'Yaraah in Dubai' ->
    'Yaraah'), ', <emirate alias>' (e.g. 'Al Mafraq, Abu Dhabi' -> 'Al Mafraq', the common "Area,
    Emirate" way of writing a UAE location), or a bare trailing alias with no delimiter at all
    (e.g. 'Al Mafraq Abu Dhabi' -> 'Al Mafraq', the "omitted comma" form) -- reusing
    EMIRATE_ALIASES, never a second, place-name-specific list. Without stripping these, the
    literal emirate tail stayed glued onto the area-name text handed to fuzzy matching, where it
    outscored the true, shorter match against the unrelated-but-textually-closer emirate name
    itself (which is also a real area_name in this dataset, e.g. 'Abu Dhabi') -- silently
    returning the wrong, much bigger area. The bare-trailing-alias form only fires when something
    remains before it (never strips a query that IS just the emirate name to nothing) and only
    when no real area_name in this dataset itself ends in that alias as a trailing word (verified
    empty in practice -- see tests/test_t6_copilot_evaluation.py) -- so this can't silently
    truncate a genuine multi-word place name. Hyphens are normalized to spaces only for detecting
    the match (1-for-1 character replacement, so the match position still lines up with the
    original, case-preserved `phrase` being sliced)."""
    hyphen_normalized = phrase.replace("-", " ")
    for aliases in EMIRATE_ALIASES.values():
        for alias in aliases:
            m = re.search(rf"(?:\s+in|\s*,)\s+{re.escape(alias)}\s*$", hyphen_normalized, flags=re.IGNORECASE)
            if m:
                return phrase[:m.start()].strip()
    for aliases in EMIRATE_ALIASES.values():
        for alias in aliases:
            m = re.search(rf"\s+{re.escape(alias)}\s*$", hyphen_normalized, flags=re.IGNORECASE)
            if m and phrase[:m.start()].strip():
                return phrase[:m.start()].strip()
    return phrase


def _normalize_area_text(text: str) -> str:
    """Case/punctuation/hyphen/space-insensitive normalization for area-name matching only --
    kept separate from _normalize_geo_text (emirate matching), because area names carry
    apostrophes _normalize_geo_text never needs to touch (e.g. \"Al Sharya'a\")."""
    text = text.lower().replace("-", " ")
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


_SENTENCE_MARKER_WORDS = frozenset({
    "that", "is", "are", "am", "was", "were", "isn't", "aren't", "wasn't", "weren't",
    "doesn't", "don't", "didn't", "as", "than", "over", "doing", "similar", "getting",
})


def _looks_like_a_sentence_not_a_name(phrase: str) -> bool:
    """True when a locate-style trigger ('show me X') matched, but X reads like an ordinary
    English clause rather than an attempted place name -- e.g. "areas that are worsening over
    time" vs. "Al Seyouh Suburb" or even genuine gibberish like "asdkjaksjdaksjd". Real UAE
    place names in this dataset run up to 6 words (checked against `ct.get_known_area_names()`
    at the time this was written) but never contain a plain English function word like these --
    used only to decide whether a FAILED fuzzy match should fall through to "unmatched" (this
    might not have been a locate question at all) or commit to a confident "not found" (a
    genuine, if garbled, short place-name attempt) -- see route()'s own locate-lookup branch."""
    words = set(_normalize_area_text(phrase).split())
    return bool(words & _SENTENCE_MARKER_WORDS)


def _strip_leading_article(tokens: list[str]) -> list[str]:
    """Drops a leading 'al' (the definite article) before a FUZZY comparison only -- almost
    every name in this dataset starts with 'Al ', so leaving it in inflates similarity between
    otherwise-unrelated names purely because they share that one common prefix token (e.g. 'al
    majaz' scoring closer to 'Al Maha' than the two names actually are, just from the shared
    'al'). Exact/normalized matching (tiers 1-2 in _match_area_name) never uses this -- a literal
    'Al Ain' typed by the user should still equality-match 'Al Ain' exactly, article and all."""
    return tokens[1:] if tokens and tokens[0] == "al" and len(tokens) > 1 else tokens


def _area_name_ratio(query_norm: str, name_norm: str) -> float:
    """Best difflib similarity between the normalized query phrase and one normalized area name
    (both with any leading 'al' stripped first -- see _strip_leading_article): the whole-string
    ratio, or (when the query has fewer words than the name) the best ratio against any same-
    length contiguous run of the name's own words. This is what lets a single mistyped word
    ('seyoh') match strongly against the one word it corresponds to inside a multi-word name ('Al
    Seyouh Suburb') without ever hardcoding that name -- a whole-string comparison alone would be
    dragged down by the name's other, unrelated words."""
    q_tokens = _strip_leading_article(query_norm.split())
    n_tokens = _strip_leading_article(name_norm.split())
    query_core, name_core = " ".join(q_tokens), " ".join(n_tokens)
    best = difflib.SequenceMatcher(None, query_core, name_core).ratio()
    k = len(q_tokens)
    if 0 < k < len(n_tokens):
        for i in range(len(n_tokens) - k + 1):
            window = " ".join(n_tokens[i:i + k])
            best = max(best, difflib.SequenceMatcher(None, query_core, window).ratio())
    return best


def _match_area_name(phrase: str, known: list[dict]) -> dict:
    """Matches free text against the finite set of area names actually present in the processed
    dataset (`known`, from ct.get_known_area_names()) -- in order: exact match, normalized match,
    stored alias, then fuzzy (typo-tolerant) match against ONLY that real set. Never invents a
    name. Returns one of:
      {'status': 'matched', 'area_name': <canonical>}           -- one confident name
      {'status': 'ambiguous', 'candidates': [<canonical>, ...]} -- >1 DISTINCT name tied at top
      {'status': 'no_match'}                                    -- nothing close enough to trust
    (Two rows in `known` that share one area_name but differ only by emirate are ONE name here --
    that's a separate, emirate-level ambiguity route() resolves afterward, not a naming one.)"""
    canonical_names = sorted({row["area_name"] for row in known})
    if not canonical_names:
        return {"status": "no_match"}

    if phrase in canonical_names:  # 1) exact match (raw, case-sensitive)
        return {"status": "matched", "area_name": phrase}

    phrase_norm = _normalize_area_text(phrase)  # 2) normalized match
    normalized_lookup: dict[str, list[str]] = {}
    for name in canonical_names:
        normalized_lookup.setdefault(_normalize_area_text(name), []).append(name)
    if phrase_norm in normalized_lookup:
        hits = normalized_lookup[phrase_norm]
        if len(hits) == 1:
            return {"status": "matched", "area_name": hits[0]}
        return {"status": "ambiguous", "candidates": hits}

    alias_target = AREA_NAME_ALIASES.get(phrase_norm)  # 3) stored alias/spelling variant
    if alias_target and alias_target in canonical_names:
        return {"status": "matched", "area_name": alias_target}

    # 4) fuzzy match against only the real area-name list above
    scored = sorted(
        ((name, _area_name_ratio(phrase_norm, _normalize_area_text(name))) for name in canonical_names),
        key=lambda pair: pair[1], reverse=True,
    )
    best_name, best_score = scored[0]
    if best_score < _AREA_FUZZY_THRESHOLD:
        return {"status": "no_match"}
    tied = [name for name, score in scored if score >= best_score - _AREA_TIE_MARGIN]
    if len(tied) > 1:
        return {"status": "ambiguous", "candidates": tied}
    return {"status": "matched", "area_name": best_name}


def route(question: str, zone_id: str | None = None) -> tuple[str, dict]:
    """Returns (tool_name, tool_args) for the first canonical pattern the question matches.
    Zone-scoped questions (why/confidence/trend/measurements) require a `zone_id` to already
    be in context -- exactly like the dashboard's drill-down panel, where a question is asked
    about the currently-selected zone, not typed from nothing."""
    q = question.lower()
    emirate = _extract_emirate(question)

    # 'causing'/'behind'/'flagged'/'its priority' paraphrase "why was this flagged" ("what is
    # causing the poor performance here?", "why was this zone flagged?", "what factors contribute
    # to its priority?") without the literal word 'why' -- this only ever reaches route() after
    # check_refusal() has already cleared it of every genuine operator/root-cause attribution
    # phrasing (e.g. literal "root cause", "e& site"), so it's safe to answer the same way "why is
    # this zone high priority" already does: the zone's own deterministic Priority factors, never
    # an invented network cause. Without 'flagged'/'factors...priorit', these fell through to the
    # GLOBAL priority branch further below, silently ignoring the selected zone_id and returning
    # the national top-5 list instead of explaining the one zone actually asked about.
    if zone_id and re.search(
        r"\bwhy\b.*(priorit|invest)|\bcausing\b|\bwhat'?s\s+behind\b|\bwhat\s+is\s+behind\b"
        r"|\bflagged\b|\bfactors?\b.*priorit|\bits\s+priorit", q,
    ):
        return "get_zone_details", {"zone_id": zone_id}
    # 'evidence' catches "does this area have enough public evidence?" -- without it, this fell
    # through every zone-scoped branch (no 'confiden' substring) into the global fuzzy Experience
    # resolver below, where 'evidence' fuzzy-matches 'experience' (ratio 0.67, above the fuzzy
    # floor but below the confident-auto-run bar) and asks an unrelated "highest or lowest
    # experience?" clarification instead of answering the actual evidence-sufficiency question.
    if zone_id and re.search(r"\bconfiden|\bevidence\b", q):
        return "get_zone_details", {"zone_id": zone_id}
    if zone_id and re.search(r"\bchanged?\b.*(quarter|since)", q):
        return "get_zone_trend", {"zone_id": zone_id}
    # A raw-measurement quarter-over-quarter question ("did download improve?", "did latency
    # deteriorate?") -- distinct from the Experience-Index-trend question above (no "quarter"/
    # "since" wording here, just one of the three raw metrics paired with a change verb) but
    # still answered from the same get_zone_trend tool, which now also carries each raw metric's
    # own *_change_qoq field (see copilot_tools.get_zone_trend) precisely for this question.
    if zone_id and re.search(r"\b(download|upload|latency)\b.*\b(improve|increase|decreas\w*|declin\w*|deteriorat\w*|better|worse|worsen\w*)\b", q):
        return "get_zone_trend", {"zone_id": zone_id}
    if zone_id and re.search(r"\bmeasurements?\b.*(support|behind|recommend)", q):
        return "get_zone_details", {"zone_id": zone_id}
    # A bare raw-value or evidence-volume question about the selected zone ("what is the download
    # speed here?", "how many tests and devices support this?") -- no change verb (that's the QoQ
    # branch above) and no "measurements ... support/behind/recommend" phrasing (the branch just
    # above), just asking what the zone's own numbers are. get_zone_details already carries
    # download_mbps/upload_mbps/latency_ms/tests/devices for this zone -- no new tool needed.
    # Guarded by "and no CONFIDENT global metric intent": a zone happening to be selected must
    # never hijack an explicitly GLOBAL listing/value question ("areas above median download",
    # "what is the highest download speed") into a single-zone answer just because it also
    # contains the word "download" -- those already carry their own operation/comparison word
    # (median/above/below/highest/better than/...), which a bare per-zone value question never
    # does. A global resolver returning its OWN 'clarify' dict (metric word present but no
    # operation/comparison word at all -- exactly the bare-value shape this branch is for) does
    # NOT count as a confident global intent, so this branch still claims those.
    if zone_id and re.search(r"\b(download|upload|latency|tests?|devices?)\b", q) and not _global_metric_intent_present(q):
        return "get_zone_details", {"zone_id": zone_id}
    if zone_id and re.search(r"\bpeer\b", q):
        return "get_zone_peer_comparison", {"zone_id": zone_id}

    # 'getting worse' added alongside the literal 'worse over time' -- without it, "where is
    # experience getting worse?" has no 'deteriorat'/'declin' root at all, so it fell through to
    # the Experience-ranking resolver below, where 'worse' fuzzy-matches 'worst' (a CURRENT-
    # weakest, not a TREND, question) and silently answers the wrong question.
    # 'decreas\w*' added (2026-09-17) alongside 'declin\w*' for the identical reason 'declin\w*'
    # itself was added: without it, "where is experience decreasing over time?" has no
    # deterioration-root word at all, so it fell through to the Experience-ranking resolver
    # below, where "experience" alone (with no clear direction word) claims the question with a
    # confident-looking but WRONG clarify ("highest experience or lowest experience?"),
    # short-circuiting it before it could ever reach "unmatched" and the semantic classifier.
    # 'worsening' NOT added here even though it means the same thing -- deliberate: this
    # deterministic branch only needs to catch phrasings that were already reliable, unambiguous
    # keyword matches; an unfamiliar synonym like 'worsening' is exactly what the semantic
    # classifier (route()'s caller falls back to it on "unmatched") is for, and "worsening" alone
    # doesn't get wrongly claimed by any earlier, more-eager resolver the way "decreasing" did.
    if re.search(r"\b(deteriorat\w*|declin\w*|decreas\w*|worse over time|getting worse)\b", q):
        args = {"emirate": emirate}
        # No explicit count keeps the full flagged list (unchanged default behavior) UNLESS the
        # question is itself singular ("which AREA deteriorated the most" wants exactly one
        # answer, not all 17) -- same singular-phrasing signal get_weakest_zones uses below.
        explicit_n = _extract_explicit_n(q)
        if explicit_n is not None:
            args["n"] = explicit_n
        elif _wants_single_zone(q):
            args["n"] = 1
        # Severity (2026-09-17): "high/medium/low deterioration" filters on deterioration_band
        # (a DIFFERENT question from the bare "is it deteriorating" flag above); a superlative
        # ("strongest"/"most") instead asks for a ranked top N by severity, which takes
        # precedence over a plain band word if both somehow appear.
        if _wants_severity_ranking(q) and "n" not in args:
            args["n"] = 5
        elif not _wants_severity_ranking(q):
            severity = _resolve_severity_band(q)
            if severity:
                args["severity"] = severity
        return "get_deteriorating_zones", args
    # Broadened from the original 'anomal(ous)'/'unusual' pair: 'unusual\b' alone never matched
    # its own adverb form ('unusually'), and 'underperform'/'peer gap(s)' are realistic ways to
    # ask for the same peer-comparison listing without either root word at all (e.g. "which zones
    # underperform comparable areas", "show the biggest peer gaps").
    if re.search(r"\banomal\w*|unusual\w*|underperform\w*|\bpeer gaps?\b", q):
        # 'temporal' checked literally (2026-09-17 fix) -- "areas with high TEMPORAL anomaly"
        # has no 'history'/'own past'/'over time' phrase at all, so this used to silently fall
        # through to kind='peer_gap' (an entirely different ML model) despite the question
        # naming the exact kind it wanted.
        kind = "temporal" if re.search(r"\btemporal\b|\bhistory|own past|over time\b", q) else "peer_gap"
        args = {"kind": kind, "emirate": emirate}
        # Severity (2026-09-17): "high/medium/low temporal anomaly" must filter on the zone's
        # own *_band (the Priority factor's severity), never quietly substitute the ML flag's
        # top-~8%-contamination set for it -- see get_anomalous_zones' own docstring for why
        # those are not guaranteed to be the same set. A superlative ("highest"/"strongest")
        # instead ranks by the real factor score and caps at N.
        if _wants_severity_ranking(q):
            args["n"] = _extract_explicit_n(q) or 5
        else:
            severity = _resolve_severity_band(q)
            if severity:
                args["severity"] = severity
        return "get_anomalous_zones", args
    # 'people' accepted alongside the literal 'population' -- "which weak areas affect the most
    # people?" never says "population" at all, and previously fell through to the bare
    # weak/lowest/worst bucket further below (a plain Experience-ranking listing, silently
    # dropping the population-exposure intent).
    if (re.search(r"\bpopulation\b", q) or re.search(r"\bpeople\b", q)) and re.search(r"\bweak\b", q):
        return "get_high_population_weak_zones", {"emirate": emirate}
    # Metric-threshold intent ("areas above/below median download/upload/latency") -- one
    # resolver reused across all three raw metrics and both comparison directions (see
    # _resolve_metric_threshold_query above), never a route per metric/direction/phrasing
    # combination. Runs BEFORE _resolve_metric_query below because both can match on
    # 'median'+a metric word -- only this one also requires an explicit comparison-direction
    # word; when that's absent (e.g. "What is the median download?"), it returns None here and
    # falls straight through to _resolve_metric_query's own median-VALUE handling, unaffected.
    threshold_query = _resolve_metric_threshold_query(q)
    if threshold_query is not None:
        if "clarify" in threshold_query:
            return "clarify", {"suggestion": threshold_query["clarify"]}
        return "get_metric_threshold_zones", {
            "metric": threshold_query["metric"], "comparison": threshold_query["comparison"], "emirate": emirate,
        }

    # Experience-ranking intent ("where is experience highest/lowest", "best/worst experience",
    # "top experience areas", ...) -- one resolver for both directions (see
    # _resolve_experience_query above), never a separate route per phrasing. Runs BEFORE the raw
    # metric-extreme resolver below (Experience Index is a computed composite, never one of
    # download/upload/latency) and BEFORE the bare weak/lowest/worst bucket further down, which
    # stays as a fallback for phrasing that omits the word "experience" entirely (e.g. "show the
    # weakest zones in Dubai") -- that bucket's tool name/args shape is left completely
    # unchanged, so this only ADDS the 'highest' direction and never touches existing 'lowest'
    # routing.
    experience_query = _resolve_experience_query(q)
    if experience_query is not None:
        if "clarify" in experience_query:
            return "clarify", {"suggestion": experience_query["clarify"]}
        # Same singular-vs-plural default-n rule the bare weak/lowest/worst bucket already uses
        # below -- "which zone has the highest experience" wants exactly one zone, not ten.
        default_n = 1 if _wants_single_zone(q) else 10
        tool = "get_weakest_zones" if experience_query["direction"] == "lowest" else "get_strongest_zones"
        return tool, {"n": _extract_n(q, default_n), "emirate": emirate}

    metric_query = _resolve_metric_query(q)
    if metric_query is not None:
        if "clarify" in metric_query:
            return "clarify", {"suggestion": metric_query["clarify"]}
        return "get_metric_extreme", {
            "metric": metric_query["metric"], "operation": metric_query["operation"], "emirate": emirate,
        }

    if re.search(r"\b(priorit\w*|investigate\w*)\b", q):
        # Same singular-vs-plural default-n rule get_weakest_zones/get_strongest_zones already use
        # below -- "What is THE highest-priority AREA?" wants exactly one zone, not the usual
        # top-5 list; "Which areas need the highest priority?" (plural) keeps the unchanged
        # default of 5.
        default_n = 1 if _wants_single_zone(q) else 5
        return "get_top_priority_zones", {"n": _extract_n(q, default_n), "emirate": emirate}
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

    # Peer-group filter (2026-09-17) -- "show me industrial areas", "which zones are commercial".
    # Checked late (after every ranking/severity/coverage branch above), since none of those
    # keyword sets overlap with the 4 peer-group names/synonyms -- a genuine peer-group question
    # never reaches here having already been mis-claimed by an earlier branch.
    peer_group = _resolve_peer_group(question)
    if peer_group is not None:
        args = {"peer_group": peer_group, "emirate": emirate}
        explicit_n = _extract_explicit_n(q)
        if explicit_n is not None:
            args["n"] = explicit_n
        return "get_zones_by_peer_group", args

    # Area/place lookup -- last resort; see the block above for why this ordering never steals a
    # question any earlier branch already claims. Uses the ORIGINAL `question` (not the
    # lowercased `q`) so the extracted phrase keeps its real casing for exact-match matching.
    locate_phrase = _extract_locate_phrase(question)
    if locate_phrase is not None:
        tool_name, tool_args = _resolve_locate_phrase(locate_phrase, emirate)
        # A failed match on something that reads like a full English clause (2026-09-17 fix)
        # falls through to "unmatched" instead of committing to "not found" -- this heuristic is
        # the LAST resort in this file, so a "show me <sentence>" question that isn't really
        # about a place (e.g. "show me areas that are worsening over time") deserves a real shot
        # at the semantic classifier, not a locate failure message. A short/garbled genuine
        # place-name attempt (e.g. "show me asdkjaksjdaksjd") has no sentence-marker word, so it
        # still resolves to a confident "not found" exactly as before.
        if tool_name == "locate_area_not_found" and _looks_like_a_sentence_not_a_name(locate_phrase):
            return "unmatched", {}
        return tool_name, tool_args

    return "unmatched", {}


def _resolve_locate_phrase(phrase: str, emirate: str | None) -> tuple[str, dict]:
    """Shared by route()'s own locate-trigger branch above and the semantic classifier's
    'locate_area' intent below -- one place-name resolution path, so a free-text place name
    coming from either a matched trigger phrase or an LLM's own reading of the question goes
    through the exact same exact/normalized/alias/fuzzy matching against the real area-name
    universe (`_match_area_name`), never a second, looser one. `emirate` is always the
    deterministically-resolved one (`_extract_emirate` over the whole original question) --
    this function never receives or trusts an emirate guess from anywhere else."""
    known = ct.get_known_area_names()
    match = _match_area_name(phrase, known)
    if match["status"] == "no_match":
        return "locate_area_not_found", {}
    if match["status"] == "ambiguous":
        return "clarify", {"suggestion": _join_or(match["candidates"])}
    area_name = match["area_name"]
    emirates_for_name = sorted({row["emirate"] for row in known if row["area_name"] == area_name})
    if emirate:
        if emirate not in emirates_for_name:
            # The user named an emirate this place doesn't actually have a zone in -- treat
            # as no confident match rather than silently ignoring the stated scope.
            return "locate_area_not_found", {}
        resolved_emirate = emirate
    elif len(emirates_for_name) == 1:
        resolved_emirate = emirates_for_name[0]
    else:
        return "clarify", {"suggestion": f"{area_name} in {' or '.join(emirates_for_name)}"}
    return "locate_area", {"area_name": area_name, "emirate": resolved_emirate}


_TOOLS = {
    "get_zone_details": ct.get_zone_details,
    "get_zone_peer_comparison": ct.get_zone_peer_comparison,
    "get_zone_trend": ct.get_zone_trend,
    "get_top_priority_zones": ct.get_top_priority_zones,
    "get_priority_zones": ct.get_priority_zones,
    "get_weakest_zones": ct.get_weakest_zones,
    "get_strongest_zones": ct.get_strongest_zones,
    "get_deteriorating_zones": ct.get_deteriorating_zones,
    "get_anomalous_zones": ct.get_anomalous_zones,
    "get_high_population_weak_zones": ct.get_high_population_weak_zones,
    "get_zones_by_peer_group": ct.get_zones_by_peer_group,
    "get_metric_threshold_zones": ct.get_metric_threshold_zones,
    "get_metric_extreme": ct.get_metric_extreme,
    "get_coverage_summary": ct.get_coverage_summary,
    "get_methodology": ct.get_methodology,
    "locate_area": ct.locate_area,
}
# get_priority_zones is deliberately NOT reachable from route() -- "which N areas first" (a
# canonical question, caller-chosen N) and "every zone actually flagged" (a fixed, data-
# determined set, used only by generate_all_priority_zone_briefs below) are different
# questions. It's registered in _TOOLS so answer_question() can still dispatch to it directly
# if a caller passes the tool name explicitly.


# ---------------------------------------------------------------------------
# Semantic intent classification -- the LLM half of the hybrid architecture.
#
#     question -> route() (deterministic) -> confident match? -> run tool
#                                           -> "unmatched"?     -> classify_intent_semantic()
#                                                                  -> validated intent? -> run tool
#                                                                  -> low confidence?    -> clarify
#                                                                  -> no match/no LLM?   -> "unmatched"
#
# Called ONLY from answer_question() when route() has already returned "unmatched" -- it never
# runs on, and never overrides, a confident deterministic match. The model chooses from a fixed
# enum of the SAME intents/tools route() itself can already produce (SEMANTIC_INTENTS below) --
# it is never offered a new capability, never asked for a raw number, score, or area name it
# invents from nothing, and it never computes the answer itself (that's still entirely
# `src/copilot_tools.py`, exactly as for a deterministic match). Every field the model returns is
# re-validated against a strict allowlist in `_validate_semantic_output` before anything runs;
# anything that doesn't validate -- wrong intent name, out-of-enum parameter, missing required
# parameter, low self-reported confidence -- falls back to "clarify" or the same "could not map"
# message a deterministic miss already gives, never a guess. `n` and `emirate` are resolved the
# same deterministic way as everywhere else in this file (`_extract_n`/`_extract_emirate`) rather
# than trusted from the model's own output, for the same reason route() never lets the LLM decide
# an emirate or a count: both are simple, unambiguous lookups this file already does reliably.
# ---------------------------------------------------------------------------

SEMANTIC_MODEL = "claude-sonnet-5"
SEMANTIC_MAX_TOKENS = 300
# Same bar every other fuzzy resolver in this file uses (_CONFIDENCE_THRESHOLD) -- one
# "confident enough to act without asking" bar across the whole file, deterministic or semantic.
SEMANTIC_CONFIDENCE_THRESHOLD = _CONFIDENCE_THRESHOLD

# intent label (what the model sees/returns) -> which tool it maps to, whether it requires a
# zone already selected (offered to the model only when zone_id is not None), and which EXTRA
# parameters (beyond the always-deterministic n/emirate) the model may supply for it. This is
# the complete list of intents the model can ever choose -- adding a new one here is the only
# way to expand what the semantic path can reach, and it must correspond to a tool `route()`
# could already reach deterministically (this file adds no new analytics capability).
SEMANTIC_INTENTS: dict[str, dict] = {
    "priority_areas": {"tool": "get_top_priority_zones", "zone_scoped": False, "takes_n": True, "extra": []},
    "weakest_areas": {"tool": "get_weakest_zones", "zone_scoped": False, "takes_n": True, "extra": []},
    "strongest_areas": {"tool": "get_strongest_zones", "zone_scoped": False, "takes_n": True, "extra": []},
    "deteriorating_areas": {"tool": "get_deteriorating_zones", "zone_scoped": False, "takes_n": True, "extra": ["severity"]},
    "anomalous_areas": {"tool": "get_anomalous_zones", "zone_scoped": False, "takes_n": True, "extra": ["kind", "severity"]},
    "population_exposure_weak_areas": {"tool": "get_high_population_weak_zones", "zone_scoped": False, "takes_n": True, "extra": []},
    "peer_group_filter": {"tool": "get_zones_by_peer_group", "zone_scoped": False, "takes_n": True, "extra": ["peer_group"]},
    "metric_threshold_areas": {"tool": "get_metric_threshold_zones", "zone_scoped": False, "takes_n": False, "extra": ["metric", "comparison"]},
    "metric_extreme": {"tool": "get_metric_extreme", "zone_scoped": False, "takes_n": False, "extra": ["metric", "operation"]},
    "coverage_summary": {"tool": "get_coverage_summary", "zone_scoped": False, "takes_n": False, "extra": []},
    "locate_area": {"tool": "locate_area", "zone_scoped": False, "takes_n": False, "extra": ["area_name"]},
    "zone_details": {"tool": "get_zone_details", "zone_scoped": True, "takes_n": False, "extra": []},
    "zone_peer_comparison": {"tool": "get_zone_peer_comparison", "zone_scoped": True, "takes_n": False, "extra": []},
    "zone_trend": {"tool": "get_zone_trend", "zone_scoped": True, "takes_n": False, "extra": []},
}

_SEMANTIC_INTENT_DESCRIPTIONS = """\
- priority_areas: which areas should be investigated, enhanced, improved, or focused on first \
(a decision-maker asking where to act -- ranked by the Priority Score, which already combines \
peer comparison, trend, anomalies, and population).
- weakest_areas: where mobile experience is currently lowest/worst, as a plain ranking (not "what \
should we act on").
- strongest_areas: where mobile experience is currently highest/best.
- deteriorating_areas: where experience has been getting WORSE OVER TIME / declining / \
trending down over several recent quarters (a TREND question -- "worsening", "declining", \
"getting worse", "experience is decreasing over time" all mean this, even with none of those \
exact words). severity ("low"/"medium"/"high"), if the question names one, filters to that \
exact severity of decline; without it, every currently-declining area is returned regardless \
of how mild or severe. A superlative ("strongest"/"most severe decline") means rank-and-cap, \
not a severity filter -- put the count in n instead and leave severity null.
- anomalous_areas: kind="peer_gap" for a zone scoring unusually WORSE THAN ITS PEERS RIGHT NOW \
-- this is also where everyday, non-technical phrasings belong: "not as good as similar areas", \
"worse than comparable places", "underperforming compared with similar areas" all mean \
kind="peer_gap", even though they never say the words "peer gap". kind="temporal" for unusual \
vs. the SAME zone's own past/history (needs "over time"/"history"/"own past"/"temporal" wording \
specifically -- otherwise assume peer_gap). severity ("low"/"medium"/"high") filters to that \
exact severity if the question names one (e.g. "areas with HIGH temporal anomaly"); without \
it, returns every zone the ML model actually flagged, regardless of severity. A superlative \
("highest"/"strongest anomalies") means rank-and-cap by the real anomaly score -- put the \
count in n and leave severity null (these are different questions: "areas that ARE anomalous" \
vs. "the N most anomalous areas" vs. "areas that are HIGH severity").
- population_exposure_weak_areas: weak experience where a large number of people are affected.
- peer_group_filter: a plain categorical filter by the zone's development/land-use type -- \
"industrial areas", "commercial areas", "residential areas", "rural areas". peer_group is \
REQUIRED and must be exactly one of: "industrial", "commercial/urban-core", \
"low-density residential", "rural/edge". Map an everyday synonym to the closest of these four \
(e.g. "industry zones"/"industrial locations" -> "industrial"; "urban core"/"commercial \
zones" -> "commercial/urban-core"; "residential"/"suburban" -> "low-density residential"; \
"rural"/"edge areas"/"outskirts" -> "rural/edge") -- never invent a fifth category.
- metric_threshold_areas: which areas are above/below the median for one raw measurement \
(download, upload, or latency) -- metric + comparison ("above"/"below") required.
- metric_extreme: the single lowest/highest/median value of one raw measurement (download, \
upload, or latency) -- metric + operation ("min"/"max"/"median") required.
- coverage_summary: how much of the population or area has enough measurements to be scored.
- locate_area: show, find, or locate one specific named place on the map -- area_name required \
(the place name exactly as the user wrote it, do not correct its spelling yourself).
- clarify: the question is genuinely ambiguous between two or more of the above.
- none: the question does not match any of the above at all (including anything about a specific \
operator/site/root-cause -- that is refused before you are ever called, so if you see one here \
treat it as "none").\
"""

_SEMANTIC_ZONE_INTENT_DESCRIPTIONS = """\
- zone_details: why the currently-selected area was flagged, what is driving its Priority Score, \
or its confidence/evidence level.
- zone_peer_comparison: how the currently-selected area compares specifically to similar areas.
- zone_trend: how the currently-selected area has changed over time.\
"""


def _build_semantic_system_prompt(zone_selected: bool) -> str:
    zone_block = f"\n{_SEMANTIC_ZONE_INTENT_DESCRIPTIONS}" if zone_selected else ""
    zone_note = (
        " A specific area is currently selected on the map, so the zone_* intents below are "
        "also available for a question about 'this area'/'here'/'this zone'."
        if zone_selected else
        " No specific area is currently selected, so do not choose a zone_* intent no matter how "
        "the question is phrased."
    )
    return (
        "You classify a decision-maker's free-text question into ONE of a fixed set of "
        "supported analytics intents for a UAE mobile-network dashboard. You are NOT answering "
        "the question and NOT computing any number, score, or ranking yourself -- a separate "
        "deterministic tool does that from verified data; your only job is choosing WHICH tool "
        "applies and its parameters, from the question's meaning. Paraphrases, unfamiliar "
        "wording, and spelling mistakes should still map correctly if the underlying meaning "
        "matches one of these -- e.g. 'which areas should I enhance for the future' means the "
        "same thing as 'which areas should we investigate first' (priority_areas)."
        + zone_note +
        " Never invent an intent, parameter, or value outside exactly what is listed below; if "
        "you are not sure a parameter applies, leave it null. Give your own honest confidence "
        "(0.0-1.0) that the chosen intent is correct -- a low number is expected and fine for a "
        "genuinely unclear question, and is what tells the system to ask the user to clarify "
        "instead of guessing.\n\nIntents:\n" + _SEMANTIC_INTENT_DESCRIPTIONS + zone_block
    )


def _build_semantic_tool_schema(allowed_intents: list[str]) -> dict:
    return {
        "name": "classify_intent",
        "description": "Classify the user's question into one supported analytics intent.",
        "input_schema": {
            "type": "object",
            "properties": {
                "intent": {"type": "string", "enum": allowed_intents},
                "confidence": {
                    "type": "number",
                    "description": "Your own confidence (0.0-1.0) that this intent is correct.",
                },
                "n": {
                    "type": ["integer", "null"],
                    "description": "Explicit count of areas requested (e.g. 'top 3' -> 3), if any; else null.",
                },
                "metric": {"type": ["string", "null"], "enum": ["download_mbps", "upload_mbps", "latency_ms", None]},
                "operation": {"type": ["string", "null"], "enum": ["min", "max", "median", None]},
                "comparison": {"type": ["string", "null"], "enum": ["above", "below", None]},
                "kind": {"type": ["string", "null"], "enum": ["peer_gap", "temporal", None]},
                "severity": {
                    "type": ["string", "null"],
                    "enum": ["low", "medium", "high", None],
                    "description": (
                        "Only when the question names an exact severity band ('high'/'medium'/"
                        "'low' deterioration or temporal/peer-gap anomaly). Null for a "
                        "superlative ('highest'/'strongest'/'most severe') -- that means rank "
                        "and cap by score instead, so put the count in `n` and leave this null."
                    ),
                },
                "peer_group": {
                    "type": ["string", "null"],
                    "enum": ["industrial", "commercial/urban-core", "low-density residential", "rural/edge", None],
                    "description": "Only for intent='peer_group_filter' -- the closest of these four canonical values.",
                },
                "area_name": {
                    "type": ["string", "null"],
                    "description": "The place name exactly as written in the question, only for intent='locate_area'.",
                },
                "clarify_suggestion": {
                    "type": ["string", "null"],
                    "description": (
                        "Only for intent='clarify': a short, natural-language phrase a decision-"
                        "maker would recognise, describing what you think they meant -- e.g. "
                        "'the areas with the weakest experience' or 'the median download speed'. "
                        "It will be inserted into \"Did you mean ...?\", so end it with no "
                        "trailing question mark and never use an internal intent name (e.g. "
                        "'coverage_summary') or any other snake_case identifier."
                    ),
                },
            },
            "required": ["intent", "confidence"],
        },
    }


def _validate_semantic_output(raw: dict, allowed_intents: list[str], zone_id: str | None,
                               question: str) -> dict:
    """Every field the model returned, re-checked against the allowlist -- the model's own JSON
    is never dispatched to a tool unexamined. Returns one of:
      {'status': 'ok', 'tool': ..., 'tool_args': ..., 'intent': ..., 'confidence': ...}
      {'status': 'clarify', 'suggestion': ...}
      {'status': 'none'}   -- invalid, unrecognised, zone-scoped with no zone, or genuinely 'none'
    `n`/`emirate` are deliberately NOT read from `raw` here -- see module note above; they come
    from this file's own deterministic extractors over `question`, same as every other route."""
    intent = raw.get("intent")
    confidence = raw.get("confidence")
    confidence = confidence if isinstance(confidence, (int, float)) else 0.0

    if intent not in allowed_intents or intent == "none":
        return {"status": "none"}
    if intent == "clarify" or confidence < SEMANTIC_CONFIDENCE_THRESHOLD:
        suggestion = raw.get("clarify_suggestion")
        suggestion = suggestion if isinstance(suggestion, str) and suggestion.strip() else "rephrasing your question"
        suggestion = suggestion.strip().rstrip("?").rstrip()  # this always gets wrapped in "Did you mean ...?" below
        return {"status": "clarify", "suggestion": suggestion}

    spec = SEMANTIC_INTENTS[intent]
    if spec["zone_scoped"] and not zone_id:
        return {"status": "none"}

    if intent == "locate_area":
        area_name = raw.get("area_name")
        if not isinstance(area_name, str) or not area_name.strip():
            return {"status": "none"}
        tool_name, tool_args = _resolve_locate_phrase(area_name.strip(), _extract_emirate(question))
        if tool_name in ("locate_area_not_found", "clarify"):
            return {"status": tool_name, **tool_args}
        return {"status": "ok", "tool": tool_name, "tool_args": tool_args, "intent": intent, "confidence": confidence}

    tool_args: dict = {}
    if spec["zone_scoped"]:
        tool_args["zone_id"] = zone_id
    else:
        emirate = _extract_emirate(question)
        if emirate:
            tool_args["emirate"] = emirate
        if spec["takes_n"]:
            n = _extract_explicit_n(question)
            if n is not None:
                tool_args["n"] = n

    for field in spec["extra"]:
        if field in ("kind", "severity"):
            continue  # both handled below, deterministically -- never trusted from the model
        value = raw.get(field)
        valid_values = {
            "metric": ("download_mbps", "upload_mbps", "latency_ms"),
            "operation": ("min", "max", "median"),
            "comparison": ("above", "below"),
            "peer_group": ct.PEER_GROUPS,
        }[field]
        if value not in valid_values:
            return {"status": "none"}  # a required parameter for this intent didn't validate
        tool_args[field] = value
    if "kind" in spec["extra"]:
        kind = raw.get("kind")
        tool_args["kind"] = kind if kind in ("peer_gap", "temporal") else "peer_gap"
        # Deterministic override (2026-09-17), same fix and same reasoning as route()'s own
        # anomalous-areas branch: a literal "temporal"/"history"/"own past"/"over time" in the
        # question wins over whatever the model itself guessed -- this is a simple, reliably
        # parseable signal, so it is resolved the same way n/emirate always are here, never left
        # to the model's own judgment.
        if re.search(r"\btemporal\b|\bhistory|own past|over time\b", question.lower()):
            tool_args["kind"] = "temporal"
    if "severity" in spec["extra"]:
        # Also deterministic, same reasoning -- "high/medium/low" and "highest/strongest/top N"
        # are simple, closed word classes route() already resolves the same way; the model's
        # own 'severity' field is intentionally never read here.
        if _wants_severity_ranking(question):
            tool_args["n"] = tool_args.get("n") or _extract_explicit_n(question) or 5
        else:
            severity = _resolve_severity_band(question)
            if severity:
                tool_args["severity"] = severity

    return {"status": "ok", "tool": spec["tool"], "tool_args": tool_args, "intent": intent, "confidence": confidence}


def classify_intent_semantic(question: str, zone_id: str | None = None) -> dict:
    """The semantic fallback itself. Always returns a dict with a 'status' key
    ('ok' | 'clarify' | 'none' | 'unavailable' | 'error') plus a 'usage' key (input_tokens/
    output_tokens, or None if no API call was made) -- so a caller can always tell exactly what
    happened and log/display real token usage, never a guess. Never raises: any exception from
    the API call itself is caught and reported as status='error', same "fall back, don't crash
    the demo" contract `narrate()` already uses for the narration-stage LLM call."""
    if not _llm_available():
        return {"status": "unavailable", "usage": None}
    try:
        import anthropic  # imported lazily, same as _llm_narrate -- module loads without it installed
    except ImportError:
        return {"status": "unavailable", "usage": None}

    allowed_intents = [name for name, spec in SEMANTIC_INTENTS.items()
                       if zone_id is not None or not spec["zone_scoped"]] + ["clarify", "none"]

    try:
        client = anthropic.Anthropic()
        message = client.messages.create(
            model=SEMANTIC_MODEL,
            max_tokens=SEMANTIC_MAX_TOKENS,
            system=_build_semantic_system_prompt(zone_id is not None),
            tools=[_build_semantic_tool_schema(allowed_intents)],
            tool_choice={"type": "tool", "name": "classify_intent"},
            messages=[{"role": "user", "content": question}],
        )
    except Exception as exc:
        return {"status": "error", "usage": None, "error": str(exc)}

    usage = {"input_tokens": message.usage.input_tokens, "output_tokens": message.usage.output_tokens}
    tool_use = next((block for block in message.content if block.type == "tool_use"), None)
    if tool_use is None:
        return {"status": "error", "usage": usage, "error": "model returned no tool_use block"}

    result = _validate_semantic_output(tool_use.input, allowed_intents, zone_id, question)
    result["usage"] = usage
    result["raw_model_output"] = tool_use.input
    return result


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
    "2) Never state a raw H3 cell ID. For a result listing multiple areas, never list them one "
    "by one -- refer to them as 'these areas' and say they're highlighted on the map; the map is "
    "the visual answer, your text only explains what it shows. For a single zone-scoped question "
    "(about the one area already selected on the map), you may name that area using the JSON's "
    "own area_name/emirate fields exactly as given, e.g. 'Muwaileh, Sharjah' -- never invent or "
    "abbreviate the name, and never substitute the H3 id for it. "
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


def _join_or(items: list[str]) -> str:
    if len(items) <= 1:
        return items[0] if items else ""
    if len(items) == 2:
        return f"{items[0]} or {items[1]}"
    return ", ".join(items[:-1]) + f", or {items[-1]}"


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


def _strongest_listing_narration(n: int, quarter: str, emirate: str | None) -> str:
    lead, pronoun = _lead(n)
    where = f" in {emirate} for {quarter}" if emirate else f" in {quarter}"
    if n == 1:
        return f"This area has the highest measured mobile experience{where}. I highlighted it on the map."
    return f"{lead} the {n} areas with the highest measured mobile experience{where}. I highlighted {pronoun} on the map."


def _severity_clause(question: str) -> str:
    """' (high severity)' / ' (medium severity)' / ' (low severity)' / '' -- re-resolved from
    the question text (same pattern _metric_threshold_listing_narration already uses for its
    own metric/comparison), so the narration names the severity it actually filtered on
    whenever the question named one, without route() having to thread it through separately."""
    severity = _resolve_severity_band(question)
    return f" ({severity} severity)" if severity else ""


def _deteriorating_listing_narration(question: str, n: int, quarter: str, emirate: str | None) -> str:
    where = _scope_suffix(emirate)
    severity = _severity_clause(question)
    persistence = f"{CONSECUTIVE_DECLINES_REQUIRED}+ quarters in a row"
    if n == 1:
        return (f"This area{where} has been getting worse over time compared with the national trend, "
                f"for at least {persistence} (as of {quarter}){severity}. I highlighted it on the map.")
    lead, pronoun = _lead(n)
    return (f"{lead} {n} areas{where} where performance has been getting worse over time compared "
            f"with the national trend, for at least {persistence} (as of {quarter}){severity}. "
            f"I highlighted {pronoun} on the map.")


def _anomalous_listing_narration(question: str, n: int, quarter: str, emirate: str | None) -> str:
    lead, pronoun = _lead(n)
    subject = "one area" if n == 1 else f"{n} areas"
    where = _scope_suffix(emirate)
    is_temporal = bool(re.search(r"\btemporal\b|\bhistory|own past|over time\b", question.lower()))
    clause = "compared with their own recent history" if is_temporal else "compared with similar areas"
    return (f"{lead} {subject}{where} showing unusual recent changes {clause}, in {quarter}"
            f"{_severity_clause(question)}. I highlighted {pronoun} on the map.")


def _peer_group_listing_narration(question: str, n: int, quarter: str, emirate: str | None) -> str:
    lead, pronoun = _lead(n)
    subject = "one area" if n == 1 else f"{n} areas"
    where = _scope_suffix(emirate)
    peer_group = _resolve_peer_group(question) or "that type"
    return f"{lead} {subject}{where} classified as {peer_group}, in {quarter}. I highlighted {pronoun} on the map."


def _high_population_weak_listing_narration(n: int, quarter: str, emirate: str | None) -> str:
    lead, pronoun = _lead(n)
    subject = "one area" if n == 1 else f"{n} areas"
    where = _scope_suffix(emirate)
    return (f"{lead} {subject}{where} with weak mobile experience where a large number of people "
            f"may be affected, in {quarter}. I highlighted {pronoun} on the map.")


_METRIC_WORDS = {"download_mbps": "download", "upload_mbps": "upload", "latency_ms": "latency"}


def _metric_threshold_listing_narration(question: str, n: int, quarter: str, emirate: str | None) -> str:
    """Metric + comparison direction are re-resolved from `question` here (same pattern
    _anomalous_listing_narration already uses for its own 'temporal vs peer_gap' distinction),
    not re-derived from tool_result -- by construction this narrator only ever runs on a
    question that already routed successfully to get_metric_threshold_zones, so the resolution
    is guaranteed non-None/non-clarify and deterministic (pure function of the same text
    route() itself used)."""
    resolved = _resolve_metric_threshold_query(question)
    metric_word = _METRIC_WORDS[resolved["metric"]]
    comparison_word = resolved["comparison"]  # "above" or "below" -- a literal numeric direction
    where = f" in {emirate} for {quarter}" if emirate else f" for {quarter}"
    lead, pronoun = _lead(n)
    subject = "the area with" if n == 1 else "the areas with"
    return f"{lead} {subject} {metric_word} speeds {comparison_word} the median{where}. I highlighted {pronoun} on the map."


_LISTING_NARRATORS = {
    "get_top_priority_zones": lambda q, n, quarter, emirate: _priority_listing_narration(n, quarter, emirate),
    "get_priority_zones": lambda q, n, quarter, emirate: _priority_listing_narration(n, quarter, emirate),
    "get_weakest_zones": lambda q, n, quarter, emirate: _weakest_listing_narration(n, quarter, emirate),
    "get_strongest_zones": lambda q, n, quarter, emirate: _strongest_listing_narration(n, quarter, emirate),
    "get_deteriorating_zones": lambda q, n, quarter, emirate: _deteriorating_listing_narration(q, n, quarter, emirate),
    "get_anomalous_zones": lambda q, n, quarter, emirate: _anomalous_listing_narration(q, n, quarter, emirate),
    "get_high_population_weak_zones": lambda q, n, quarter, emirate: _high_population_weak_listing_narration(n, quarter, emirate),
    "get_zones_by_peer_group": lambda q, n, quarter, emirate: _peer_group_listing_narration(q, n, quarter, emirate),
    "get_metric_threshold_zones": lambda q, n, quarter, emirate: _metric_threshold_listing_narration(q, n, quarter, emirate),
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
        if re.search(r"\bevidence\b", q):
            # Only reachable for 'full' evidence-tier zones -- the 'low'/'insufficient' tiers
            # both return their own, earlier message above before question text is ever inspected.
            return (f"Yes -- this area has enough public evidence to be scored and considered for "
                    f"the Priority list: {tool_result['tests']} tests from {tool_result['devices']} "
                    f"devices across {tool_result['quarters_observed']}/8 quarters "
                    f"({tool_result['confidence_score']}/100 confidence).")
        if re.search(r"\bmeasurements?\b.*(support|behind|recommend)", q):
            return (f"This recommendation is based on {tool_result['tests']} tests from "
                    f"{tool_result['devices']} devices across {tool_result['quarters_observed']}/8 "
                    f"quarters ({tool_result['confidence_score']}/100 confidence).")
        if re.search(r"\b(download|upload|latency|tests?|devices?)\b", q):
            return (f"For this area: download {tool_result['download_mbps']} Mbps, upload "
                    f"{tool_result['upload_mbps']} Mbps, latency {tool_result['latency_ms']} ms -- "
                    f"from {tool_result['tests']} tests on {tool_result['devices']} devices this quarter.")

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
        metric_asked = _exact_phrase_match(_normalize_geo_text(question), _METRIC_EXACT_ALIASES)
        if metric_asked:
            change_key = {"download_mbps": "download_mbps_change_qoq", "upload_mbps": "upload_mbps_change_qoq",
                          "latency_ms": "latency_ms_change_qoq"}[metric_asked]
            change = tool_result.get(change_key)
            metric_label = _METRIC_WORDS[metric_asked]
            unit = ct._METRIC_UNITS[metric_asked]
            if change is None:
                return f"Not enough scored quarters yet to compare {metric_label} to the previous quarter."
            # latency: lower is better, so a NEGATIVE change is an improvement; download/upload:
            # higher is better, so a POSITIVE change is an improvement -- same valence flip
            # _resolve_metric_query already applies for 'worst'/'best' wording, applied here to
            # phrase the raw delta in plain-language improve/worsen terms.
            improved = (change < 0) if metric_asked == "latency_ms" else (change > 0)
            direction = "improved" if improved else "worsened" if change != 0 else "stayed about the same"
            return (f"{metric_label.capitalize()} {direction} by {abs(change)} {unit} versus the previous "
                    f"scored quarter for this area.")
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

    if tool_name == "locate_area":
        if not tool_result or tool_result[0].get("error"):
            # Defensive only -- route() already resolves area/emirate ambiguity before this tool
            # is ever called in the normal flow, so a real caller should never see this.
            return AREA_NOT_FOUND_MESSAGE
        area_name, area_emirate = tool_result[0]["area_name"], tool_result[0]["emirate"]
        return f"Here is {area_name} in {area_emirate}. I highlighted it on the map."

    return UNMATCHED_MESSAGE


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
    greeting = check_greeting(question)
    if greeting:
        return {"question": question, "tool": None, "tool_args": None, "tool_result": None,
                "answer": greeting, "mode": "greeting"}

    refusal = check_refusal(question)
    if refusal:
        return {"question": question, "tool": None, "tool_args": None, "tool_result": None,
                "answer": refusal, "mode": "refusal"}

    data_source_answer = check_data_source_question(question)
    if data_source_answer:
        return {"question": question, "tool": None, "tool_args": None, "tool_result": None,
                "answer": data_source_answer, "mode": "fixed_fact"}

    tool_name, tool_args = route(question, zone_id)
    routing = {"path": "deterministic"}
    if tool_name == "unmatched":
        # Deterministic keyword routing found nothing -- try the semantic (LLM) classifier
        # before giving up. Never runs on, and never overrides, a confident deterministic match
        # above; only reached on the exact same "unmatched" path that used to dead-end here.
        semantic = classify_intent_semantic(question, zone_id)
        routing = {"path": "semantic", **{k: v for k, v in semantic.items() if k != "raw_model_output"}}
        if semantic["status"] == "ok":
            tool_name, tool_args = semantic["tool"], semantic["tool_args"]
        elif semantic["status"] == "clarify":
            return {"question": question, "tool": None, "tool_args": None, "tool_result": None,
                    "answer": f"Did you mean {semantic['suggestion']}?", "mode": "clarification",
                    "routing": routing}
        elif semantic["status"] == "locate_area_not_found":
            return {"question": question, "tool": None, "tool_args": None, "tool_result": None,
                    "answer": AREA_NOT_FOUND_MESSAGE, "mode": "area_not_found", "routing": routing}
        else:
            # 'none' (didn't validate / not one of ours), 'unavailable' (no LLM connected), or
            # 'error' (API call itself failed) -- all fall back to the same honest message a
            # deterministic miss already gives, never a guess.
            return {"question": question, "tool": None, "tool_args": None, "tool_result": None,
                    "answer": UNMATCHED_MESSAGE, "mode": "unmatched", "routing": routing}
    if tool_name == "locate_area_not_found":
        # A locate-style question was detected (e.g. "show me X") but nothing in the processed
        # area-name universe matched confidently enough -- an honest "not found," never a guess.
        return {"question": question, "tool": None, "tool_args": None, "tool_result": None,
                "answer": AREA_NOT_FOUND_MESSAGE, "mode": "area_not_found", "routing": routing}
    if tool_name == "clarify":
        # A metric/typo match was found but confidence was below _CONFIDENCE_THRESHOLD -- ask
        # rather than guess. No tool ran, so tool_result stays None; the map is untouched (the
        # frontend only ever moves the map from a tool's own zone_id field(s), and there isn't
        # one here).
        return {"question": question, "tool": None, "tool_args": None, "tool_result": None,
                "answer": f"Did you mean {tool_args['suggestion']}?", "mode": "clarification",
                "routing": routing}

    if quarter and "quarter" in _TOOLS[tool_name].__code__.co_varnames:
        tool_args = {**tool_args, "quarter": quarter}
    tool_args = {k: v for k, v in tool_args.items() if v is not None}

    tool_result = _TOOLS[tool_name](**tool_args)
    answer, mode = narrate(question, tool_name, tool_result, tool_args)
    result = {"question": question, "tool": tool_name, "tool_args": tool_args,
              "tool_result": tool_result, "answer": answer, "mode": mode, "routing": routing}
    if tool_name == "locate_area" and tool_result and not tool_result[0].get("error"):
        # Structured, top-level fields for a locate_area answer -- the frontend highlights from
        # h3_ids directly (never parsed from `answer`'s prose), matching the same "map is driven
        # by structured JSON, never narrated text" contract every other geo tool already follows.
        result["intent"] = "locate_area"
        result["area_name"] = tool_result[0]["area_name"]
        result["emirate"] = tool_result[0]["emirate"]
        result["h3_ids"] = [z["zone_id"] for z in tool_result]
    return result


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
    "Write a 3-4 sentence zone brief for a decision-maker, using ONLY the facts in the JSON "
    "provided. Open by naming the area using its area_name and emirate fields exactly as given "
    "(e.g. 'Muwaileh, Sharjah') -- never a raw H3 cell id. Cover: how this zone compares to its "
    "peer group, its confidence level, its population exposure, why it received its Priority "
    "Score (name the High-banded factors), and what to investigate next. End with one sentence "
    "noting this is public data and cannot identify an operator-specific root cause. Never "
    "invent a number, or an area name, not present in the JSON."
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
            f"{context['area_name']}, {context['emirate']} shows {peer_summary} {confidence_summary} "
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
        f"{context['area_name']}, {context['emirate']} shows {peer_summary} {confidence_summary} "
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
