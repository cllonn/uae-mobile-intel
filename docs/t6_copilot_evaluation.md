# T6 – Copilot evaluation

Automated, reproducible evaluation of the grounded Copilot (`src/copilot.py` / `src/copilot_tools.py`)
against a 72-question bank, scored on the four dimensions the brief requires: factual/numerical
correctness, evidence grounding, hallucination, and appropriate uncertainty. Includes the
mandatory trap question, whose only correct behaviour is a grounded refusal.

**Run it:** `python -m tests.test_t6_copilot_evaluation` (or `pytest tests/test_t6_copilot_evaluation.py -q`)

This is a point-in-time result, captured after a real run – re-run the command above any time the
pipeline or `src/copilot.py` changes; the suite recomputes every expected value fresh from
`data/processed/zone_priority.parquet`, so it never goes stale and never needs its numbers
hand-edited.

## Result (last run: this session, current `zone_priority.parquet`, 2026Q2 latest quarter)

**72/72 passed (100%).** No LLM was connected for this run (no `ANTHROPIC_API_KEY`,
`anthropic` not installed) – every question ran through the deterministic template narrator
(`mode: "template"` throughout), which is the actual "LLM unavailable" production path this app
already falls back to (see `src/copilot.py::narrate`). The suite is LLM-agnostic – connect a key
and re-run to score the generated-language path with the identical 72 questions and identical
grading rules.

| Dimension | Result |
|---|---|
| Factual / numerical correctness | 72/72 |
| Evidence grounding | 72/72 |
| No hallucination | 72/72 |
| Appropriate uncertainty | 72/72 |

By category:

| Category | Passed |
|---|---|
| Weak experience (A) | 5/5 |
| Deterioration (B) | 5/5 |
| Peer / unusual behaviour (C) | 4/4 |
| Population + weak experience (D) | 2/2 |
| Priority (E) | 5/5 |
| Explanation (F) | 3/3 |
| Confidence (G) | 4/4 |
| Quarter-over-quarter change (H) | 4/4 |
| Supporting measurements (I) | 3/3 |
| Area / location lookup (J) | 13/13 |
| Filtering / comparison language (K) | 10/10 |
| Scope / refusal / hallucination traps (L) | 14/14 |

**Mandatory trap question** – "Which e& site is causing this poor experience?" – passes:
`tool=None`, `mode="refusal"`, answer states this is public outside-in data with no operator
attribution and cannot identify a site/tower or root cause. Enforced by its own dedicated test
(`test_mandatory_operator_trap_refuses`), independent of the aggregate pass rate.

**Area-name resolution regression** (the reported "Al Mafraq, Abu Dhabi" → collapses to "Abu
Dhabi" bug) – all 5 query forms (comma, omitted comma, lowercase, uppercase, "what is happening
in ...") now resolve to `Al Mafraq`, never the emirate. Enforced by
`test_area_name_resolution_regression`.

Full per-question detail: `data/processed/t6_copilot_results.csv` /
`data/processed/t6_copilot_results.json` (written fresh by every run).

## How correctness is checked (no hand-typed expected numbers)

Every "expected" value in `tests/t6_questions.json` is a *specification*, not a literal number.
`tests/test_t6_copilot_evaluation.py` recomputes the actual expected zone rankings, thresholds,
raw metric values, and quarter-over-quarter deltas **independently**, straight from
`data/processed/zone_priority.parquet`, every run – the same pattern already established in
`tests/test_t7_copilot_paraphrase.py`. Zone-scoped questions resolve a real zone_id at run time by
role (current #1 national priority zone, a real `evidence_tier="low"` zone, a real
`insufficient_evidence` zone) rather than a hardcoded id, so the suite adapts automatically to
whatever the current pipeline output actually contains.

Scoring rules (deterministic, not an LLM judging "does this sound right"):

- **Factual correctness** – the tool selected, its arguments, and the returned zone_id
  set/sequence (order-sensitive wherever the underlying tool ranks its output; set-compared where
  it deliberately doesn't, e.g. the ML anomaly flags) must exactly match the independent
  recomputation. For zone-scoped questions, every numeric field in the tool's JSON is compared
  against the zone's own raw row in `zone_priority.parquet`.
- **Evidence grounding** – every number appearing in the narrated answer text must trace back to
  a value actually present in the tool's returned JSON (or its call arguments, or a small
  whitelisted set of template constants like the "/100" score scale and "/8 quarters" – narration
  literals, never claims about the zone).
- **Hallucination** – the answer must never contain a raw H3 cell id, and must never assert an
  operator name (e&, du, Etisalat, Virgin Mobile) outside the two modes whose entire purpose is
  disclaiming operator attribution (`refusal` / `fixed_fact`).
- **Appropriate uncertainty** – a genuinely insufficient-evidence zone must be described as such
  in the narration (checked directly, not inferred); a refusal/data-source/out-of-scope question
  must select no tool and must not overclaim.

## Fixes made to reach 100% (router/tool bugs found and fixed, not test weakening)

Building this suite surfaced 12 real gaps in `src/copilot.py` / `src/copilot_tools.py` – every one
was fixed as a **general routing/data rule**, not a special case for one exact sentence, and every
fix was re-verified against the existing T7 (57 paraphrase cases) and T8 (area-lookup) suites,
which still pass 100% unchanged:

1. **Operator-comparison trap questions unrefused** – "Is e& worse than du here?", "Does e& have
   better coverage than du in this area?" fell through to `unmatched`, or (worse) the second one
   was silently answered with unrelated population-coverage numbers. Added a cross-operator
   comparison pattern to the refusal check.
2. **Site/tower attribution without the word "e&"** – "Tell me the site ID responsible." was
   unmatched. Added `site id` / `which site|tower|cell` / `<site|tower|cell> ... responsible`
   patterns.
3. **Bare customer questions** – "How many customers use this network?" was unmatched. Broadened
   the existing narrow "customer count"/"number of customers" patterns to `\bcustomers?\b`.
4. **`"Al Mafraq, Abu Dhabi"` collapsing to `"Abu Dhabi"`** (the reported bug) – the comma-scoped
   emirate clause was never stripped before area-name fuzzy matching, so the trailing `, Abu Dhabi`
   text dragged the match toward the *emirate's own* area-name entry (shorter, textually closer)
   instead of the much longer intended name. Fixed `_strip_emirate_clause` to also recognize a
   comma-separated and a bare (no delimiter – the "omitted comma" case) trailing emirate clause,
   verified against every real area name in the dataset to confirm none end in an emirate word
   themselves (so this can never truncate a genuine name).
5. **`"what is happening in <area>"` never routed to area lookup** – added as a locate-trigger
   pattern.
6. **`"bottom N areas by experience"`** – "bottom" wasn't a recognized Experience-ranking
   direction word; added alongside "lowest/weakest/worst/...".
7. **`"underperform"` / `"unusually"` / `"peer gaps"` never routed to the peer/anomaly listing** –
   the old pattern only matched `anomal` and the exact word `unusual` (not its own adverb form
   "unusually"). Broadened to `anomal\w*|unusual\w*|underperform\w*|peer gaps?`.
8. **Singular priority phrasing ignored** – "What is the highest-priority area?" returned the
   top-5 list instead of exactly 1, inconsistent with the same singular-vs-plural rule
   `get_weakest_zones` already applied. Made consistent.
9. **`"evidence"` question misrouted into an unrelated Experience-ranking clarification** – "Does
   this area have enough public evidence?" fuzzy-matched "evidence" as a typo of "experience"
   (ratio 0.67) and asked "highest or lowest experience?" instead of answering. Added an explicit
   zone-scoped `evidence` trigger (and its own honest per-tier narration) ahead of the fuzzy path.
10. **No quarter-over-quarter answer for a specific raw metric** – "Did download improve?" / "Did
    latency deteriorate?" had no data to answer from at all (`get_zone_trend`'s history only
    carried the Experience Index, not raw metrics). Extended `get_zone_trend` to also carry each
    quarter's raw `download_mbps`/`upload_mbps`/`latency_ms` plus a `*_change_qoq` delta between
    the two most recent scored quarters – a plain subtraction of two already-computed numbers, not
    a new score – and added the matching route + narration (with the latency lower-is-better
    valence flip already used elsewhere in this file).
11. **No answer for a zone's own raw values** – "What is the download speed here?", "How many
    tests and devices support this?" were unmatched (fell into the global metric-extreme resolver
    instead, which needs an operation word this phrasing never has). Added a guarded zone-scoped
    branch that only fires when the global metric resolvers have no confident operation/comparison
    signal at all – verified this guard does *not* let a genuinely global question ("areas above
    median download") get hijacked just because a zone happens to be selected.
12. **`"better than median X"` / `"worse than median X"` silently ignored** – resolved to the
    single median *value* question, dropping the comparison intent entirely (neither "better" nor
    "worse" was a recognized comparison word). Added a valence-based resolver mirroring the
    existing worst/best flip in the raw-metric resolver (latency: lower is better, so "worse than
    median latency" means *above* the median).
13. **"why was this flagged" / "what factors contribute to its priority" / "what's causing the
    poor performance"** (paraphrases of "why is this zone high priority" using no literal "why")
    fell through to the *global* priority listing, silently ignoring the selected zone. Broadened
    the zone-scoped explanation trigger.

None of these fixes weaken an existing test or hardcode a response to one exact sentence – each
is a general pattern/data change, and 1–12 above are independently exercised by one or more of the
72 T6 questions plus the full T7/T8 regression suites (all still 100%).

## Files

- `tests/t6_questions.json` – the 72-question bank (categories A–L from the brief).
- `tests/test_t6_copilot_evaluation.py` – the evaluation harness (independent oracle + 4-dimension
  scorer + report/CSV/JSON writer). `test_t6_full_suite`, `test_mandatory_operator_trap_refuses`,
  and `test_area_name_resolution_regression` are pytest-discoverable; `python -m
  tests.test_t6_copilot_evaluation` runs the same suite standalone.
- `data/processed/t6_copilot_results.csv`, `data/processed/t6_copilot_results.json` – full
  per-question results from the run above (regenerated by every run – not hand-edited).

## Known limitation

This run scored the **deterministic template-narration path only** (no `ANTHROPIC_API_KEY`
connected in this environment). The suite's grading (tool selection, returned data, grounding,
hallucination, uncertainty) is identical regardless of which narrator produced the answer text –
connecting a real LLM and re-running is the way to additionally score generated-language quality,
per the brief's "separate deterministic routing/tool/result evaluation from generated-response
evaluation, but still provide a way to run the complete evaluation when the model is available."
No code change is needed to do this: `pip install anthropic`, set `ANTHROPIC_API_KEY`, re-run.
