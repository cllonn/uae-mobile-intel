# UAE Mobile Network Experience Intelligence — Methodology Summary

*For tomorrow's mentor walkthrough. Every number below is live from the current pipeline.*

### Data

Three public datasets, no e& data anywhere: **Ookla Speedtest** (crowdsourced mobile speed
tests, 8 quarters), **WorldPop** (population estimates), **OpenStreetMap** (building/road/POI
density, used to group similar areas together, and the real emirate boundaries).

### H3 geographic zones

We aggregate individual measurements into hexagonal zones (H3 resolution 7, ~5 km² each) —
chosen empirically by testing which resolution gives stable, well-supported scores.

### Reliability rule

**A zone-quarter needs at least 30 measurements (tests) to be classified at all.** Below that,
we say **"insufficient public evidence"** — never a low score, because we genuinely don't know.
Today (2026Q2): 1,815 zones have any measurement; **314 clear the 30-test bar**; those 314
represent 37.6% of the national population.

### Experience Index

A deterministic 0–100 score from download speed, upload speed, and latency. Same formula,
every time — no AI involved in calculating it.

### Confidence Score

A separate 0–100 number representing how much evidence backs the Experience Index — from test
count, device count, and how many quarters we've seen data for. A zone can pass the 30-test
reliability bar and still have a lower confidence than a heavily-tested one.

### Peer groups

We compare each zone only against similar zones — commercial/urban-core, low-density
residential, industrial, or rural/edge — grouped by building density, population density, and
road/POI density. Never compared blindly against the whole country.

### Peer Gap

How far a zone's Experience Index sits below (or above) its peer group's median, in the same
quarter. 2026Q2 peer medians: commercial/urban-core 43.9, low-density residential 43.4,
industrial 30.4, rural/edge 17.7 — real, meaningful differences.

### Trend / deterioration

A zone is flagged as deteriorating only after 3 consecutive quarters of falling behind its
peers — not one bad quarter. 39 zones hit this pattern at some point in the last 2 years; 10
are currently in it.

### ML anomaly detection

An Isolation Forest model flags zones that look statistically unusual — either against their
peers right now, or against their own history. This is genuine machine learning, not a formula.

### ML validation — the honest result

We built a simple rule (flag the bottom 10% by download speed) as a baseline, then tested both
against synthetic, injected faults on data the model never saw. **The simple rule won** —
higher precision and recall than the ML model on every measure. We report that honestly rather
than hide it, and keep the ML as a *supplementary* signal (20% weight in the final ranking)
rather than the main evidence. We also found *why*: the ML barely catches a pure download-speed
drop (1.6% recall) but does much better when multiple things go wrong at once (58% recall) —
useful to know, and something a future iteration could specifically improve.

### Population

How many people live in a zone (from WorldPop) — used so investigation priority reflects human
impact, not just technical severity.

### Priority Score

Combines four things — peer gap, ML anomaly, deterioration, and population — into one 0–100
ranking of where to investigate first. Confidence is *multiplied* in, not added, so a
low-evidence zone can never rank as high priority no matter how bad it looks. We tested this by
nudging the weights ±15%: the ranking barely moved (Spearman correlation 0.996) — it's not a
fragile, arbitrary choice.

### Limitation

This is public, outside-in data with no operator attribution. It **cannot** identify e&,
another operator, a specific site, a tower, or an internal root cause — only where public
evidence suggests investigation would be valuable.

### Next phase

Generative AI will sit **above** these analytics, not replace them: it will explain
already-computed results in plain language, routed through a fixed set of lookup functions, and
will refuse any question asking it to attribute a cause it cannot support. It will never
calculate a score itself. This phase starts after mentor confirmation of the methodology above.
