# T7 — Zone-brief audit & usability: live test protocol

Two separate checks per the brief. Neither can be simulated — both need a real run.

## Part A — live outsider test (run this before/at the demo)

Hand this to someone who has never seen the dashboard, with a link to
`data/processed/uae_dashboard.html`, a stopwatch, and nothing else:

1. Say only: **"Find the weakest-performing zone in [their home emirate] and tell me what's
   wrong with it."** Do not explain the UI.
2. Start the timer. Stop it the moment they can state (a) roughly where the weakest zone is,
   and (b) one real number from the zone panel that explains why it's weak (download,
   Experience Index, or "not enough data").
3. Record: time taken, whether they succeeded unassisted, where they got stuck if anywhere.

**What's already in place to support this** (built since the last audit found this gap): an
**Emirate** dropdown now filters the map, the priority list, and the KPI strip to one emirate;
switching to the Experience layer and sorting by color should make the weakest zone in that
emirate visually obvious without needing to know UAE geography by eye.

**Target**: under 2 minutes, unassisted. Report the actual result either way — a miss here is
useful evidence for the demo script ("we found this gap ourselves and I can show you the fix"),
not something to hide.

## Part B — zone-brief fabrication check

Now that `src/copilot.py::generate_zone_brief()` exists: generate a brief for at least the top 5
priority zones (`python -c "from src import copilot; [print(copilot.generate_zone_brief(z['zone_id'])['brief'], '\n') for z in __import__('src.copilot_tools', fromlist=['x']).get_top_priority_zones(5)]"`
or just call it per zone from a notebook), and have someone **not on the AI/ML workstream**
read each one against the zone's raw row in `zone_priority.parquet`. Flag any number in the
brief that cannot be found in that row.

Every number in the template-mode brief is pulled directly from
`copilot.generate_zone_context()`, which itself calls `copilot_tools.get_zone_details` /
`get_zone_peer_comparison` / `get_zone_trend` — there is no step where a number could be
invented, but this check is what actually confirms that, rather than assuming it from the code.

## What to record

| | Result |
|---|---|
| Part A: time taken | — |
| Part A: succeeded unassisted (Y/N) | — |
| Part A: where they got stuck (if anywhere) | — |
| Part B: briefs checked | — |
| Part B: fabricated numbers found | — |

Fill this in after running it — it is intentionally blank, not pre-filled, so the result is real.
