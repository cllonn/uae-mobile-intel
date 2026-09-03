"""
Runs the 15 canonical/test questions (the challenge brief's 10 + 5 additional checks the team
added: two emirate-scoped listings, a low-confidence zone case, and the mandatory
e&-attribution trap) through the copilot, printing the selected tool, its structured result,
and the final narrated answer for each.

This is both the Task 7 deliverable (show tool selection -> tool output -> final answer for
every canonical question) and the fastest way to sanity-check the whole
copilot_tools.py -> copilot.py chain end to end before a live demo.

Run: python -m scripts.demo_canonical_questions
"""
import sys

from src import copilot, copilot_tools as ct

QUESTIONS = [
    "Where does mobile experience appear weakest?",
    "Which five areas should we investigate first?",
    "Which three areas should we investigate first?",
    "Which areas are deteriorating?",
    "Which areas are anomalous relative to their peers?",
    "Where do weak experience and high population occur together?",
    "Why was this zone given high priority?",          # zone-scoped, full-tier zone
    "How confident are we in this zone?",               # zone-scoped, full-tier zone
    "What changed since the previous quarter?",         # zone-scoped, full-tier zone
    "Which measurements support this recommendation?",  # zone-scoped, full-tier zone
    "Show the weakest zones in Dubai.",
    "Show the highest-priority zones in Sharjah.",
    "Is this e& network data?",
    "Which e& site is causing this poor experience?",
    "Why was this zone given high priority?",           # zone-scoped, LOW-confidence zone
]

ZONE_SCOPED_INDEXES = {6, 7, 8, 9}  # 0-indexed positions of the four full-tier zone-scoped questions above
LOW_TIER_ZONE_INDEX = 14  # the low-confidence case appended at the end


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    demo_zone_id = ct.get_top_priority_zones(1)[0]["zone_id"]
    print(f"(Zone-scoped questions below use zone {demo_zone_id} -- the #1 national priority zone this quarter)\n")

    df = ct._load()
    latest_q = sorted(df["quarter"].unique())[-1]
    low_tier_rows = df[(df["quarter"] == latest_q) & (df["evidence_tier"] == "low")]
    demo_low_zone_id = low_tier_rows.iloc[0]["h3_cell"] if not low_tier_rows.empty else None
    print(f"(Q{LOW_TIER_ZONE_INDEX + 1} below uses zone {demo_low_zone_id} -- a real 'low' evidence-tier "
          f"zone this quarter (10-29 tests): scored, but not Priority-shortlist-eligible)\n")

    for i, q in enumerate(QUESTIONS):
        if i == LOW_TIER_ZONE_INDEX:
            zone_id = demo_low_zone_id
        elif i in ZONE_SCOPED_INDEXES:
            zone_id = demo_zone_id
        else:
            zone_id = None
        result = copilot.answer_question(q, zone_id=zone_id)
        print("=" * 90)
        print(f"Q{i+1}: {q}")
        print(f"  Tool selected : {result['tool']}  args={result['tool_args']}")
        tr = result["tool_result"]
        if isinstance(tr, list):
            print(f"  Tool output   : {len(tr)} zone(s), e.g. {tr[0] if tr else None}")
        else:
            print(f"  Tool output   : {tr}")
        print(f"  Mode          : {result['mode']}")
        print(f"  Answer        : {result['answer']}")
    print("=" * 90)


if __name__ == "__main__":
    main()
