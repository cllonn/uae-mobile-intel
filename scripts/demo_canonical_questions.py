"""
Runs the 14 canonical/test questions (the challenge brief's 10 + 4 additional checks the team
added: two emirate-scoped listings and the mandatory e&-attribution trap) through the copilot,
printing the selected tool, its structured result, and the final narrated answer for each.

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
    "Why was this zone given high priority?",          # zone-scoped
    "How confident are we in this zone?",               # zone-scoped
    "What changed since the previous quarter?",         # zone-scoped
    "Which measurements support this recommendation?",  # zone-scoped
    "Show the weakest zones in Dubai.",
    "Show the highest-priority zones in Sharjah.",
    "Is this e& network data?",
    "Which e& site is causing this poor experience?",
]

ZONE_SCOPED_INDEXES = {6, 7, 8, 9}  # 0-indexed positions of the four zone-scoped questions above


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    demo_zone_id = ct.get_top_priority_zones(1)[0]["zone_id"]
    print(f"(Zone-scoped questions below use zone {demo_zone_id} -- the #1 national priority zone this quarter)\n")

    for i, q in enumerate(QUESTIONS):
        zone_id = demo_zone_id if i in ZONE_SCOPED_INDEXES else None
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
