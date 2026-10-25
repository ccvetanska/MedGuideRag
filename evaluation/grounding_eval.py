"""Run the grounding check (generation/grounding.py) over saved answers and list every flagged detail.

    python -m evaluation.grounding_eval                   # the 132 answers of the thesis evaluation
    python -m evaluation.grounding_eval <answers.json>    # other saved answers

Each answer file is a list of rows with "question", "answer" and "chunk_ids" (the chunks the prompt showed).
For the thesis set, the manual verdicts on the flags are in results/thesis/grounding_flags_labelled.json.
"""
import argparse
from collections import Counter

from evaluation.common import CHUNKS, THESIS_RESULTS_DIR, chunks_by_id, load, save
from medguiderag.generation.grounding import GroundingChecker

THESIS_SETS = [
    "thesis/generation_penalty_sweep.json",
    "thesis/generation_loop_check.json",
    "thesis/generation_models_iteration2.json"
]


def describe(row: dict) -> str:
    return row.get(
        "set"
    ) or f"{row.get('generator', '')} penalty {row.get('repeat_penalty', '')} run {row.get('run', '')}"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="*", default=THESIS_SETS)
    args = parser.parse_args()

    checker = GroundingChecker([c["text"] for c in CHUNKS])
    flags, answers, flagged, kinds = [], 0, 0, Counter()
    for path in args.files:
        for row in load(path):
            found = checker.check(row["answer"],
                                  chunks_by_id(row["chunk_ids"]),
                                  row["question"])
            issues = sorted({(i.kind, i.text, i.reason) for i in found})
            answers += 1
            flagged += bool(issues)
            kinds.update(kind for kind, _, _ in issues)
            if issues:
                details = "; ".join(f"{k}: {t} ({why})"
                                    for k, t, why in issues)
                print(
                    f"{describe(row)[:28]:28} | {row['question'][:42]:42} | {details}"
                )
            flags += [{
                "set": describe(row),
                "question": row["question"],
                "kind": k,
                "text": t,
                "reason": why
            } for k, t, why in issues]
    print(
        f"\n{answers} answers, {flagged} with flagged details, {len(flags)} details by kind: {dict(kinds)}"
    )
    labelled = THESIS_RESULTS_DIR / "grounding_flags_labelled.json"
    if args.files == THESIS_SETS and labelled.exists():
        real = sum(f["real_error"] for f in load(labelled))
        print(
            f"manual verdicts on the thesis set: {real} of {len(load(labelled))} flagged details are real errors"
        )
    print("saved", save("grounding_flags.json", flags))


if __name__ == "__main__":
    main()
