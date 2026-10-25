"""Effect of the key-concept check (generation/coverage.py) on stored contexts, without generating
(thesis section "Подобряване на резултатите -- проверка дали ключовите понятия от въпроса се срещат в откъсите").

    python -m evaluation.coverage_eval

The check only decides whether to generate, so it is evaluated on the chunks the system already retrieved:
the 24 development questions (results/thesis/dev_contexts_hybrid.json) and the 69 questions of the final
evaluation run before the check existed (results/thesis/final_before_coverage/final_answers.json). Prints the
refusals per set, McNemar's test for the out-of-scope questions and the sensitivity to COVERAGE_MAX_CHUNKS.
"""
from evaluation.common import CHUNKS, DEV, chunks_by_id, load
from evaluation.significance import mcnemar, wilson
from medguiderag.config import COVERAGE_MAX_CHUNKS
from medguiderag.generation.coverage import CoverageChecker

BEFORE = "thesis/final_before_coverage/final_answers.json"


def main():
    dev_contexts = load("thesis/dev_contexts_hybrid.json")
    final = load(BEFORE)
    sets = {
        "development set (in scope)":
        [(d["question"], chunks_by_id(dev_contexts[d["question"]]))
         for d in DEV],
        "final set, in scope": [(r["question"], chunks_by_id(r["chunk_ids"]))
                                for r in final if r["id"].startswith("F")],
        "final set, out of scope":
        [(r["question"], chunks_by_id(r["chunk_ids"])) for r in final
         if r["id"].startswith("O")],
    }
    texts = [c["text"] for c in CHUNKS]
    checker = CoverageChecker(texts)
    print(
        f"specific word: stem in at most {COVERAGE_MAX_CHUNKS} of {len(CHUNKS)} chunks"
    )
    for name, items in sets.items():
        results = [checker.check(q, ch) for q, ch in items]
        stopped = [(q, c) for (q, _), c in zip(items, results)
                   if not c.answerable]
        low, high = wilson(len(stopped), len(items))
        print(
            f"\n{name}: stopped {len(stopped)}/{len(items)} (95% CI {low:.0%}-{high:.0%})"
        )
        for q, c in stopped:
            print(f"  {c.reason:6} {', '.join(c.terms):30} | {q}")

    # Out of scope: before the check none was refused (all passed the reranker threshold).
    out = sets["final set, out of scope"]
    after = [not checker.check(q, ch).answerable for q, ch in out]
    only_before, only_after, p = mcnemar([False] * len(out), after)
    print(
        f"\nout of scope, refused before -> after: 0 -> {sum(after)} of {len(out)}; McNemar exact p = {p:.4f}"
    )

    print(
        "\nsensitivity (stopped: development / final in scope / final out of scope):"
    )
    for limit in (5, 10, 20, 30, 40, 60, 80):
        c = CoverageChecker(texts, max_chunks=limit)
        counts = [
            sum(not c.check(q, ch).answerable for q, ch in items)
            for items in sets.values()
        ]
        print(f"  {limit:3} chunks: " + " / ".join(map(str, counts)))


if __name__ == "__main__":
    main()
