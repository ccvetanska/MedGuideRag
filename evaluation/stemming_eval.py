"""Word normalisation for BM25: the 6-character prefix of retrieval/bm25.py vs. other prefix lengths and the
BulStem stemmer (thesis paragraphs "Лексикално търсене", section 4.3, and "Извличане", section 4.7).
"""
import re

import numpy as np
from bulstem.stem import BulStemmer
from rank_bm25 import BM25Okapi

from evaluation.common import CHUNKS, GOLD, QUESTIONS, load, save
from evaluation.final_eval import QUESTIONS_PATH, inspect, load_questions
from evaluation.significance import mcnemar, randomization
from medguiderag.retrieval.bm25 import STEM_LENGTH
from medguiderag.retrieval.hybrid import FUSION_DEPTH, reciprocal_rank_fusion

BASE = f"prefix-{STEM_LENGTH}"
# Word pairs the thesis gives as examples of what each normalisation conflates.
EXAMPLES = [("лечение", "лечения"), ("доза", "дозата"),
            ("инсулин", "инсулинова"), ("бременност", "бременни"),
            ("противопоказания", "противогъбични")]


def _prefix(n: int):
    return lambda w: w[:n]


# BulStem with its three rule sets (1-3 letters of left context before the removed ending), package defaults.
VARIANTS = {
    "whole words": lambda w: w,
    **{
        f"prefix-{n}": _prefix(n)
        for n in (5, 6, 7)
    },
    **{
        f"BulStem context {c}":
        BulStemmer.from_file(f"stem-context-{c}", min_freq=2, left_context=c).stem
        for c in (1, 2, 3)
    }
}


class Bm25:

    def __init__(self, normalise):
        self._normalise = normalise
        self._model = BM25Okapi([self._tokenize(c["text"]) for c in CHUNKS])

    def _tokenize(self, text: str) -> list[str]:
        return [self._normalise(w) for w in re.findall(r"\w+", text.lower())]

    def search(self, query: str) -> list[int]:
        return np.argsort(-self._model.get_scores(self._tokenize(query)),
                          kind="stable")[:FUSION_DEPTH].tolist()


def _rank(ranking: list[int], gold: list[int], k: int) -> int | None:
    return next((pos for pos, idx in enumerate(ranking[:k]) if idx in gold),
                None)


def scores(rankings: list[list[int]],
           golds: list[list[int]]) -> dict[str, list]:
    """Per question: hits at 1, 5 and 20 and the reciprocal rank within the top 10 (strict)."""
    out = {
        f"R@{k}":
        [_rank(r, g, k) is not None for r, g in zip(rankings, golds)]
        for k in (1, 5, 20)
    }
    out["MRR"] = [
        1 / (p + 1) if (p := _rank(r, g, 10)) is not None else 0.0
        for r, g in zip(rankings, golds)
    ]
    return out


def report(title: str, results: dict[str, dict[str, list]]):
    print(
        f"\n{title}\n{'variant':20} |  R@1   R@5  R@20   MRR | vs. {BASE}: only {BASE} / only variant, p"
    )
    base = results[BASE]
    for name, s in results.items():
        line = f"{name:20} | " + " ".join(
            f"{sum(s[m]) / len(s[m]):5.2f}"
            for m in ("R@1", "R@5", "R@20", "MRR"))
        if name != BASE:
            tests = [
                f"R@{k} {a}/{b} {p:.2f}" for k in (1, 5, 20)
                for a, b, p in [mcnemar(base[f"R@{k}"], s[f"R@{k}"])]
            ]
            line += " | " + ", ".join(
                tests) + f", MRR {randomization(base['MRR'], s['MRR']):.2f}"
        print(line)


def main():
    models = {name: Bm25(fn) for name, fn in VARIANTS.items()}

    dev_dense = load("thesis/retrieval_rankings.json")["BAAI/bge-m3"]
    rows = load("thesis/final_answers.json")
    _, _, gold_of = inspect(load_questions(QUESTIONS_PATH))
    final = [r for r in rows if gold_of.get(r["id"])]
    sets = {
        "development set": (QUESTIONS, [[g] for g in GOLD], dev_dense),
        "final set, in scope":
        ([r["question"] for r in final], [gold_of[r["id"]] for r in final],
         [r["rankings"]["bge-m3"] for r in final])
    }

    archived = {
        "development set":
        load("thesis/retrieval_rankings.json")["bm25-stem6"],
        "final set, in scope": [r["rankings"]["BM25"] for r in final]
    }
    rankings = {}
    for set_name, (questions, golds, dense) in sets.items():
        bm25 = {
            name: [m.search(q) for q in questions]
            for name, m in models.items()
        }
        # The system's prefix reproduces the archived BM25 rankings.
        assert bm25[BASE] == archived[
            set_name], f"{BASE} differs from the archived BM25 ({set_name})"
        fused = {
            name: [
                reciprocal_rank_fusion([d, b])[:FUSION_DEPTH]
                for d, b in zip(dense, ranks)
            ]
            for name, ranks in bm25.items()
        }
        rankings[set_name] = {"bm25": bm25, "rrf": fused}
        report(f"{set_name} ({len(questions)} questions): BM25 alone", {
            name: scores(r, golds)
            for name, r in bm25.items()
        })
        report(
            f"{set_name} ({len(questions)} questions): RRF(bge-m3, BM25); R@20 = candidates of the reranker",
            {
                name: scores(r, golds)
                for name, r in fused.items()
            })

    print("\nexamples:")
    for pair in EXAMPLES:
        print("  " + " / ".join(pair) + ": " + "; ".join(
            f"{name} {' / '.join(fn(w) for w in pair)}"
            for name, fn in VARIANTS.items() if name != "whole words"))
    print("saved", save("stemming_rankings.json", rankings))


if __name__ == "__main__":
    main()
