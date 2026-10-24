"""Statistical significance of the comparisons reported in the thesis, from the archived results.

    python -m evaluation.significance

- Retrieval, paired by question: exact two-sided McNemar test on Recall@1 and Recall@5 hits, and a paired
  randomization (sign-flip) test on reciprocal ranks for MRR.
- Facts in generated answers, paired by fact: exact McNemar test (facts of one answer are not independent,
  so these p-values are, if anything, too optimistic).
- Shares from manual verdicts: Wilson 95% confidence intervals.
- Questions needed for 80% power at alpha 0.05 to confirm the observed Recall@5 gain (Connor's formula).
"""
import math
import random
import re

from evaluation.common import FACTS, GOLD, load

SAMPLES = 100_000
Z_ALPHA, Z_BETA = 1.959964, 0.841621  # two-sided alpha 0.05, power 0.80


def mcnemar(a: list[bool], b: list[bool]) -> tuple[int, int, float]:
    """Cases only a gets right, cases only b gets right, exact two-sided p-value."""
    only_a = sum(x and not y for x, y in zip(a, b))
    only_b = sum(y and not x for x, y in zip(a, b))
    n, k = only_a + only_b, min(only_a, only_b)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return only_a, only_b, p


def randomization(a: list[float], b: list[float], seed: int = 0) -> float:
    """Two-sided paired sign-flip test on the mean difference."""
    diffs = [y - x for x, y in zip(a, b)]
    observed = abs(sum(diffs))
    rng = random.Random(seed)
    hits = sum(abs(sum(d if rng.random() < 0.5 else -d for d in diffs)) >= observed - 1e-12 for _ in range(SAMPLES))
    return (hits + 1) / (SAMPLES + 1)


def wilson(successes: int, n: int) -> tuple[float, float]:
    p, z = successes / n, Z_ALPHA
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def questions_needed(p_only_a: float, p_only_b: float) -> int:
    """Connor (1987): pairs needed for a McNemar test to detect the given discordance rates."""
    pd, delta = p_only_a + p_only_b, abs(p_only_b - p_only_a)
    return math.ceil((Z_ALPHA * math.sqrt(pd) + Z_BETA * math.sqrt(pd - delta ** 2)) ** 2 / delta ** 2)


def hits(ranking: list[list[int]], k: int) -> list[bool]:
    return [g in r[:k] for g, r in zip(GOLD, ranking)]


def reciprocal_ranks(ranking: list[list[int]]) -> list[float]:
    return [1 / (r[:10].index(g) + 1) if g in r[:10] else 0.0 for g, r in zip(GOLD, ranking)]


def main():
    runs = load("thesis/retrieval_rankings.json")
    runs.update(load("thesis/retrieval_rankings_ollama.json"))
    runs.update(load("thesis/representation_variants.json"))
    runs["rerank"] = load("thesis/rerank_rankings_cpu.json")
    comparisons = [
        ("BAAI/bge-m3", "rerank", "bge-m3 -> fusion + reranker"),
        ("rrf(bge-m3, bm25-stem6)", "rerank", "fusion -> fusion + reranker"),
        ("BAAI/bge-m3", "rrf(bge-m3, bm25-stem6)", "bge-m3 -> fusion"),
        ("bm25", "bm25-stem6", "BM25 -> BM25 with stems"),
        ("BAAI/bge-m3", "intfloat/multilingual-e5-base", "bge-m3 -> e5-base"),
        ("BAAI/bge-m3", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2", "bge-m3 -> mpnet"),
        ("BAAI/bge-m3", "ollama qwen3-embedding:0.6b", "bge-m3 -> qwen3-embedding"),
        ("BAAI/bge-m3", "ollama embeddinggemma", "bge-m3 -> embeddinggemma"),
    ]
    # Every representation variant of bge-m3 (thesis paragraph "Варианти на представянето").
    comparisons += [("BAAI/bge-m3", key, f"bge-m3 -> {key.removeprefix('bge-m3 ')}")
                    for key in load("thesis/representation_variants.json")]
    print(f"{'retrieval, 24 questions':36} | R@1 only A/only B   p | R@5 only A/only B   p | MRR  A -> B     p")
    for a, b, name in comparisons:
        r1 = mcnemar(hits(runs[a], 1), hits(runs[b], 1))
        r5 = mcnemar(hits(runs[a], 5), hits(runs[b], 5))
        rr_a, rr_b = reciprocal_ranks(runs[a]), reciprocal_ranks(runs[b])
        print(f"{name:36} |        {r1[0]:2} / {r1[1]:2}    {r1[2]:.3f} |        {r5[0]:2} / {r5[1]:2}    {r5[2]:.3f} | "
              f"{sum(rr_a) / 24:.2f} -> {sum(rr_b) / 24:.2f}  {randomization(rr_a, rr_b):.3f}")

    print("\nfacts in answers, paired by fact (control questions):")
    for path, key, a, b, label in [
        ("thesis/generation_models_iteration3.json", "generator", "2.6B", "4B", "2.6B -> 4B (current pipeline, run 1)"),
        ("thesis/generation_penalty_sweep.json", "repeat_penalty", 1.0, 1.2, "repeat penalty 1.0 -> 1.2 (2.6B)"),
    ]:
        rows = [r for r in load(path) if r["question"] in FACTS and r.get("run", 1) == 1]
        answer = {(r[key], r["question"]): r["answer"].lower() for r in rows}
        found_a = [bool(re.search(f, answer[(a, q)])) for q in FACTS for f in FACTS[q]]
        found_b = [bool(re.search(f, answer[(b, q)])) for q in FACTS for f in FACTS[q]]
        only_a, only_b, p = mcnemar(found_a, found_b)
        print(f"  {label:42} {sum(found_a)} -> {sum(found_b)} of {len(found_a)} | "
              f"only A {only_a}, only B {only_b} | p {p:.3f}")

    print("\nmanual verdicts, Wilson 95% intervals:")
    flags = load("thesis/grounding_flags_labelled.json")
    links = [x for x in load("thesis/attribution_links_labelled.json") if x["chunk_sentence"] is not None]
    real_flags = sum(f["real_error"] for f in flags)
    correct_links = sum(x["verdict"] == "correct" for x in links)
    for label, k, n in [("grounding flags that are real errors", real_flags, len(flags)),
                        ("attribution links pointing to the right place", correct_links, len(links))]:
        low, high = wilson(k, n)
        print(f"  {label:46} {k}/{n} = {k / n:.0%}  [{low:.0%}, {high:.0%}]")

    r5_a, r5_b = hits(runs["BAAI/bge-m3"], 5), hits(runs["rerank"], 5)
    only_a, only_b, _ = mcnemar(r5_a, r5_b)
    # With no case where bge-m3 alone wins, also assume one in 24 to avoid an optimistic zero.
    print(f"\nquestions for 80% power on the R@5 gain bge-m3 -> reranker: "
          f"{questions_needed(only_a / 24, only_b / 24)} (as observed: {only_a}/{only_b} discordant), "
          f"{questions_needed(1 / 24, only_b / 24)} (assuming 1 reverse case in 24)")


if __name__ == "__main__":
    main()
