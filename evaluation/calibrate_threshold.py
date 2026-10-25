"""Reranker scores on the development set and on out-of-scope questions, behind RERANK_MIN_SCORE.

    python -m evaluation.calibrate_threshold

Prints the top-5 reranker scores per development question (G gold chunk, N its neighbour, D same guideline,
x other guideline) and per out-of-scope question, and how many chunks the threshold keeps.
"""
from evaluation.common import CHUNK_INDEX, CHUNKS, GOLD, OUT_OF_SCOPE, QUESTIONS, save
from medguiderag.config import RERANK_MIN_SCORE
from medguiderag.retrieval.hybrid import HybridRetriever


def label(i: int, gold: int) -> str:
    if i == gold:
        return "G"
    if CHUNKS[i]["document"] == CHUNKS[gold]["document"]:
        return "N" if abs(i - gold) == 1 else "D"
    return "x"


def main():
    retriever = HybridRetriever("BAAI/bge-m3")
    dev, top1, gold_scores, kept = [], [], [], []
    print("Development set, top-5 reranker scores:")
    for q, gold in zip(QUESTIONS, GOLD):
        res = retriever.search(q, k=20, min_score=0.0)
        ids = [CHUNK_INDEX[r["chunk_id"]] for r in res]
        scores = [r["score"] for r in res]
        top1.append(scores[0])
        gold_scores += [s for i, s in zip(ids, scores) if i == gold]
        kept.append(sum(s >= RERANK_MIN_SCORE for s in scores[:5]))
        top5 = list(zip(ids[:5], scores[:5]))
        dev.append({
            "question":
            q,
            "top5": [[CHUNKS[i]["chunk_id"],
                      round(s, 3),
                      label(i, gold)] for i, s in top5]
        })
        print(f"  {q[:55]:55} " + " ".join(f"{s:.3f}{label(i, gold)}"
                                           for i, s in top5))
    print("\nOut of scope, top-5 reranker scores:")
    ooc = []
    for q in OUT_OF_SCOPE:
        scores = [r["score"] for r in retriever.search(q, k=5, min_score=0.0)]
        ooc.append({"question": q, "top5": [round(s, 3) for s in scores]})
        print(f"  {q[:55]:55} " + " ".join(f"{s:.3f}" for s in scores))

    ooc_max = max(o["top5"][0] for o in ooc)
    print(
        f"\nlowest in-scope top-1 score {min(top1):.3f} | lowest gold score {min(gold_scores):.3f} | "
        f"highest out-of-scope score {ooc_max:.3f} | threshold {RERANK_MIN_SCORE}"
    )
    print(
        f"chunks kept at k=5 per question: {kept} -> {sum(k == 5 for k in kept)} of {len(kept)} keep all 5; "
        f"refused out-of-scope questions: {sum(o['top5'][0] < RERANK_MIN_SCORE for o in ooc)}/{len(ooc)}"
    )
    print(
        "saved",
        save("threshold_calibration.json", {
            "development": dev,
            "out_of_scope": ooc
        }))


if __name__ == "__main__":
    main()
