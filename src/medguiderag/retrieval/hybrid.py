from pathlib import Path

from medguiderag.config import INDEX_DIR, RERANK_CANDIDATES, RERANK_MIN_SCORE
from medguiderag.retrieval.bm25 import BM25Index
from medguiderag.retrieval.reranker import rerank_scores
from medguiderag.retrieval.retriever import Retriever

# Length of the dense and BM25 lists fed to the fusion.
FUSION_DEPTH = 50
# Standard constant from Cormack et al. (2009); damps the weight of the very top ranks.
RRF_K = 60


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = RRF_K) -> list[int]:
    score = {}
    for ranking in rankings:
        for pos, idx in enumerate(ranking):
            score[idx] = score.get(idx, 0.0) + 1 / (k + pos + 1)
    return sorted(score, key=lambda i: -score[i])


class HybridRetriever:
    """Dense and BM25 candidates fused by RRF, then reordered by a cross-encoder."""

    def __init__(self, model_name: str, index_dir: Path = INDEX_DIR):
        self.dense = Retriever(model_name, index_dir)
        self.metadata = self.dense.metadata
        self.bm25 = BM25Index([m["text"] for m in self.metadata])

    def candidates(self, query: str) -> list[int]:
        dense_ids = [idx for idx, _ in self.dense.search_indices(query, FUSION_DEPTH)]
        return reciprocal_rank_fusion([dense_ids, self.bm25.search(query, FUSION_DEPTH)])[:RERANK_CANDIDATES]

    def search(self, query: str, k: int = 5, min_score: float = RERANK_MIN_SCORE) -> list[dict]:
        """Top-k chunks by reranker score; chunks below min_score are dropped, so the list can be empty."""
        ids = self.candidates(query)
        scores = rerank_scores(query, [" ".join(self.metadata[i]["text"].split()) for i in ids])
        results = []
        for idx, score in sorted(zip(ids, scores), key=lambda x: -x[1])[:k]:
            if score < min_score:
                break
            item = dict(self.metadata[idx])
            item["score"] = score
            results.append(item)
        return results
