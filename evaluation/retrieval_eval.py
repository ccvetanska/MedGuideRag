"""Compare retrieval methods on the development set (thesis table "Сравнение на методите за извличане").

    python -m evaluation.retrieval_eval            # dense, BM25, fusion, fusion + reranker
    python -m evaluation.retrieval_eval --ollama   # also qwen3-embedding, embeddinggemma and bge-m3 served by Ollama
"""
import argparse
import re
import time

import faiss
import numpy as np
import torch
from rank_bm25 import BM25Okapi

from evaluation.common import CHUNK_INDEX, CHUNKS, QUESTIONS, metrics, save
from medguiderag.config import EMBEDDING_MODEL_CANDIDATES
from medguiderag.indexing import embed
from medguiderag.retrieval.bm25 import BM25Index
from medguiderag.retrieval.hybrid import FUSION_DEPTH, HybridRetriever, reciprocal_rank_fusion
from medguiderag.retrieval.retriever import Retriever

DEPTH = FUSION_DEPTH
OLLAMA_QWEN_TASK = "Given a web search query, retrieve relevant passages that answer the query"
# Ollama embedding models with the document and query formats their authors prescribe.
OLLAMA_MODELS = {
    "qwen3-embedding:0.6b": (lambda t: t, lambda q: f"Instruct: {OLLAMA_QWEN_TASK}\nQuery:{q}"),
    "embeddinggemma": (lambda t: f"title: none | text: {t}", lambda q: f"task: search result | query: {q}"),
    "bge-m3": (lambda t: t, lambda q: q),
}


def dense(model_name: str) -> list[list[int]]:
    retriever = Retriever(model_name)
    rankings = [[idx for idx, _ in retriever.search_indices(q, DEPTH)] for q in QUESTIONS]
    embed._load_model.cache_clear()
    torch.cuda.empty_cache()
    return rankings


def bm25_plain() -> list[list[int]]:
    """BM25 on whole words, to show what the 6-character prefix stemming of retrieval/bm25.py adds."""
    def tokenize(text):
        return re.findall(r"\w+", text.lower())
    model = BM25Okapi([tokenize(c["text"]) for c in CHUNKS])
    return [np.argsort(-model.get_scores(tokenize(q)), kind="stable")[:DEPTH].tolist() for q in QUESTIONS]


def ollama_dense(model: str) -> list[list[int]]:
    import ollama

    fmt_doc, fmt_query = OLLAMA_MODELS[model]

    def vectors(texts):
        out = []
        for i in range(0, len(texts), 16):
            out.extend(ollama.embed(model=model, input=texts[i:i + 16])["embeddings"])
        v = np.array(out, dtype="float32")
        faiss.normalize_L2(v)
        return v

    index = faiss.IndexFlatIP(len(vectors(["x"])[0]))
    index.add(vectors([fmt_doc(c["text"]) for c in CHUNKS]))
    _, idx = index.search(vectors([fmt_query(q) for q in QUESTIONS]), DEPTH)
    return idx.tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ollama", action="store_true", help="also evaluate embedding models served by Ollama")
    args = parser.parse_args()

    runs = {}
    for model in EMBEDDING_MODEL_CANDIDATES:
        runs[model.split("/")[-1]] = dense(model)
    if args.ollama:
        for model in OLLAMA_MODELS:
            runs[f"{model} (Ollama)"] = ollama_dense(model)
    runs["BM25"] = bm25_plain()
    stemmed = BM25Index([c["text"] for c in CHUNKS])
    runs["BM25, 6-character stems"] = [stemmed.search(q, DEPTH) for q in QUESTIONS]
    runs["RRF(bge-m3, BM25 stems)"] = [reciprocal_rank_fusion([d, b])[:DEPTH]
                                       for d, b in zip(runs["bge-m3"], runs["BM25, 6-character stems"])]
    start = time.perf_counter()
    hybrid = HybridRetriever("BAAI/bge-m3")
    runs["RRF + reranker"] = [[CHUNK_INDEX[r["chunk_id"]] for r in hybrid.search(q, k=20, min_score=0.0)]
                              for q in QUESTIONS]
    rerank_seconds = (time.perf_counter() - start) / len(QUESTIONS)

    print(f"\n{'method':38} | strict  R@1  R@5 R@20  MRR | lenient R@1  R@5  MRR")
    for name, rankings in runs.items():
        m = metrics(rankings)
        s, l = m["strict"], m["lenient"]
        print(f"{name:38} |        {s['R@1']:.2f} {s['R@5']:.2f} {s['R@20']:.2f} {s['MRR']:.2f} |"
              f"         {l['R@1']:.2f} {l['R@5']:.2f} {l['MRR']:.2f}")
    print(f"\nRRF + reranker: {rerank_seconds:.2f} s per question including model loading "
          f"({'GPU' if torch.cuda.is_available() else 'CPU'})")
    print("saved", save("retrieval_rankings.json", runs))


if __name__ == "__main__":
    main()
