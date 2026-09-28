import json
from functools import lru_cache
from pathlib import Path

import faiss

from medguiderag.config import INDEX_DIR
from medguiderag.indexing.build_index import model_slug
from medguiderag.indexing.embed import embed_texts


@lru_cache(maxsize=None)
def _load(model_name: str, index_dir: Path = INDEX_DIR):
    out_dir = index_dir / model_slug(model_name)
    index = faiss.read_index(str(out_dir / "chunks.index"))
    with (out_dir / "metadata.jsonl").open(encoding="utf-8") as f:
        metadata = [json.loads(line) for line in f]
    return index, metadata


class Retriever:
    def __init__(self, model_name: str, index_dir: Path = INDEX_DIR):
        self.model_name = model_name
        self.index, self.metadata = _load(model_name, index_dir)

    def search_indices(self, query: str, k: int) -> list[tuple[int, float]]:
        query_vec = embed_texts([query], self.model_name, kind="query")
        scores, indices = self.index.search(query_vec, k)
        return [(int(idx), float(score)) for score, idx in zip(scores[0], indices[0]) if idx != -1]

    def search(self, query: str, k: int = 5) -> list[dict]:
        results = []
        for idx, score in self.search_indices(query, k):
            item = dict(self.metadata[idx])
            item["score"] = score
            results.append(item)
        return results
