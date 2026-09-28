import argparse
import json
import time
from pathlib import Path

import faiss

from medguiderag.config import INDEX_DIR
from medguiderag.indexing.embed import embed_texts
from medguiderag.ingest.load_chunks import load_chunks


def model_slug(model_name: str) -> str:
    return model_name.replace("/", "__")


def build_index(model_name: str, index_dir: Path = INDEX_DIR) -> dict:
    chunks = load_chunks()
    texts = [c["text"] for c in chunks]

    start = time.perf_counter()
    embeddings = embed_texts(texts, model_name, kind="passage")
    elapsed = time.perf_counter() - start

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    out_dir = index_dir / model_slug(model_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / "chunks.index"))
    with (out_dir / "metadata.jsonl").open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps({
                "chunk_id": c["chunk_id"],
                "document": c["document"],
                "page_start": c["page_start"],
                "page_end": c["page_end"],
                "section": c["section"],
                "text": c["text"],
            }, ensure_ascii=False) + "\n")

    stats = {
        "model": model_name,
        "num_chunks": len(chunks),
        "dimension": dim,
        "embedding_seconds": round(elapsed, 2),
    }
    with (out_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name")
    args = parser.parse_args()
    print(build_index(args.model_name))
