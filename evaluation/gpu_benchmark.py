"""GPU speed and half-precision equivalence (thesis table "Време за изпълнение на централния и на графичния процесор").

    python -m evaluation.gpu_benchmark

Reranks the 20 candidates of each development question in fp32 and fp16 and compares the top 5 with the
CPU run archived in results/thesis/rerank_rankings_cpu.json; then builds the bge-m3 index on the GPU into a
temporary directory and compares its vectors with the index in data/index (built on the CPU).
"""
import tempfile
import time
from pathlib import Path

import faiss
import torch
from sentence_transformers import CrossEncoder

from evaluation.common import QUESTIONS, load
from medguiderag.config import INDEX_DIR, RERANKER_MODEL
from medguiderag.indexing import embed
from medguiderag.indexing.build_index import build_index, model_slug
from medguiderag.retrieval.hybrid import HybridRetriever
from medguiderag.retrieval.reranker import MAX_LENGTH


def main():
    free, total = torch.cuda.mem_get_info()
    print(
        f"{torch.__version__} | {torch.cuda.get_device_name(0)} | VRAM free {free / 2**30:.2f} of {total / 2**30:.2f} GB"
    )
    retriever = HybridRetriever("BAAI/bge-m3")
    pools = [retriever.candidates(q) for q in QUESTIONS]
    texts = [" ".join(m["text"].split()) for m in retriever.metadata]
    embed._load_model.cache_clear()
    torch.cuda.empty_cache()
    cpu = load("thesis/rerank_rankings_cpu.json")

    for dtype in (torch.float32, torch.float16):
        model = CrossEncoder(RERANKER_MODEL,
                             max_length=MAX_LENGTH,
                             device="cuda",
                             model_kwargs={"torch_dtype": dtype})
        model.predict([(QUESTIONS[0], texts[pools[0][0]])])  # warm-up
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        reranked = []
        for q, pool in zip(QUESTIONS, pools):
            scores = model.predict([(q, texts[i]) for i in pool], batch_size=4)
            reranked.append([
                pool[j]
                for j in sorted(range(len(pool)), key=lambda j: -scores[j])
            ])
        torch.cuda.synchronize()
        seconds = (time.perf_counter() - start) / len(QUESTIONS)
        same = sum(a[:5] == b[:5] for a, b in zip(reranked, cpu))
        print(
            f"reranker {str(dtype):14} {seconds:.2f} s per question (20 pairs) | top 5 identical to the CPU run: "
            f"{same}/{len(QUESTIONS)} | peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GB"
        )
        del model
        torch.cuda.empty_cache()

    def vectors(index_dir: Path, count: int):
        return faiss.read_index(
            str(index_dir / model_slug("BAAI/bge-m3") /
                "chunks.index")).reconstruct_n(0, count)

    with tempfile.TemporaryDirectory() as tmp:
        stats = build_index("BAAI/bge-m3", index_dir=Path(tmp))
        new = vectors(Path(tmp), stats["num_chunks"])
    old = vectors(INDEX_DIR, stats["num_chunks"])
    cos = (old * new).sum(1)
    print(
        f"bge-m3 index built on the GPU in {stats['embedding_seconds']:.0f} s | cosine to the CPU-built vectors: "
        f"min {cos.min():.5f}, mean {cos.mean():.5f}")


if __name__ == "__main__":
    main()
