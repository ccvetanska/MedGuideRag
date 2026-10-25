"""Shared pieces of the evaluation scripts: the development set, retrieval metrics, result files."""
import json
import sys
from pathlib import Path

from medguiderag.ingest.load_chunks import load_chunks

EVAL_DIR = Path(__file__).resolve().parent
DATA_DIR = EVAL_DIR / "data"
RESULTS_DIR = EVAL_DIR / "results"  # where the scripts write
THESIS_RESULTS_DIR = RESULTS_DIR / "thesis"  # archived outputs the thesis numbers come from; never overwritten

sys.stdout.reconfigure(encoding="utf-8")

# Same order as every FAISS index's metadata.jsonl, so a ranking is a list of indices into this list.
CHUNKS = load_chunks()
CHUNK_INDEX = {c["chunk_id"]: i for i, c in enumerate(CHUNKS)}

DEV = json.load(open(DATA_DIR / "dev_questions.json", encoding="utf-8"))
QUESTIONS = [d["question"] for d in DEV]
GOLD = [CHUNK_INDEX[d["chunk_id"]] for d in DEV]
OUT_OF_SCOPE = json.load(open(DATA_DIR / "out_of_scope.json",
                              encoding="utf-8"))
# Key facts of the source chunk(s) as regular expressions, for the control questions.
FACTS = json.load(open(DATA_DIR / "facts.json", encoding="utf-8"))


def _lenient_hit(pred: int, gold: int) -> bool:
    """The gold chunk or its neighbour in the same guideline, which often holds the rest of the answer."""
    return pred == gold or (abs(pred - gold) == 1 and CHUNKS[pred]["document"]
                            == CHUNKS[gold]["document"])


def metrics(rankings: list[list[int]]) -> dict:
    """Recall@1/5/10/20 and MRR@10 over the development set, strict (exact chunk) and lenient (±1 neighbour)."""
    out = {}
    for mode, hit in (("strict", lambda p, g: p == g), ("lenient",
                                                        _lenient_hit)):
        recall = {k: 0.0 for k in (1, 5, 10, 20)}
        mrr = 0.0
        for ranks, gold in zip(rankings, GOLD):
            pos = next((i for i, p in enumerate(ranks[:20]) if hit(p, gold)),
                       None)
            if pos is None:
                continue
            for k in recall:
                recall[k] += pos < k
            mrr += 1 / (pos + 1) if pos < 10 else 0
        n = len(GOLD)
        out[mode] = {
            **{
                f"R@{k}": v / n
                for k, v in recall.items()
            }, "MRR": mrr / n
        }
    return out


def save(name: str, data) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / name
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    return path


def load(path: str | Path):
    """A results file by path, or by name under results/ (e.g. "thesis/generation_penalty_sweep.json")."""
    path = Path(path)
    return json.load(
        open(path if path.exists() else RESULTS_DIR / path, encoding="utf-8"))


def chunks_by_id(chunk_ids: list[str]) -> list[dict]:
    """Chunk dicts as the retrievers return them (with the text the prompt and the checks use)."""
    return [CHUNKS[CHUNK_INDEX[i]] for i in chunk_ids]
