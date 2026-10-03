"""Generate answers for the development set with the current pipeline and score them.

    python -m evaluation.generation_eval --generators 2.6B --penalties 1.0 1.05 1.1 1.2   # table "Влияние на наказанието за повторение"
    python -m evaluation.generation_eval --generators 4B 2.6B --runs 2                    # 4B vs 2.6B (iteration 3)
    python -m evaluation.generation_eval --limit 3                                        # quick check

Chunks are those of the hybrid retriever (bge-m3 + BM25 + reranker, threshold applied), read from
results/thesis/dev_contexts_hybrid.json, or with --retrieve computed again (the retrieval models are freed
before generation). Per generator, penalty and run it reports
loops (answers cut by num_predict), facts found in the six control answers, answers flagged by the grounding
check, mean answer length and speed. Answers are saved for grounding_eval.py and attribution_eval.py.
"""
import argparse
import time

import ollama
import torch

from evaluation.common import CHUNKS, FACTS, GENERATORS, QUESTIONS, chunks_by_id, count_facts, load, save
from medguiderag.generation.generator import GENERATION_OPTIONS, build_prompt
from medguiderag.generation.grounding import GroundingChecker
from medguiderag.indexing import embed
from medguiderag.retrieval import reranker
from medguiderag.retrieval.hybrid import HybridRetriever


# The Ollama runner keeps every prompt's state in RAM (~0.5 GB for 5,500 tokens); unloading the model this
# often keeps it from running out of memory ("model runner has unexpectedly stopped", std::bad_alloc).
UNLOAD_EVERY = 8


def hybrid_contexts(questions: list[str]) -> dict[str, list[dict]]:
    retriever = HybridRetriever("BAAI/bge-m3")
    contexts = {q: retriever.search(q, k=5) for q in questions}
    embed._load_model.cache_clear()
    reranker._load_model.cache_clear()
    torch.cuda.empty_cache()
    return contexts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generators", nargs="+", default=["4B"], choices=list(GENERATORS))
    parser.add_argument("--penalties", nargs="+", type=float, default=[GENERATION_OPTIONS["repeat_penalty"]])
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--limit", type=int, help="only the first N development questions")
    parser.add_argument("--retrieve", action="store_true",
                        help="retrieve the chunks again instead of reading results/thesis/dev_contexts_hybrid.json")
    parser.add_argument("--out", default="generation_answers.json")
    args = parser.parse_args()

    questions = QUESTIONS[: args.limit] if args.limit else QUESTIONS
    if args.retrieve:
        contexts = hybrid_contexts(questions)
    else:
        # The archived chunks are what the retriever returns today too; reading them keeps PyTorch models out
        # of this process, whose memory the Ollama runner needs on a 16 GB machine.
        archived = load("thesis/dev_contexts_hybrid.json")
        contexts = {q: chunks_by_id(archived[q]) for q in questions}
    print("chunks kept per question:", [len(contexts[q]) for q in questions])
    checker = GroundingChecker([c["text"] for c in CHUNKS])
    control = [q for q in questions if q in FACTS]
    rows = []
    for run in range(1, args.runs + 1):
        for name in args.generators:
            model = GENERATORS[name]
            for penalty in args.penalties:
                options = {**GENERATION_OPTIONS, "repeat_penalty": penalty}
                loops = facts = flagged = tokens = 0
                speeds, start = [], time.perf_counter()
                for n, q in enumerate(questions):
                    if not contexts[q]:  # refused: the reranker kept no chunk
                        continue
                    if n and n % UNLOAD_EVERY == 0:
                        ollama.generate(model=model, keep_alive=0)
                    r = ollama.chat(model=model, messages=[{"role": "user", "content": build_prompt(q, contexts[q])}], options=options)
                    answer = r["message"]["content"]
                    issues = checker.check(answer, contexts[q], q)
                    loops += r["done_reason"] == "length"
                    tokens += r["eval_count"]
                    speeds.append(r["eval_count"] / (r["eval_duration"] / 1e9))
                    flagged += bool(issues)
                    facts += count_facts(q, answer) if q in FACTS else 0
                    rows.append({"run": run, "generator": name, "repeat_penalty": penalty, "question": q, "answer": answer,
                                 "chunk_ids": [c["chunk_id"] for c in contexts[q]], "tokens": r["eval_count"],
                                 "loop": r["done_reason"] == "length",
                                 "grounding_issues": sorted({(i.kind, i.text, i.reason) for i in issues})})
                n = len(speeds)
                print(f"run {run} | {name:4} | penalty {penalty:<4} | loops {loops}/{n} | facts {facts}/{sum(len(FACTS[q]) for q in control)} "
                      f"({len(control)} control questions) | flagged {flagged}/{n} | mean tokens {tokens / n:.0f} | "
                      f"{sum(speeds) / n:.0f} tok/s | {(time.perf_counter() - start) / n:.1f} s/answer", flush=True)
            ollama.generate(model=model, keep_alive=0)  # unload; also frees the runner's prompt cache in RAM
    print("saved", save(args.out, rows))


if __name__ == "__main__":
    main()
