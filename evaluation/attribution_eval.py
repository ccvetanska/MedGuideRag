"""Run the source highlighting (generation/attribution.py) over saved answers and dump every link.

    python -m evaluation.attribution_eval                   # 4B answers to the 24 questions (thesis)
    python -m evaluation.attribution_eval <answers.json>    # other saved answers with "question", "answer", "chunk_ids"

Prints the summary counts and every link (answer sentence -> chunk sentence) for manual review; sentences with
no link are listed as UNMATCHED. The manual verdicts behind the thesis numbers are in
results/thesis/attribution_links_labelled.json.
"""
import argparse
from collections import Counter

from evaluation.common import THESIS_RESULTS_DIR, chunks_by_id, load, save
from medguiderag.generation.attribution import Attributor, normalized_text
from medguiderag.indexing.embed import embed_texts

THESIS_SET = "thesis/generation_models_iteration3.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file", nargs="?", default=THESIS_SET)
    args = parser.parse_args()

    rows = load(args.file)
    if args.file == THESIS_SET:
        rows = [r for r in rows if r["generator"] == "4B" and r["run"] == 1]
    attributor = Attributor(lambda texts: embed_texts(texts, "BAAI/bge-m3", kind="query"))
    out, sentences, unmatched, sources, used = [], 0, 0, 0, 0
    for row in rows:
        chunks = chunks_by_id(row["chunk_ids"])
        result = attributor.attribute(row["answer"], chunks)
        sentences += len({l.sentence for l in result.links}) + len(result.unsupported)
        unmatched += len(result.unsupported)
        sources += len(chunks)
        used += len(result.highlights)
        for link in result.links:
            start, end = link.span
            out.append({"question": row["question"], "sentence": link.sentence,
                        "chunk_sentence": normalized_text(chunks[link.chunk])[start:end], "score": round(link.score, 2)})
        out += [{"question": row["question"], "sentence": s, "chunk_sentence": None, "score": None}
                for s in result.unsupported]

    for n, x in enumerate(out):
        if x["chunk_sentence"] is None:
            print(f"{n:3} UNMATCHED: {x['sentence'][:150]}")
        else:
            print(f"{n:3} {x['score']:.2f} A: {x['sentence'][:120]}\n          C: {x['chunk_sentence'][:140]}")
    links = sum(x["chunk_sentence"] is not None for x in out)
    print(f"\n{len(rows)} answers | sentences {sentences}, linked {sentences - unmatched}, unmatched {unmatched} | "
          f"links {links} | sources shown {sources}, used {used}")
    labelled = THESIS_RESULTS_DIR / "attribution_links_labelled.json"
    if args.file == THESIS_SET and labelled.exists():
        verdicts = Counter(x["verdict"] for x in load(labelled))
        print(f"manual verdicts on the thesis set: {dict(verdicts)}")
    print("saved", save("attribution_links.json", out))


if __name__ == "__main__":
    main()
