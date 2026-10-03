"""Iteration 1: prompt variants on fixed chunks, and the same prompt across embedding models (thesis tables
"Възпроизведени факти ... при различни варианти на подсказката" and "Позиция на откъса с отговора ...").

    python -m evaluation.prompt_variants

Reproduces the iteration-1 setup, not the current system: dense retrieval only (k=5), BgGPT 2.6B, no repeat
penalty, the prompt variants as they were then. Generation on the GPU is not fully repeatable, so counts may
differ by a fact or two from the archived run in results/thesis/.
"""
import ollama
import torch

from evaluation.common import DEV, FACTS, M26, count_facts, save
from medguiderag.config import EMBEDDING_MODEL_CANDIDATES
from medguiderag.indexing import embed
from medguiderag.retrieval.retriever import Retriever

QUESTIONS = ["Как се лекува хиперемезис гравидарум?",
             "Кога се прилага инсулин при захарен диабет тип 2?",
             "Какво е лечението на атопичен дерматит?"]
GOLD = {d["question"]: d["chunk_id"] for d in DEV}
OPTIONS = {"num_ctx": 8192, "temperature": 0, "seed": 42}  # iteration-1 settings

BASE = ("Отговори на въпроса само с информация от горните откъси от национални "
        "фармакотерапевтични ръководства. Не добавяй факти, които не са в откъсите.")
CONCRETE = ("Отговори на въпроса подробно и конкретно, само с информация от горните откъси от национални "
            "фармакотерапевтични ръководства. Изброй конкретните мерки, лекарства и дозировки, които "
            "откъсите посочват по въпроса. Пропусни откъсите, които не се отнасят до въпроса. "
            "Не добавяй нищо, което не е в откъсите.")
VARIANTS = {  # name: (normalized text, number of chunks, most relevant chunk last, instruction)
    "V0 raw, 5, relevance order": (False, 5, False, BASE),
    "V1 normalized": (True, 5, False, BASE),
    "V2 most relevant last": (True, 5, True, BASE),
    "V3 V2 with 3 chunks": (True, 3, True, BASE),
    "V4 V2, itemizing instruction": (True, 5, True, CONCRETE),
    "V5 V4 with 3 chunks": (True, 3, True, CONCRETE),
    "V6 top chunk only, itemizing": (True, 1, True, CONCRETE),
}


def build(question, chunks, clean, k, last, instruction):
    chunks = chunks[:k]
    if last:
        chunks = list(reversed(chunks))
    parts = [f"[{c['document']}, стр. {c['page_start']}-{c['page_end']}]\n" + (" ".join(c["text"].split()) if clean else c["text"])
             for c in chunks]
    return "Откъси:\n" + "\n\n".join(parts) + f"\n\n{instruction}\n\nВъпрос: {question}"


def generate(prompt):
    r = ollama.chat(model=M26, messages=[{"role": "user", "content": prompt}], options=OPTIONS)
    return r["message"]["content"], r["prompt_eval_count"]


def retrieve(model_name):
    retriever = Retriever(model_name)
    out = {q: retriever.search(q, k=5) for q in QUESTIONS}
    embed._load_model.cache_clear()
    torch.cuda.empty_cache()
    return out


def main():
    results = {"prompt_variants": [], "embedding_models": []}
    retrieved = {m: retrieve(m) for m in EMBEDDING_MODEL_CANDIDATES}

    print("Prompt variants, chunks from bge-m3 (facts found / facts listed):")
    for name, variant in VARIANTS.items():
        row = []
        for q in QUESTIONS:
            answer, prompt_tokens = generate(build(q, retrieved["BAAI/bge-m3"][q], *variant))
            facts = count_facts(q, answer)
            row.append(f"{facts}/{len(FACTS[q])}")
            results["prompt_variants"].append({"variant": name, "question": q, "prompt_tokens": prompt_tokens,
                                               "facts": facts, "answer": answer})
        print(f"  {name:32} hyperemesis {row[0]:6} insulin {row[1]:5} dermatitis {row[2]}")

    print("\nV2 across embedding models (position of the gold chunk among the 5; facts found):")
    for model, by_question in retrieved.items():
        cells = []
        for q in QUESTIONS:
            ids = [c["chunk_id"] for c in by_question[q]]
            position = ids.index(GOLD[q]) + 1 if GOLD[q] in ids else None
            answer, _ = generate(build(q, by_question[q], *VARIANTS["V2 most relevant last"]))
            cells.append(f"{'#' + str(position) if position else 'not retrieved'}; {count_facts(q, answer)}")
            results["embedding_models"].append({"model": model, "question": q, "gold_position": position,
                                                "chunk_ids": ids, "facts": count_facts(q, answer), "answer": answer})
        print(f"  {model.split('/')[-1]:40} " + " | ".join(cells))
    ollama.generate(model=M26, keep_alive=0)  # unload; frees the runner's prompt cache too
    print("saved", save("iteration1_prompt_variants.json", results))


if __name__ == "__main__":
    main()
