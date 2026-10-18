"""Final evaluation on 60 independently written questions and 10 outside the guidelines (thesis section
"Качествен анализ и оценка").

    python -m evaluation.final_eval check    # while writing data/final_questions.json: finds each answer's chunk
    python -m evaluation.final_eval run      # app stopped: retrieval, answers, checks; writes the files to label
    python -m evaluation.final_eval report   # after the manual verdicts in results/thesis/final_labels.json

The pipeline is the app's with its defaults, fixed before the questions were written: hybrid retrieval with
the reranker and its threshold, k = 5, BgGPT 4B with repeat penalty 1.2, the grounding check and the
attribution. Nothing is tuned on this set. How to write the questions and label the answers: evaluation/README.md.
"""
import argparse
import json
import re
import statistics
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path

from evaluation.common import CHUNK_INDEX, CHUNKS, DATA_DIR, GOLD, THESIS_RESULTS_DIR, chunks_by_id
from evaluation.significance import mcnemar, randomization, wilson
from medguiderag.config import DOCUMENTS_PATH

QUESTIONS_PATH = DATA_DIR / "final_questions.json"
ANSWERS_NAME, LABELS_NAME, REVIEW_NAME = "final_answers.json", "final_labels.json", "final_review.md"
K = 5
# Share of the quote's words that must appear, in order, in a chunk for it to count as the answer's chunk.
MATCH_MIN = 0.8
QUOTE_MIN_WORDS = 6
# See generation_eval.py: unloading the model this often keeps the Ollama runner's prompt cache within RAM.
UNLOAD_EVERY = 8
OTHER_EMBEDDINGS = {"intfloat/multilingual-e5-base": "e5-base",
                    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": "mpnet"}
# Pairs compared for significance, as in the thesis table "Значимост на основните разлики при извличането".
COMPARISONS = [("bge-m3", "e5-base"), ("bge-m3", "mpnet"), ("bge-m3", "BM25"), ("bge-m3", "RRF"),
               ("RRF", "RRF + reranker"), ("bge-m3", "RRF + reranker")]

ANSWER_VERDICTS = {"complete", "partial", "wrong"}
OUT_OF_SCOPE_VERDICTS = {"declined", "fabricated"}
LINK_VERDICTS = {"correct", "wrong", "unclear"}
UNMATCHED_VERDICTS = {"unsupported", "missed_match", "unclear"}

# OCR mixes Latin lookalikes into Cyrillic words ("А02ВА02" with Latin A or B); both sides are unified so a
# quote copied from the PDF matches the extracted text.
_HOMOGLYPHS = str.maketrans("ABCEHKMOPTXaceopxy", "АВСЕНКМОРТХасеорху")


def _words(text: str) -> list[str]:
    return re.findall(r"\w+", text.translate(_HOMOGLYPHS).lower())


_CHUNK_WORDS, _PAGE_WORDS = {}, {}


def _found_share(quote: list[str], text: list[str]) -> float:
    """Share of the quote's words found in order in the text, in runs of at least two words: single common
    words ("на", "mg", "1") match about half of an unrelated quote."""
    blocks = SequenceMatcher(None, quote, text, autojunk=False).get_matching_blocks()
    return sum(b.size for b in blocks if b.size >= 2) / len(quote)


def _coverage(quote: list[str], *chunks: int) -> float:
    for idx in chunks:
        if idx not in _CHUNK_WORDS:
            _CHUNK_WORDS[idx] = _words(CHUNKS[idx]["text"])
    return _found_share(quote, [w for idx in chunks for w in _CHUNK_WORDS[idx]])


def _lost_in_chunking(item: dict) -> bool:
    """Whether the quote is on its page in the PDF text layer although no chunk holds it. About 1% of the
    text never reached the chunks, mostly drug tables (pages 30-37 of the urology guideline)."""
    if not _PAGE_WORDS:
        for line in open(DOCUMENTS_PATH, encoding="utf-8"):
            d = json.loads(line)
            _PAGE_WORDS[d["document"], d["page"]] = _words(d["text_raw"])
    text = [w for p in (item["page"] - 1, item["page"], item["page"] + 1)
            for w in _PAGE_WORDS.get((item["document"], p), [])]
    return _found_share(_words(item["quote"]), text) >= MATCH_MIN


def find_gold(item: dict) -> tuple[list[int], list[int], int, float]:
    """Chunks of the guideline holding the quote (chunks overlap, so often two), chunks of other guidelines
    with the same text, the closest chunk and its coverage."""
    quote = _words(item["quote"])
    doc = [i for i, c in enumerate(CHUNKS) if c["document"] == item["document"]]
    on_page = [i for i in doc if CHUNKS[i]["page_start"] <= item["page"] <= CHUNKS[i]["page_end"]]
    cover = {i: _coverage(quote, i) for i in doc}
    best = max(on_page or doc, key=cover.get)
    gold = [i for i in doc if cover[i] >= MATCH_MIN]
    # A phrase repeated elsewhere in the guideline (the same regimen in several protocols): the page decides.
    gold = [i for i in gold if i in on_page] or gold
    if not gold:
        # A quote across a chunk boundary: the closest chunk and a neighbour hold it together.
        for pair in ((best - 1, best), (best, best + 1)):
            if all(i in cover for i in pair) and _coverage(quote, *pair) >= MATCH_MIN:
                gold = list(pair)
                break
    # The 2018 hepatitis and IBD guidelines repeat parts of the 2016 gastroenterology one word for word;
    # the same passage there is an equally good source.
    copies = [i for i, c in enumerate(CHUNKS)
              if gold and c["document"] != item["document"] and _coverage(quote, i) >= MATCH_MIN]
    return gold, copies, best, cover[best]


def load_questions(path: Path) -> dict:
    try:
        return json.load(open(path, encoding="utf-8"))
    except json.JSONDecodeError as e:
        sys.exit(f"{path.name}: грешка в JSON на ред {e.lineno}, позиция {e.colno}: {e.msg}. "
                 "Най-често: липсваща или излишна запетая, или кавичка \" в текста (пишете „…“).")


def _pages(idx: int) -> str:
    c = CHUNKS[idx]
    return f"стр. {c['page_start']}-{c['page_end']}"


def _as_list(value) -> list:
    if isinstance(value, list):
        return value
    return [] if value is None or value == "" else [value]


def _quotes(item: dict) -> list[tuple[str, int]]:
    """(quote, page) pairs: an answer spread over several places has a list of quotes, with one page for all
    of them or a list of pages of the same length."""
    quotes = [q for q in _as_list(item.get("quote")) if isinstance(q, str) and q.strip()]
    pages = _as_list(item.get("page"))
    return list(zip(quotes, pages * len(quotes) if len(pages) == 1 else pages))


def inspect(questions: dict) -> tuple[list[str], int, dict[str, list[int]]]:
    """Report lines, number of problems that block the run, and the gold chunks of every complete question."""
    lines, problems, gold_of = [], 0, {}
    dev_gold, used, empty = set(GOLD), {}, []
    doc_names = {i["guideline"]: i["document"] for i in questions["in_scope"]}
    documents = {c["document"] for c in CHUNKS}
    for item in questions["in_scope"]:
        quotes = [q for q in _as_list(item.get("quote")) if isinstance(q, str) and q.strip()]
        pages = _as_list(item.get("page"))
        missing = [f for f, v in (("question", item.get("question", "").strip()), ("page", pages), ("quote", quotes))
                   if not v]
        if not any(f.strip() for f in item.get("facts", [])):
            missing.append("facts")
        if len(missing) == 4:
            empty.append(item["id"])
            problems += 1
            continue
        if missing:
            lines.append(f"{item['id']} не е попълнен: {', '.join(missing)}")
            problems += 1
            continue
        if not all(isinstance(p, int) for p in pages) or len(pages) not in (1, len(quotes)):
            lines.append(f"{item['id']} page трябва да е число без кавички, а при няколко цитата от различни страници "
                         f"– списък с по една страница за всеки цитат, напр. [85, 86]")
            problems += 1
            continue
        gold, copies, failed, notes = [], [], [], []
        for n, (quote, page) in enumerate(_quotes(item), 1):
            label = f"цитат {n}" if len(quotes) > 1 else "цитатът"
            part = {"document": item["document"], "quote": quote, "page": page}
            if len(_words(quote)) < QUOTE_MIN_WORDS:
                failed.append(f"{label} е твърде кратък: копирайте поне {QUOTE_MIN_WORDS} поредни думи")
                continue
            found, same_text, best, score = find_gold(part)
            if not found and _lost_in_chunking(part):
                failed.append(f"{label} е на стр. {page} в PDF-а, но този текст не е влязъл в откъсите на системата "
                              "(част от таблиците не е извлечена); изберете друго място")
                continue
            if not found:
                failed.append(f"{label} не е намерен в {item['document']}; най-близо е "
                              f"{CHUNKS[best]['chunk_id']} ({_pages(best)}) с {score:.0%} от думите")
                continue
            if not any(CHUNKS[g]["page_start"] <= page <= CHUNKS[g]["page_end"] for g in found):
                notes.append(f"внимание: страница {page} не съвпада с откъса на {label}. В page се пише номерът "
                             "на страницата в PDF четеца (корицата е 1), а не отпечатаният номер")
            if max(found) - min(found) >= len(found):
                notes.append(f"внимание: {label} се среща на няколко места в ръководството и всички се броят за "
                             "верни; ако отговорът е само на едно от тях, копирайте по-дълъг цитат")
            if len(same_text) > 3:
                notes.append(f"внимание: {label} се среща в {len(same_text)} откъса от други ръководства (общ "
                             "текст); изберете място със същинско медицинско съдържание")
            elif same_text:
                notes.append("същият текст има и в " + ", ".join(f"{CHUNKS[g]['chunk_id']} ({_pages(g)})"
                                                                 for g in same_text)
                             + "; и той се брои за верен източник")
            gold += [g for g in found if g not in gold]
            copies += [g for g in same_text if g not in copies]
        # A passage of another guideline that answers the question too, e.g. the same dose written differently in
        # a newer version: it counts as correct for retrieval, and the review shows which one the system used.
        for extra in item.get("also_correct", []):
            document = doc_names.get(extra.get("document", ""), extra.get("document", ""))
            part = {"document": document, "quote": extra.get("quote", ""), "page": extra.get("page")}
            if (document not in documents or not isinstance(part["page"], int)
                    or len(_words(part["quote"])) < QUOTE_MIN_WORDS):
                failed.append("also_correct изисква document (ръководство, както е в guideline), page (число) и "
                              f"quote (поне {QUOTE_MIN_WORDS} думи)")
                continue
            found, same_text, best, score = find_gold(part)
            if not found:
                failed.append(f"цитатът в also_correct не е намерен в {document}; най-близо е "
                              f"{CHUNKS[best]['chunk_id']} ({_pages(best)}) с {score:.0%} от думите")
                continue
            notes.append("верен е и " + ", ".join(f"{CHUNKS[g]['chunk_id']} ({_pages(g)})" for g in found)
                         + " (also_correct)")
            copies += [g for g in found + same_text if g not in copies]
        if failed:
            lines += [f"{item['id']} {f}" for f in failed]
            problems += 1
            continue
        gold_of[item["id"]] = gold + [g for g in copies if g not in gold]
        lines.append(f"{item['id']} наред: " + ", ".join(f"{CHUNKS[g]['chunk_id']} ({_pages(g)})" for g in gold))
        lines += [f"    {n}" for n in dict.fromkeys(notes)]
        if dev_gold & set(gold):
            lines.append("    внимание: същият откъс е използван в набора за разработка; изберете друго място")
        earlier = sorted({used[g] for g in gold if g in used})
        if earlier:
            lines.append(f"    внимание: същият откъс като при {', '.join(earlier)}; въпросите трябва да са за "
                         "различни места в ръководството")
        used.update({g: item["id"] for g in gold})
        facts = [f for f in item["facts"] if f.strip()]
        if not 2 <= len(facts) <= 4:
            count = "1 ключов факт" if len(facts) == 1 else f"{len(facts)} ключови факта"
            lines.append(f"    внимание: {count}; препоръчват се 2-4")
    # A missing detail names its guideline by file or by the guideline field of the in-scope questions.
    known = {i["document"] for i in questions["in_scope"]} | {i["guideline"] for i in questions["in_scope"]}
    for item in questions["out_of_scope"]:
        if not item.get("question", "").strip():
            empty.append(item["id"])
            problems += 1
        elif "document" in item and item["document"].strip() not in known:
            lines.append(f"{item['id']} в document напишете ръководството, което покрива темата, както е в полето "
                         "guideline на въпросите F (напр. „Неврологични заболявания“)")
            problems += 1
        else:
            lines.append(f"{item['id']} наред")
    if empty:
        lines.append(f"Непопълнени ({len(empty)}): {', '.join(empty)}")
    texts = [i.get("question", "").strip() for i in questions["in_scope"] + questions["out_of_scope"]]
    for text in {t for t in texts if t and texts.count(t) > 1}:
        lines.append(f"Повтарящ се въпрос: {text}")
        problems += 1
    return lines, problems, gold_of


def check(args):
    questions = load_questions(Path(args.questions))
    lines, problems, _ = inspect(questions)
    print("\n".join(lines))
    total = len(questions["in_scope"]) + len(questions["out_of_scope"])
    print(f"\nГотови: {total - problems} от {total}." + ("" if problems else " Може да се пусне run (със спряно приложение)."))


# ---- run ----

def retrieve(rows: list[dict]):
    import torch

    from medguiderag.config import RERANK_CANDIDATES, RERANK_MIN_SCORE
    from medguiderag.indexing import embed
    from medguiderag.retrieval import reranker
    from medguiderag.retrieval.hybrid import FUSION_DEPTH, HybridRetriever, reciprocal_rank_fusion
    from medguiderag.retrieval.retriever import Retriever

    hybrid = HybridRetriever("BAAI/bge-m3")
    for row in rows:
        q = row["question"]
        start = time.perf_counter()
        reranked = hybrid.search(q, k=RERANK_CANDIDATES, min_score=0.0)
        row["retrieval_seconds"] = round(time.perf_counter() - start, 2)
        dense = [idx for idx, _ in hybrid.dense.search_indices(q, FUSION_DEPTH)]
        bm25 = hybrid.bm25.search(q, FUSION_DEPTH)
        # Rankings as indices in the order of chunks.jsonl, like the other archived rankings.
        row["rankings"] = {
            "bge-m3": dense,
            "BM25": bm25,
            "RRF": reciprocal_rank_fusion([dense, bm25])[:FUSION_DEPTH],
            "RRF + reranker": [CHUNK_INDEX[c["chunk_id"]] for c in reranked],
        }
        row["rerank_scores"] = [round(c["score"], 4) for c in reranked]
        # What the app's search(k=5) returns: the reranked top 5 without the chunks under the threshold.
        row["chunk_ids"] = [c["chunk_id"] for c in reranked[:K] if c["score"] >= RERANK_MIN_SCORE]
        print(f"{row['id']} retrieved {len(row['chunk_ids'])} chunks, top score {row['rerank_scores'][0]:.3f}", flush=True)
    embed._load_model.cache_clear()
    reranker._load_model.cache_clear()
    torch.cuda.empty_cache()
    # The other embedding candidates of the thesis table "Сравнение на методите за извличане", one at a time.
    for model, label in OTHER_EMBEDDINGS.items():
        retriever = Retriever(model)
        for row in rows:
            row["rankings"][label] = [idx for idx, _ in retriever.search_indices(row["question"], FUSION_DEPTH)]
        embed._load_model.cache_clear()
        torch.cuda.empty_cache()
        print(f"ranked with {label}", flush=True)


def generate(rows: list[dict]):
    import ollama

    from medguiderag.config import OLLAMA_MODEL
    from medguiderag.generation.generator import generate_answer

    answered = [r for r in rows if r["chunk_ids"]]
    for n, row in enumerate(answered):
        if n and n % UNLOAD_EVERY == 0:
            ollama.generate(model=OLLAMA_MODEL, keep_alive=0)
        start = time.perf_counter()
        row["answer"] = generate_answer(row["question"], chunks_by_id(row["chunk_ids"]))
        row["generation_seconds"] = round(time.perf_counter() - start, 2)
        print(f"{row['id']} answered in {row['generation_seconds']:.1f} s", flush=True)
    ollama.generate(model=OLLAMA_MODEL, keep_alive=0)


def check_answers(rows: list[dict]):
    from medguiderag.generation.attribution import Attributor, normalized_text
    from medguiderag.generation.grounding import GroundingChecker
    from medguiderag.indexing.embed import embed_texts

    checker = GroundingChecker([c["text"] for c in CHUNKS])
    attributor = Attributor(lambda texts: embed_texts(texts, "BAAI/bge-m3", kind="query"))
    for row in rows:
        if row.get("answer") is None:
            continue
        chunks = chunks_by_id(row["chunk_ids"])
        issues = checker.check(row["answer"], chunks, row["question"])
        row["grounding_spans"] = sorted({(i.start, i.end) for i in issues})
        row["grounding_issues"] = [{"kind": k, "text": t, "reason": why}
                                   for k, t, why in sorted({(i.kind, i.text, i.reason) for i in issues})]
        result = attributor.attribute(row["answer"], chunks)
        row["links"] = []
        for link in result.links:
            start, end = link.span
            row["links"].append({"sentence": link.sentence, "chunk": link.chunk, "score": round(link.score, 2),
                                 "source": normalized_text(chunks[link.chunk])[start:end]})
        row["unmatched"] = result.unsupported
        row["sources_used"] = len(result.highlights)


def _answer_lines(answer: str | None) -> list[str] | None:
    return None if answer is None else [line for line in answer.splitlines() if line.strip()]


LABELS_GUIDE = {
    "answer_verdict": "complete (верен и пълен) | partial (верен, но непълен) | wrong (съдържа невярно твърдение); "
                      "за O-въпросите: declined (казва, че няма информация) | fabricated (отговаря без опора)",
    "present, real_error": "true | false",
    "missed_errors": "неподкрепени подробности от отговора, които проверката не е отбелязала, напр. [\"8-16 mg/kg\"]",
    "links.verdict": "correct | wrong | unclear (по желание)",
    "unmatched.verdict": "unsupported (наистина не е в откъсите) | missed_match (има го, но не е свързано) | unclear "
                         "(по желание)",
}


def labels_template(rows: list[dict]) -> dict:
    labels = {"_възможни_стойности": LABELS_GUIDE}
    for row in rows:
        refused = not row["chunk_ids"]
        entry = {"question": row["question"], "answer": _answer_lines(row.get("answer"))}
        if row["id"].startswith("F"):
            entry["answer_verdict"] = "refused" if refused else None
            entry["facts"] = [{"fact": f, "present": False if refused else None} for f in row["facts"]]
        else:
            entry["refused"] = refused
            if not refused:
                entry["answer_verdict"] = None
        if not refused:
            entry["flags"] = [{"text": i["text"], "reason": i["reason"], "real_error": None}
                              for i in row["grounding_issues"]]
            entry["missed_errors"] = []
            if row["id"].startswith("F"):
                entry["links"] = [{"answer": l["sentence"], "source": l["source"], "verdict": None} for l in row["links"]]
                entry["unmatched"] = [{"sentence": s, "verdict": None} for s in row["unmatched"]]
        entry["note"] = ""
        labels[row["id"]] = entry
    return labels


def dump_labels(labels: dict, path: Path):
    """JSON with every fact, flag and link on one line, so that labelling is one edit per line."""
    def value(v, indent):
        if isinstance(v, list) and v:
            inner = ",\n".join(indent + " " + json.dumps(x, ensure_ascii=False) for x in v)
            return f"[\n{inner}\n{indent}]"
        return json.dumps(v, ensure_ascii=False)

    blocks = []
    for qid, entry in labels.items():
        fields = ",\n".join(f'  "{k}": {value(v, "  ")}' for k, v in entry.items())
        blocks.append(f' "{qid}": {{\n{fields}\n }}')
    path.write_text("{\n" + ",\n".join(blocks) + "\n}\n", encoding="utf-8")


def _marked(answer: str, spans: list[list[int]]) -> str:
    out, pos = [], 0
    for start, end in spans:
        if start < pos:
            continue
        out += [answer[pos:start], f"<mark>{answer[start:end]}</mark>"]
        pos = end
    return "".join(out + [answer[pos:]])


def review_markdown(rows: list[dict]) -> str:
    out = ["# Окончателна оценка: отговори за преглед",
           "Присъдите се записват в `final_labels.json`. С <mark>маркиране</mark> са подробностите, "
           "отбелязани от проверката.", ""]
    for row in rows:
        title = row.get("guideline") or row.get("kind", "")
        out += [f"## {row['id']} · {title}", f"**Въпрос:** {row['question']}", ""]
        if row.get("note"):
            out += [f"**Бележка:** {row['note']}", ""]
        if row["id"].startswith("F"):
            out += ["**Отговор в ръководството:** " + " … ".join(f"„{q}“ (стр. {p})" for q, p in _quotes(row)), ""]
            for e in row.get("also_correct", []):
                out += [f"**Верен е и:** {e['document']}, стр. {e['page']}: „{e['quote']}“", ""]
            out += ["**Ключови факти:** " + "; ".join(f"{n}) {f}" for n, f in enumerate(row["facts"], 1)), ""]
        out.append("**Показани откъси:**")
        gold = set(row.get("gold", []))
        for n, (cid, score) in enumerate(zip(row["chunk_ids"], row["rerank_scores"]), 1):
            idx = CHUNK_INDEX[cid]
            mark = " **← откъсът с отговора**" if idx in gold else ""
            out.append(f"{n}. {CHUNKS[idx]['document']}, {_pages(idx)}, релевантност {score:.3f}{mark}")
        if not row["chunk_ids"]:
            out += ["няма: системата отказва да отговори "
                    f"(най-висока оценка {row['rerank_scores'][0]:.3f}, праг 0.1)", ""]
            continue
        out += ["", "**Отговор:**", ""]
        out += ["> " + line for line in _marked(row["answer"], row["grounding_spans"]).splitlines() if line.strip()]
        out.append("")
        if row["grounding_issues"]:
            out.append("**Сигнали на проверката:**")
            out += [f"- {i['text']} ({i['reason']})" for i in row["grounding_issues"]]
            out.append("")
        if row["id"].startswith("F") and row["links"]:
            out.append("**Връзки към източниците** (изречение от отговора → изречение от откъс):")
            for n, l in enumerate(row["links"], 1):
                out.append(f"{n}. {l['sentence']}  \n   → откъс {l['chunk'] + 1}: {l['source']}")
            out.append("")
        if row["id"].startswith("F") and row["unmatched"]:
            out.append("**Изречения без съответствие:**")
            out += [f"- {s}" for s in row["unmatched"]]
            out.append("")
    return "\n".join(out) + "\n"


def run(args):
    questions = load_questions(Path(args.questions))
    lines, problems, gold_of = inspect(questions)
    if problems:
        print("\n".join(lines))
        sys.exit(f"\n{problems} въпроса не са готови; поправете ги и пуснете отново (виж check).")
    out_dir = Path(args.out_dir)
    targets = [out_dir / name for name in (ANSWERS_NAME, LABELS_NAME, REVIEW_NAME)]
    if not args.force and any(p.exists() for p in targets):
        sys.exit(f"{out_dir} вече съдържа резултати от окончателната оценка. Пускането наново дава други "
                 "отговори и изтрива ръчните присъди; ако това е нарочно, добавете --force.")

    rows = []
    for item in questions["in_scope"]:
        rows.append({**item, "facts": [f for f in item["facts"] if f.strip()], "gold": gold_of[item["id"]]})
    rows += [dict(item) for item in questions["out_of_scope"]]
    start = time.perf_counter()
    retrieve(rows)
    generate(rows)
    check_answers(rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(targets[0], "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    dump_labels(labels_template(rows), targets[1])
    targets[2].write_text(review_markdown(rows), encoding="utf-8")
    print(f"\n{len(rows)} questions in {(time.perf_counter() - start) / 60:.1f} min")
    for p in targets:
        print("saved", p)


# ---- report ----

def _unlabelled(labels: dict) -> list[str]:
    """Per question, the required verdicts still missing or misspelled; the link verdicts are optional."""
    out = []
    for qid, e in labels.items():
        allowed = ANSWER_VERDICTS | {"refused"} if qid.startswith("F") else OUT_OF_SCOPE_VERDICTS
        missing = []
        if "answer_verdict" in e and e["answer_verdict"] not in allowed:
            missing.append(f"answer_verdict ({' | '.join(sorted(allowed - {'refused'}))})")
        facts = sum(not isinstance(f["present"], bool) for f in e.get("facts", []))
        flags = sum(not isinstance(f["real_error"], bool) for f in e.get("flags", []))
        missing += [f"present при {facts} факта"] if facts else []
        missing += [f"real_error при {flags} сигнала"] if flags else []
        if missing:
            out.append(f"{qid}: " + ", ".join(missing))
    return out


def _hit(ranking: list[int], gold: list[int], k: int, lenient: bool) -> int | None:
    """Rank (0-based) of the first gold chunk in the top k; with lenient, a neighbour in the same guideline counts."""
    for pos, idx in enumerate(ranking[:k]):
        if idx in gold or lenient and any(abs(idx - g) == 1 and CHUNKS[idx]["document"] == CHUNKS[g]["document"]
                                          for g in gold):
            return pos
    return None


def _share(n: int, total: int) -> str:
    if not total:
        return "0/0"
    low, high = wilson(n, total)
    return f"{n}/{total} ({n / total:.0%}, 95% CI {low:.0%}-{high:.0%})"


def report(args):
    out_dir = Path(args.out_dir)
    rows = json.load(open(out_dir / ANSWERS_NAME, encoding="utf-8"))
    try:
        labels = json.load(open(out_dir / LABELS_NAME, encoding="utf-8"))
    except json.JSONDecodeError as e:
        sys.exit(f"{LABELS_NAME}: грешка в JSON на ред {e.lineno}, позиция {e.colno}: {e.msg}.")
    labels = {k: v for k, v in labels.items() if not k.startswith("_")}
    missing = _unlabelled(labels)
    if missing:
        print("Липсващи или неразпознати присъди:\n  " + "\n  ".join(missing))
        sys.exit(f"\nНе са готови {len(missing)} въпроса; попълнете {out_dir / LABELS_NAME}.")
    in_scope = [r for r in rows if r["id"].startswith("F")]
    out_scope = [r for r in rows if r["id"].startswith("O")]
    result = {"questions": {"in_scope": len(in_scope), "out_of_scope": len(out_scope)}}

    print(f"Извличане ({len(in_scope)} въпроса в обхвата; строго / с откъсите съседи):")
    print(f"{'метод':16} | R@1   R@5   MRR  | R@1   R@5   MRR")
    retrieval, strict_ranks = {}, {}
    for method in in_scope[0]["rankings"]:
        m = {}
        for mode, lenient in (("strict", False), ("lenient", True)):
            ranks = [_hit(r["rankings"][method], r["gold"], 10, lenient) for r in in_scope]
            m[mode] = {"R@1": sum(p is not None and p < 1 for p in ranks) / len(ranks),
                       "R@5": sum(p is not None and p < 5 for p in ranks) / len(ranks),
                       "MRR": sum(1 / (p + 1) for p in ranks if p is not None) / len(ranks)}
            if not lenient:
                strict_ranks[method] = ranks
        retrieval[method] = m
        s, l = m["strict"], m["lenient"]
        print(f"{method:16} | {s['R@1']:.2f}  {s['R@5']:.2f}  {s['MRR']:.2f} | {l['R@1']:.2f}  {l['R@5']:.2f}  {l['MRR']:.2f}")
    result["retrieval"] = retrieval

    print("\nЗначимост (строго; при R@1 и R@5: случаи, в които само A / само B намира верния откъс, точен тест "
          "на Макнемар; при MRR: рандомизационен тест):")
    result["significance"] = {}
    for a, b in [(a, b) for a, b in COMPARISONS if a in strict_ranks and b in strict_ranks]:
        ra, rb = strict_ranks[a], strict_ranks[b]
        test = {}
        for k in (1, 5):
            test[f"R@{k}"] = mcnemar([p is not None and p < k for p in ra], [p is not None and p < k for p in rb])
        test["MRR"] = randomization([1 / (p + 1) if p is not None else 0.0 for p in ra],
                                    [1 / (p + 1) if p is not None else 0.0 for p in rb])
        print(f"  A = {a:7} B = {b:15} | R@1 {test['R@1'][0]}/{test['R@1'][1]}, p = {test['R@1'][2]:.3f} | "
              f"R@5 {test['R@5'][0]}/{test['R@5'][1]}, p = {test['R@5'][2]:.3f} | MRR p = {test['MRR']:.3f}")
        result["significance"][f"{a} vs {b}"] = test

    shown = sum(bool(set(r["gold"]) & {CHUNK_INDEX[c] for c in r["chunk_ids"]}) for r in in_scope)
    print(f"\nОткъсът с отговора е показан на модела: {_share(shown, len(in_scope))}")
    result["gold_shown"] = shown

    false_refusals = sum(not r["chunk_ids"] for r in in_scope)
    refused_out = sum(not r["chunk_ids"] for r in out_scope)
    print(f"\nОткази: погрешни при въпросите в обхвата {false_refusals}/{len(in_scope)}, "
          f"правилни при въпросите извън обхвата {refused_out}/{len(out_scope)}")
    for r in out_scope:
        verdict = "отказ" if not r["chunk_ids"] else labels[r["id"]]["answer_verdict"]
        print(f"  {r['id']} ({r['kind']}): най-висока оценка {r['rerank_scores'][0]:.3f} -> {verdict}")
    result["refusals"] = {"false_in_scope": false_refusals, "correct_out_of_scope": refused_out,
                          "out_of_scope": {r["id"]: labels[r["id"]].get("answer_verdict", "refused") for r in out_scope}}

    verdicts = [labels[r["id"]]["answer_verdict"] for r in in_scope]
    counts = {v: verdicts.count(v) for v in ("complete", "partial", "wrong", "refused")}
    facts = [f["present"] for r in in_scope for f in labels[r["id"]]["facts"]]
    print(f"\nОтговори: верни и пълни {counts['complete']}, верни, но непълни {counts['partial']}, "
          f"с невярно твърдение {counts['wrong']}, отказ {counts['refused']}")
    print(f"Ключови факти в отговорите: {_share(sum(facts), len(facts))}")
    result["answers"] = counts
    result["facts"] = {"present": sum(facts), "total": len(facts)}

    flags = [f["real_error"] for e in labels.values() for f in e.get("flags", [])]
    missed = sum(len(e.get("missed_errors", [])) for e in labels.values())
    answered = sum("flags" in e for e in labels.values())
    print(f"\nПроверка за неподкрепени подробности ({answered} отговора): сигнали {len(flags)}, "
          f"истински грешки {_share(sum(flags), len(flags))}")
    print(f"Пропуснати от проверката грешки: {missed}; дял на откритите сред всички забелязани: "
          f"{_share(sum(flags), sum(flags) + missed)}")
    result["grounding"] = {"answers": answered, "flags": len(flags), "real": sum(flags), "missed": missed}

    links = [l["verdict"] for r in in_scope for l in labels[r["id"]].get("links", [])]
    unmatched = [u["verdict"] for r in in_scope for u in labels[r["id"]].get("unmatched", [])]
    if links and all(v in LINK_VERDICTS for v in links) and all(v in UNMATCHED_VERDICTS for v in unmatched):
        print(f"\nМаркиране на източниците: верни връзки {_share(links.count('correct'), len(links))}, "
              f"грешни {links.count('wrong')}, неясни {links.count('unclear')}")
        print(f"Изречения без съответствие: {len(unmatched)} " + str({v: unmatched.count(v) for v in UNMATCHED_VERDICTS}))
        result["attribution"] = {"links": {v: links.count(v) for v in LINK_VERDICTS},
                                 "unmatched": {v: unmatched.count(v) for v in UNMATCHED_VERDICTS}}
    else:
        print(f"\nМаркиране на източниците: {len(links)} връзки, не всички са оценени (по желание)")
    used = sum(r.get("sources_used", 0) for r in in_scope)
    total_sources = sum(len(r["chunk_ids"]) for r in in_scope if r.get("answer") is not None)
    print(f"Използвани в отговора откъси: {used} от {total_sources}")

    retrieval_s = [r["retrieval_seconds"] for r in rows[1:]]  # the first includes loading the models
    generation_s = [r["generation_seconds"] for r in rows if "generation_seconds" in r]
    print(f"\nВреме: извличане медиана {statistics.median(retrieval_s):.1f} s, "
          f"генериране медиана {statistics.median(generation_s):.1f} s (макс. {max(generation_s):.1f} s)")
    result["seconds"] = {"retrieval_median": statistics.median(retrieval_s),
                         "generation_median": statistics.median(generation_s), "generation_max": max(generation_s)}
    path = Path(args.report_out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print("saved", path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["check", "run", "report"])
    parser.add_argument("--questions", default=str(QUESTIONS_PATH))
    parser.add_argument("--out-dir", default=str(THESIS_RESULTS_DIR), help="where run writes and report reads")
    parser.add_argument("--report-out", default=str(THESIS_RESULTS_DIR.parent / "final_report.json"))
    parser.add_argument("--force", action="store_true", help="let run overwrite earlier answers and verdicts")
    args = parser.parse_args()
    {"check": check, "run": run, "report": report}[args.step](args)


if __name__ == "__main__":
    main()
