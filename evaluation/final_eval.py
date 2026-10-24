"""Final evaluation on 59 independently written questions and 10 outside the guidelines (thesis section
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
# The Ollama runner keeps every prompt's state in RAM (~0.5 GB for 5,500 tokens); unloading the model this
# often keeps it from running out of memory ("model runner has unexpectedly stopped", std::bad_alloc).
UNLOAD_EVERY = 8
OTHER_EMBEDDINGS = {"intfloat/multilingual-e5-base": "e5-base",
                    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": "mpnet"}
# Pairs compared for significance, as in the thesis table "Значимост на основните разлики при извличането".
COMPARISONS = [("bge-m3", "e5-base"), ("bge-m3", "mpnet"), ("bge-m3", "BM25"), ("bge-m3", "RRF"),
               ("RRF", "RRF + reranker"), ("bge-m3", "RRF + reranker")]

ANSWER_VERDICTS = {"complete", "partial", "wrong"}
# Only for "wrong": the error is in what the question asks (core) or in a side detail of a correct answer (minor).
ERROR_SEVERITIES = {"core", "minor"}
OUT_OF_SCOPE_VERDICTS = {"declined", "fabricated"}
# Tuples, not sets: the report lists the counts in this order.
LINK_VERDICTS = ("correct", "wrong", "unclear")
UNMATCHED_VERDICTS = ("unsupported", "missed_match", "unclear")

# OCR mixes Latin lookalikes into Cyrillic words ("А02ВА02" with Latin A or B); both sides are unified so a
# quote copied from the PDF matches the extracted text.
_HOMOGLYPHS = str.maketrans("ABCEHKMOPTXaceopxy", "АВСЕНКМОРТХасеорху")


def _words(text: str) -> list[str]:
    return re.findall(r"\w+", text.translate(_HOMOGLYPHS).lower())


_CHUNK_WORDS, _PAGE_WORDS = {}, {}
_DOCUMENTS = {c["document"] for c in CHUNKS}


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


def _chunk_list(chunks: list[int]) -> str:
    return ", ".join(f"{CHUNKS[i]['chunk_id']} ({_pages(i)})" for i in chunks)


def _quote_texts(item: dict) -> list[str]:
    return [q for q in _as_list(item.get("quote")) if isinstance(q, str) and q.strip()]


def _quotes(item: dict) -> list[tuple[str, int]]:
    """(quote, page) pairs: an answer spread over several places has a list of quotes, with one page for all
    of them or a list of pages of the same length."""
    quotes, pages = _quote_texts(item), _as_list(item.get("page"))
    return list(zip(quotes, pages * len(quotes) if len(pages) == 1 else pages))


def _unfilled(item: dict) -> list[str]:
    fields = {"question": item.get("question", "").strip(), "page": _as_list(item.get("page")),
              "quote": _quote_texts(item), "facts": [f for f in item.get("facts", []) if f.strip()]}
    return [name for name, value in fields.items() if not value]


def _pages_valid(item: dict) -> bool:
    pages = _as_list(item.get("page"))
    return all(isinstance(p, int) for p in pages) and len(pages) in (1, len(_quote_texts(item)))


def _check_quote(label: str, part: dict) -> tuple[list[int], list[int], str | None, list[str]]:
    """One quote's gold chunks, chunks of other guidelines with the same text, why it fails (None if it does
    not) and warnings."""
    page = part["page"]
    if len(_words(part["quote"])) < QUOTE_MIN_WORDS:
        return [], [], f"{label} е твърде кратък: копирайте поне {QUOTE_MIN_WORDS} поредни думи", []
    found, same_text, best, score = find_gold(part)
    if not found and _lost_in_chunking(part):
        return [], [], (f"{label} е на стр. {page} в PDF-а, но този текст не е влязъл в откъсите на системата "
                        "(част от таблиците не е извлечена); изберете друго място"), []
    if not found:
        return [], [], (f"{label} не е намерен в {part['document']}; най-близо е {_chunk_list([best])} "
                        f"с {score:.0%} от думите"), []
    notes = []
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
        notes.append(f"същият текст има и в {_chunk_list(same_text)}; и той се брои за верен източник")
    return found, same_text, None, notes


def _check_also_correct(extra: dict, doc_names: dict[str, str]) -> tuple[list[int], str | None, str | None]:
    """A passage of another guideline that answers the question too, e.g. the same dose written differently in
    a newer version: its chunks, why it fails (None if it does not) and a note."""
    document = doc_names.get(extra.get("document", ""), extra.get("document", ""))
    part = {"document": document, "quote": extra.get("quote", ""), "page": extra.get("page")}
    if (document not in _DOCUMENTS or not isinstance(part["page"], int)
            or len(_words(part["quote"])) < QUOTE_MIN_WORDS):
        return [], ("also_correct изисква document (ръководство, както е в guideline), page (число) и "
                    f"quote (поне {QUOTE_MIN_WORDS} думи)"), None
    found, same_text, best, score = find_gold(part)
    if not found:
        return [], (f"цитатът в also_correct не е намерен в {document}; най-близо е {_chunk_list([best])} "
                    f"с {score:.0%} от думите"), None
    return found + same_text, None, f"верен е и {_chunk_list(found)} (also_correct)"


def _check_sources(item: dict, doc_names: dict[str, str]) -> tuple[list[int], list[int], list[str], list[str]]:
    """Gold chunks, equally correct chunks of other guidelines, failures and warnings of an in-scope question."""
    gold, copies, failed, notes = [], [], [], []
    quotes = _quotes(item)
    for n, (quote, page) in enumerate(quotes, 1):
        label = f"цитат {n}" if len(quotes) > 1 else "цитатът"
        part = {"document": item["document"], "quote": quote, "page": page}
        found, same_text, failure, warnings = _check_quote(label, part)
        if failure:
            failed.append(failure)
            continue
        notes += warnings
        gold += [g for g in found if g not in gold]
        copies += [g for g in same_text if g not in copies]
    for extra in item.get("also_correct", []):
        found, failure, note = _check_also_correct(extra, doc_names)
        if failure:
            failed.append(failure)
            continue
        notes.append(note)
        copies += [g for g in found if g not in copies]
    return gold, copies, failed, notes


def _overlap_warnings(item: dict, gold: list[int], used: dict[int, str]) -> list[str]:
    """Gold chunks already used by the development set or by an earlier question, and an unusual fact count."""
    warnings = []
    if set(GOLD) & set(gold):
        warnings.append("същият откъс е използван в набора за разработка; изберете друго място")
    earlier = sorted({used[g] for g in gold if g in used})
    if earlier:
        warnings.append(f"същият откъс като при {', '.join(earlier)}; въпросите трябва да са за "
                        "различни места в ръководството")
    facts = [f for f in item["facts"] if f.strip()]
    if not 2 <= len(facts) <= 4:
        count = "1 ключов факт" if len(facts) == 1 else f"{len(facts)} ключови факта"
        warnings.append(f"{count}; препоръчват се 2-4")
    return warnings


def inspect(questions: dict) -> tuple[list[str], int, dict[str, list[int]]]:
    """Report lines, number of problems that block the run, and the gold chunks of every complete question."""
    lines, problems, gold_of, empty = [], 0, {}, []
    used = {}  # gold chunk -> the question that has it
    doc_names = {i["guideline"]: i["document"] for i in questions["in_scope"]}
    for item in questions["in_scope"]:
        qid, unfilled = item["id"], _unfilled(item)
        if len(unfilled) == 4:
            empty.append(qid)
            problems += 1
            continue
        if unfilled:
            lines.append(f"{qid} не е попълнен: {', '.join(unfilled)}")
            problems += 1
            continue
        if not _pages_valid(item):
            lines.append(f"{qid} page трябва да е число без кавички, а при няколко цитата от различни страници "
                         "– списък с по една страница за всеки цитат, напр. [85, 86]")
            problems += 1
            continue
        gold, copies, failed, notes = _check_sources(item, doc_names)
        if failed:
            lines += [f"{qid} {f}" for f in failed]
            problems += 1
            continue
        gold_of[qid] = gold + [g for g in copies if g not in gold]
        lines.append(f"{qid} наред: {_chunk_list(gold)}")
        lines += [f"    {n}" for n in dict.fromkeys(notes)]
        lines += [f"    внимание: {w}" for w in _overlap_warnings(item, gold, used)]
        used.update({g: qid for g in gold})
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
    ready = f"Готови: {total - problems} от {total}."
    print(f"\n{ready}" + ("" if problems else " Може да се пусне run (със спряно приложение)."))


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


def _refused(row: dict) -> bool:
    """No answer: no chunk passed the reranker threshold, or the key-concept check stopped the question."""
    return not row["chunk_ids"] or not row.get("coverage", {}).get("answerable", True)


def check_coverage(rows: list[dict]):
    """The app's key-concept check (generation/coverage.py) on the retrieved chunks, before generation."""
    from medguiderag.generation.coverage import CoverageChecker

    checker = CoverageChecker([c["text"] for c in CHUNKS])
    for row in rows:
        if row["chunk_ids"]:
            c = checker.check(row["question"], chunks_by_id(row["chunk_ids"]))
            row["coverage"] = {"answerable": c.answerable, "reason": c.reason, "terms": list(c.terms)}
            if not c.answerable:
                print(f"{row['id']} stopped by the key-concept check ({c.reason}: {', '.join(c.terms)})", flush=True)


def generate(rows: list[dict]):
    import ollama

    from medguiderag.config import OLLAMA_MODEL
    from medguiderag.generation.generator import generate_answer

    answered = [r for r in rows if not _refused(r)]
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
    "error_severity": "само при wrong: core (грешката е в това, което въпросът пита: лекарство, доза, показание, "
                      "популация, продължителност) | minor (основният отговор е верен, грешката е в странична "
                      "подробност); иначе null",
    "present, real_error": "true | false",
    "facts.relevant": "по желание: false, ако фактът не се отнася пряко за въпроса; тогава не се брои за изчерпателността",
    "missed_errors": "неподкрепени подробности от отговора, които проверката не е отбелязала, напр. [\"8-16 mg/kg\"]",
    "links.verdict": "correct | wrong | unclear (по желание)",
    "unmatched.verdict": "unsupported (наистина не е в откъсите) | missed_match (има го, но не е свързано) | unclear "
                         "(по желание)",
}


def labels_template(rows: list[dict]) -> dict:
    labels = {"_възможни_стойности": LABELS_GUIDE}
    for row in rows:
        refused = _refused(row)
        entry = {"question": row["question"], "answer": _answer_lines(row.get("answer"))}
        if row["id"].startswith("F"):
            entry["answer_verdict"] = "refused" if refused else None
            if not refused:
                entry["error_severity"] = None
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
        if _refused(row):
            out += ["", "**Отказ от проверката на ключовите понятия** "
                    f"({row['coverage']['reason']}: {', '.join(row['coverage']['terms'])}); отговор не е генериран.", ""]
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
    check_coverage(rows)
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


def _unlabelled(labels: dict) -> list[str]:
    """Per question, the required verdicts still missing or misspelled; the link verdicts are optional."""
    out = []
    for qid, e in labels.items():
        allowed = ANSWER_VERDICTS | {"refused"} if qid.startswith("F") else OUT_OF_SCOPE_VERDICTS
        missing = []
        if "answer_verdict" in e and e["answer_verdict"] not in allowed:
            missing.append(f"answer_verdict ({' | '.join(sorted(allowed - {'refused'}))})")
        if e.get("answer_verdict") == "wrong" and e.get("error_severity") not in ERROR_SEVERITIES:
            missing.append(f"error_severity ({' | '.join(sorted(ERROR_SEVERITIES))})")
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


def _refresh_gold(rows: list[dict], questions_path: Path):
    """Gold chunks from the current question file, so a passage accepted as correct after reviewing the shown
    chunks (also_correct) counts without generating the answers again."""
    _, _, gold_of = inspect(load_questions(questions_path))
    changed = [r["id"] for r in rows if r["id"] in gold_of and set(gold_of[r["id"]]) != set(r.get("gold", []))]
    for r in rows:
        r["gold"] = gold_of.get(r["id"], r.get("gold", []))
    if changed:
        print(f"Верните откъси са взети от {questions_path.name}; променени спрямо пускането: {', '.join(changed)}\n")


def _load_labels(path: Path) -> dict:
    try:
        labels = json.load(open(path, encoding="utf-8"))
    except json.JSONDecodeError as e:
        sys.exit(f"{path.name}: грешка в JSON на ред {e.lineno}, позиция {e.colno}: {e.msg}.")
    return {k: v for k, v in labels.items() if not k.startswith("_")}


def _found_within(ranks: list[int | None], k: int) -> list[bool]:
    return [p is not None and p < k for p in ranks]


def _reciprocal(ranks: list[int | None]) -> list[float]:
    return [1 / (p + 1) if p is not None else 0.0 for p in ranks]


def _report_retrieval(in_scope: list[dict], result: dict) -> dict[str, list[int | None]]:
    """Recall and MRR of every method, strict and with neighbours; returns the strict gold ranks per method."""
    print(f"Извличане ({len(in_scope)} въпроса в обхвата; строго / с откъсите съседи):")
    print(f"{'метод':16} | R@1   R@5   MRR  | R@1   R@5   MRR")
    result["retrieval"], strict_ranks = {}, {}
    for method in in_scope[0]["rankings"]:
        m = {}
        for mode, lenient in (("strict", False), ("lenient", True)):
            ranks = [_hit(r["rankings"][method], r["gold"], 10, lenient) for r in in_scope]
            n = len(ranks)
            m[mode] = {"R@1": sum(_found_within(ranks, 1)) / n, "R@5": sum(_found_within(ranks, 5)) / n,
                       "MRR": sum(_reciprocal(ranks)) / n}
            if not lenient:
                strict_ranks[method] = ranks
        result["retrieval"][method] = m
        s, l = m["strict"], m["lenient"]
        print(f"{method:16} | {s['R@1']:.2f}  {s['R@5']:.2f}  {s['MRR']:.2f} | "
              f"{l['R@1']:.2f}  {l['R@5']:.2f}  {l['MRR']:.2f}")
    return strict_ranks


def _report_significance(strict_ranks: dict[str, list[int | None]], result: dict):
    print("\nЗначимост (строго; при R@1 и R@5: случаи, в които само A / само B намира верния откъс, точен тест "
          "на Макнемар; при MRR: рандомизационен тест):")
    result["significance"] = {}
    for a, b in [(a, b) for a, b in COMPARISONS if a in strict_ranks and b in strict_ranks]:
        ra, rb = strict_ranks[a], strict_ranks[b]
        test = {f"R@{k}": mcnemar(_found_within(ra, k), _found_within(rb, k)) for k in (1, 5)}
        test["MRR"] = randomization(_reciprocal(ra), _reciprocal(rb))
        print(f"  A = {a:7} B = {b:15} | R@1 {test['R@1'][0]}/{test['R@1'][1]}, p = {test['R@1'][2]:.3f} | "
              f"R@5 {test['R@5'][0]}/{test['R@5'][1]}, p = {test['R@5'][2]:.3f} | MRR p = {test['MRR']:.3f}")
        result["significance"][f"{a} vs {b}"] = test


def _outcome(row: dict, labels: dict) -> str:
    if not row["chunk_ids"]:
        return "отказ (праг)"
    if _refused(row):
        return f"отказ ({row['coverage']['reason']}: {', '.join(row['coverage']['terms'])})"
    return labels[row["id"]].get("answer_verdict") or "unlabelled"


def _report_refusals(rows: list[dict], in_scope: list[dict], out_scope: list[dict], labels: dict, result: dict):
    false_refusals = sum(_refused(r) for r in in_scope)
    refused_out = sum(_refused(r) for r in out_scope)
    by_threshold = sum(not r["chunk_ids"] for r in out_scope)
    print(f"\nОткази: погрешни при въпросите в обхвата {false_refusals}/{len(in_scope)}, "
          f"правилни при въпросите извън обхвата {refused_out}/{len(out_scope)} "
          f"(праг {by_threshold}, проверка на ключовите понятия {refused_out - by_threshold})")
    out_verdicts = {r["id"]: _outcome(r, labels) for r in out_scope}
    for r in out_scope:
        print(f"  {r['id']} ({r['kind']}): най-висока оценка {r['rerank_scores'][0]:.3f} -> {out_verdicts[r['id']]}")
    for r in in_scope:
        if _refused(r):
            print(f"  погрешен отказ {r['id']}: {_outcome(r, labels)}, най-висока оценка {r['rerank_scores'][0]:.3f}")
    result["refusals"] = {"false_in_scope": false_refusals, "correct_out_of_scope": refused_out,
                          "out_of_scope_by_threshold": by_threshold, "out_of_scope": out_verdicts,
                          "top_scores": {r["id"]: r["rerank_scores"][0] for r in rows}}


def _report_grounding_auto(rows: list[dict], result: dict):
    answered = [r for r in rows if r.get("answer") is not None]
    flagged = sum(bool(r["grounding_issues"]) for r in answered)
    flags = sum(len(r["grounding_issues"]) for r in answered)
    print(f"\nПроверка за неподкрепени подробности: сигнали в {flagged} от {len(answered)} отговора, общо {flags}")
    result["grounding_auto"] = {"answers": len(answered), "flagged_answers": flagged, "flags": flags}
    loops = sum(r["answer"].endswith("[…]") for r in answered)
    print(f"Прекъснати отговори (зацикляне): {loops}")
    result["cut_answers"] = loops


def report(args):
    out_dir = Path(args.out_dir)
    rows = json.load(open(out_dir / ANSWERS_NAME, encoding="utf-8"))
    _refresh_gold(rows, Path(args.questions))
    labels = _load_labels(out_dir / LABELS_NAME)
    missing = _unlabelled(labels)
    # Retrieval, refusals and timings need no verdicts; the answer-quality parts wait until all are filled in.
    labelled = not missing
    if missing:
        print(f"Ръчните присъди не са готови за {len(missing)} въпроса (виж {LABELS_NAME}); "
              "отчитат се само автоматичните части.\n")
    in_scope = [r for r in rows if r["id"].startswith("F")]
    out_scope = [r for r in rows if r["id"].startswith("O")]
    result = {"questions": {"in_scope": len(in_scope), "out_of_scope": len(out_scope)}}

    strict_ranks = _report_retrieval(in_scope, result)
    _report_significance(strict_ranks, result)
    shown = sum(bool(set(r["gold"]) & {CHUNK_INDEX[c] for c in r["chunk_ids"]}) for r in in_scope)
    print(f"\nОткъсът с отговора е показан на модела: {_share(shown, len(in_scope))}")
    result["gold_shown"] = shown
    _report_refusals(rows, in_scope, out_scope, labels, result)
    _report_grounding_auto(rows, result)
    if labelled:
        _report_answers(in_scope, labels, result)
        _report_grounding_verdicts(labels, result)
    _report_links_and_time(rows, in_scope, labels, result, labelled)
    _save_report(args, result)


def _report_answers(in_scope: list[dict], labels: dict, result: dict):
    """Verdicts on the in-scope answers, strict and lenient, and the share of key facts they contain."""
    verdicts = [labels[r["id"]]["answer_verdict"] for r in in_scope]
    counts = {v: verdicts.count(v) for v in ("complete", "partial", "wrong", "refused")}
    severities = [labels[r["id"]]["error_severity"] for r in in_scope if labels[r["id"]]["answer_verdict"] == "wrong"]
    core, minor = severities.count("core"), severities.count("minor")
    answered_in = len(in_scope) - counts["refused"]
    # A fact marked "relevant": false does not answer the question as asked and is left out of the share.
    facts = [f["present"] for r in in_scope for f in labels[r["id"]]["facts"] if f.get("relevant", True)]
    print(f"\nОтговори, строго (всяко невярно твърдение е грешка): верни и пълни {counts['complete']}, "
          f"верни, но непълни {counts['partial']}, с невярно твърдение {counts['wrong']}, отказ {counts['refused']}")
    print(f"Отговори, меко (грешка е само грешката по същество): верни и пълни {counts['complete']}, "
          f"непълни или със странична неточност {counts['partial'] + minor}, с грешка по същество {core}")
    print(f"Вярност ({answered_in} отговора): без невярно твърдение "
          f"{_share(counts['complete'] + counts['partial'], answered_in)}, странична неточност "
          f"{_share(minor, answered_in)}, грешка по същество {_share(core, answered_in)}")
    print(f"Изчерпателност (ключови факти в отговорите): {_share(sum(facts), len(facts))}")
    result["answers"] = counts
    result["answers_lenient"] = {"complete": counts["complete"], "partial": counts["partial"] + minor,
                                 "wrong": core, "refused": counts["refused"]}
    result["correctness"] = {"no_error": counts["complete"] + counts["partial"], "minor": minor, "core": core}
    result["facts"] = {"present": sum(facts), "total": len(facts)}


def _report_grounding_verdicts(labels: dict, result: dict):
    flags = [f["real_error"] for e in labels.values() for f in e.get("flags", [])]
    missed = sum(len(e.get("missed_errors", [])) for e in labels.values())
    answered = sum("flags" in e for e in labels.values())
    print(f"\nПроверка за неподкрепени подробности ({answered} отговора): сигнали {len(flags)}, "
          f"истински грешки {_share(sum(flags), len(flags))}")
    print(f"Пропуснати от проверката грешки: {missed}; дял на откритите сред всички забелязани: "
          f"{_share(sum(flags), sum(flags) + missed)}")
    result["grounding"] = {"answers": answered, "flags": len(flags), "real": sum(flags), "missed": missed}


def _report_links_and_time(rows: list[dict], in_scope: list[dict], labels: dict, result: dict, labelled: bool):
    answered = [r for r in in_scope if r.get("answer") is not None]
    n_links = sum(len(r["links"]) for r in answered)
    n_unmatched = sum(len(r["unmatched"]) for r in answered)
    print(f"\nМаркиране на източниците: {n_links} връзки, {n_unmatched} изречения без съответствие")
    result["attribution_auto"] = {"links": n_links, "unmatched": n_unmatched}
    links = [l["verdict"] for r in in_scope for l in labels[r["id"]].get("links", [])]
    unmatched = [u["verdict"] for r in in_scope for u in labels[r["id"]].get("unmatched", [])]
    if labelled and links and all(v in LINK_VERDICTS for v in links) and all(v in UNMATCHED_VERDICTS for v in unmatched):
        print(f"Верни връзки {_share(links.count('correct'), len(links))}, "
              f"грешни {links.count('wrong')}, неясни {links.count('unclear')}")
        print(f"Изречения без съответствие: {len(unmatched)} " + str({v: unmatched.count(v) for v in UNMATCHED_VERDICTS}))
        result["attribution"] = {"links": {v: links.count(v) for v in LINK_VERDICTS},
                                 "unmatched": {v: unmatched.count(v) for v in UNMATCHED_VERDICTS}}
    else:
        print("Присъдите за връзките не са попълнени (по желание)")
    used = sum(r.get("sources_used", 0) for r in answered)
    total_sources = sum(len(r["chunk_ids"]) for r in answered)
    print(f"Използвани в отговора откъси: {used} от {total_sources}")
    result["sources_used"] = {"used": used, "shown": total_sources}

    retrieval_s = [r["retrieval_seconds"] for r in rows[1:]]  # the first includes loading the models
    generation_s = [r["generation_seconds"] for r in rows if "generation_seconds" in r]
    print(f"\nВреме: извличане медиана {statistics.median(retrieval_s):.1f} s, "
          f"генериране медиана {statistics.median(generation_s):.1f} s (макс. {max(generation_s):.1f} s)")
    result["seconds"] = {"retrieval_median": statistics.median(retrieval_s),
                         "generation_median": statistics.median(generation_s), "generation_max": max(generation_s)}


def _save_report(args, result: dict):
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
