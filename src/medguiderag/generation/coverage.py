"""Decide before generation whether the retrieved chunks can answer the question at all.

A medical question outside the guidelines still gets chunks that pass the reranker threshold (they share a drug,
an organ or a kind of treatment with it), and the generator then answers from them. The check compares the key
concepts of the question with the chunks:
- a drug-like word of the question that the corpus never mentions ("еренумаб") -> no answer;
- the question's specific words (rare in the corpus) all missing from the shown chunks ("бизопролол") -> no answer.
"""
import re
from collections import defaultdict
from dataclasses import dataclass

from medguiderag.config import COVERAGE_MAX_CHUNKS
from medguiderag.generation.grounding import INN_STEMS, WORD, cyrillic_forms, skeletons

# Question words and generic words of medical questions: frequent in questions, rare in guideline text.
_STOP = set("""какво каква какви какъв кои кой коя кое кога колко как защо дали може трябва има няма
които който която което между около според след преди при като както също така само още много
дава дават дадат даде прилага прилагат прави правят започва започват започне лекува лекуват
пускат пуска спира спират използва използват препоръчва препоръчват назначава
пациент пациенти пациента човек хора дете деца детето възрастен възрастни
лечение лечението лечения доза дозата дози дозите схема схемата първи първа първо линия""".split())
# Endings of Bulgarian verb forms and verbal nouns ("спадне", "целят", "постъпване"): rare in the guidelines
# but not concepts of the question.
_VERB_ENDINGS = ("не", "ят", "ат", "ва", "ли", "ше", "ме", "те")
# Endings of drug names besides the WHO INN stems ("семаглутид", "ескетамин").
_DRUG_ENDINGS = ("ин", "ид", "ол", "ат", "он", "ам", "ил", "ен", "ан")


@dataclass(frozen=True)
class Coverage:
    answerable: bool
    reason: str           # "" or "corpus" (not in any guideline) or "chunks" (not in the shown chunks)
    terms: tuple[str, ...]


def _forms(word: str) -> set[str]:
    w = word.lower()
    forms = ({w} | cyrillic_forms(w)) if re.search("[а-я]", w) else {w}
    return forms | skeletons(word)


def _prefixes(word: str) -> set[str]:
    """Stems cut to 4-6 letters in both scripts, so "натрият" finds "натрий" and "ризатриптан" finds "rizatriptan"."""
    return {f[:n] for f in _forms(word) for n in (4, 5, 6) if len(f) >= n}


def _keys(word: str) -> set[str]:
    return {f[:6] for f in _forms(word) if len(f) >= 4}


def _parts(text: str):
    for w in WORD.findall(text):
        for part in w.split("-"):
            if len(part) >= 4:
                yield part


def _content_words(question: str):
    for n, w in enumerate(WORD.findall(question)):
        # Eponyms ("Гилен-Баре", "Уилсън") are transliterated differently in the guidelines; abbreviations stay.
        if n and re.match("[А-Я]", w) and not w.isupper():
            continue
        for part in w.split("-"):
            low = part.lower()
            if len(part) < 4 or low in _STOP:
                continue
            if re.search("[а-я]", low) and not part.isupper() and low.endswith(_VERB_ENDINGS):
                continue
            yield part


def _drug_like(word: str) -> bool:
    stems = cyrillic_forms(word) if re.search("[а-яА-Я]", word) else {word.lower()}
    return len(word) >= 6 and any(s.endswith(INN_STEMS + _DRUG_ENDINGS) for s in stems)


class CoverageChecker:
    def __init__(self, corpus_texts: list[str], max_chunks: int = COVERAGE_MAX_CHUNKS):
        self.max_chunks = max_chunks
        chunks_of = defaultdict(set)
        for i, text in enumerate(corpus_texts):
            for part in _parts(text):
                for p in _prefixes(part):
                    chunks_of[p].add(i)
        # Number of corpus chunks that contain each stem.
        self.chunk_count = {p: len(ids) for p, ids in chunks_of.items()}

    def check(self, question: str, chunks: list[dict]) -> Coverage:
        shown = {p for c in chunks for part in _parts(c["text"]) for p in _prefixes(part)}
        specific, absent = [], []
        for word in _content_words(question):
            keys = _keys(word)
            count = max(self.chunk_count.get(k, 0) for k in keys)
            if count == 0:
                if _drug_like(word):
                    absent.append(word)
            elif count <= self.max_chunks:
                specific.append((word, keys))
        if absent:
            return Coverage(False, "corpus", tuple(absent))
        if specific and not any(keys & shown for _, keys in specific):
            return Coverage(False, "chunks", tuple(w for w, _ in specific))
        return Coverage(True, "", ())


def refusal_message(coverage: Coverage) -> str:
    """The UI text for a question the check stops (shown instead of an answer)."""
    terms = ", ".join(f"„{t}“" for t in coverage.terms)
    if coverage.reason == "corpus":
        return f"{terms} не се споменава в ръководствата, затова отговор не беше генериран."
    return (f"В намерените откъси не се споменава {terms}, затова отговор не беше генериран: "
            "ръководствата вероятно не покриват въпроса.")
