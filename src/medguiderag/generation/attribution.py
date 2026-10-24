"""Link the sentences of a generated answer to the chunk sentences that support them."""
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Callable

import numpy as np

from medguiderag.generation.grounding import NUMBER, WORD, number_values, skeletons

# A guideline sentence ends with ".", "!", "?" or ";" before a capital, a digit, a bullet or a dash; a bullet
# also starts one. "i.v. bolus" or "5 mg. дневно" stay whole because a lowercase word follows.
_CHUNK_BREAK = re.compile(r"(?<=[.!?;])\s+(?=[A-ZА-Я0-9•\-–(])|\s+(?=•)")
_ANSWER_BREAK = re.compile(r"(?<=[.!?])\s+(?=[A-ZА-Я0-9*(])|\n+")
# A break right after one of these is not a sentence end: "(стр. 17)", "напр. Esomeprazole".
_ABBREVIATION = re.compile(r"\b(?:стр|напр|вкл|табл|др|мин|гр|мл|мг|ед|т\.е|т\.нар)\.$")
_MARKDOWN = re.compile(r"\*\*|__|`|^#+\s*|^\s*(?:[-*•]|\d+[.)])\s+", re.M)
# An answer sentence may join facts from several chunk sentences; link up to this many.
_MAX_LINKS = 3
# The further links must score within this margin of the best one.
_SECONDARY_MARGIN = 0.08


@dataclass(frozen=True)
class Link:
    sentence: str        # answer sentence, markdown stripped
    chunk: int           # index into the chunks passed to attribute()
    span: tuple[int, int]  # character span in the chunk's whitespace-normalized text
    semantic: float
    lexical: float

    @property
    def score(self) -> float:
        return (self.semantic + self.lexical) / 2


@dataclass
class Attribution:
    links: list[Link]
    unsupported: list[str]                      # answer sentences with no chunk sentence above the threshold
    highlights: dict[int, list[tuple[int, int]]]  # chunk index -> spans to mark


def normalized_text(chunk: dict) -> str:
    """The chunk text as the UI shows it; attribution spans index into this string."""
    return " ".join(chunk["text"].split())


def _spans(text: str, breaks: re.Pattern) -> list[tuple[int, int]]:
    spans, start = [], 0
    for m in breaks.finditer(text):
        if "\n" not in m.group(0) and _ABBREVIATION.search(text, 0, m.start()):
            continue
        spans.append((start, m.start()))
        start = m.end()
    spans.append((start, len(text)))
    return [(s, e) for s, e in spans if text[s:e].strip()]


def _tokens(text: str) -> list[frozenset[str]]:
    """Content tokens as sets of spelling-independent keys: words of 5+ letters and numbers."""
    tokens = [frozenset(skeletons(w)) for w in WORD.findall(text) if len(w) >= 5]
    return tokens + [frozenset(number_values(m.group(0))) for m in NUMBER.finditer(text)]


def _lexical_scores(sentences: list[str], unit_texts: list[str]) -> np.ndarray:
    """Share of each answer sentence's content tokens found in each chunk sentence, every token weighted by its
    rarity among the chunk sentences, so "лечение" counts little and "ondansetron" much."""
    unit_keys = [set().union(*_tokens(t)) for t in unit_texts]
    chunk_sentences_with = Counter(k for keys in unit_keys for k in keys)
    n = len(unit_texts)
    scores = np.zeros((len(sentences), n))
    for a, sentence in enumerate(sentences):
        tokens = _tokens(sentence)
        weights = [math.log((n + 1) / (1 + min(chunk_sentences_with[k] for k in tok))) + 1 for tok in tokens]
        total = sum(weights)
        if not total:
            continue
        for u, keys in enumerate(unit_keys):
            scores[a, u] = sum(w for tok, w in zip(tokens, weights) if tok & keys) / total
    return scores


class Attributor:
    # Set on the 4B answers to the 24 development questions: below 0.45 the best match was mostly wrong or the
    # sentence was a list introduction; from 0.45 up about 9 in 10 matches support the answer sentence.
    def __init__(self, embed: Callable[[list[str]], np.ndarray], threshold: float = 0.45):
        # embed returns L2-normalized vectors, so a dot product is the cosine similarity.
        self.embed = embed
        self.threshold = threshold

    def score_matrix(self, answer: str, chunks: list[dict]):
        """Answer sentences, chunk sentences (chunk index, span) and their semantic and lexical similarities."""
        sentences = [_MARKDOWN.sub("", answer[s:e]).strip() for s, e in _spans(answer, _ANSWER_BREAK)]
        # "Лечението включва:" only introduces a list; it states nothing to find in the chunks.
        sentences = [s for s in sentences if len(WORD.findall(s)) >= 3 and not s.endswith(":")]
        texts = [normalized_text(c) for c in chunks]
        units = [(i, span) for i, t in enumerate(texts) for span in _spans(t, _CHUNK_BREAK)]
        if not sentences or not units:
            empty = np.zeros((len(sentences), len(units)))
            return sentences, units, empty, empty
        unit_texts = [texts[i][s:e] for i, (s, e) in units]
        vectors = self.embed(sentences + unit_texts)
        semantic = vectors[: len(sentences)] @ vectors[len(sentences):].T
        return sentences, units, semantic, _lexical_scores(sentences, unit_texts)

    def attribute(self, answer: str, chunks: list[dict]) -> Attribution:
        sentences, units, semantic, lexical = self.score_matrix(answer, chunks)
        links, unsupported = [], []
        for a, sentence in enumerate(sentences):
            combined = (semantic[a] + lexical[a]) / 2
            if not combined.size or combined.max() < self.threshold:
                unsupported.append(sentence)
                continue
            best = combined.max()
            for u in np.argsort(-combined)[:_MAX_LINKS]:
                if combined[u] < max(self.threshold, best - _SECONDARY_MARGIN):
                    break
                chunk, span = units[u]
                links.append(Link(sentence, chunk, span, float(semantic[a, u]), float(lexical[a, u])))
        highlights = {}
        for link in links:
            spans = highlights.setdefault(link.chunk, [])
            if link.span not in spans:
                spans.append(link.span)
        return Attribution(links, unsupported, {c: sorted(s) for c, s in highlights.items()})
