import re

import numpy as np
from rank_bm25 import BM25Okapi

# Crude Bulgarian stemming: "лечение", "лечението", "лечения" all become "лечени".
STEM_LENGTH = 6


def tokenize(text: str) -> list[str]:
    return [t[:STEM_LENGTH] for t in re.findall(r"\w+", text.lower())]


class BM25Index:
    def __init__(self, texts: list[str]):
        self._bm25 = BM25Okapi([tokenize(t) for t in texts])

    def search(self, query: str, k: int) -> list[int]:
        scores = self._bm25.get_scores(tokenize(query))
        return np.argsort(-scores, kind="stable")[:k].tolist()
