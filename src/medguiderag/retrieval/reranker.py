from functools import lru_cache

import torch
from sentence_transformers import CrossEncoder

from medguiderag.config import RERANKER_MODEL

# Chunks are ~860 tokens (median); 1024 covers most of them while attention cost grows quadratically.
MAX_LENGTH = 1024


@lru_cache(maxsize=None)
def _load_model() -> CrossEncoder:
    if torch.cuda.is_available():
        return CrossEncoder(RERANKER_MODEL, max_length=MAX_LENGTH, device="cuda",
                            model_kwargs={"torch_dtype": torch.float16})
    return CrossEncoder(RERANKER_MODEL, max_length=MAX_LENGTH, device="cpu")


def rerank_scores(query: str, texts: list[str]) -> list[float]:
    """Relevance of each text to the query in [0, 1] (sigmoid of the cross-encoder logit)."""
    return _load_model().predict([(query, t) for t in texts], batch_size=4).tolist()
