from functools import lru_cache

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

E5_PREFIX_MODELS = ("multilingual-e5",)


@lru_cache(maxsize=None)
def _load_model(model_name: str) -> SentenceTransformer:
    if torch.cuda.is_available():
        # Half precision halves VRAM (bge-m3: ~1.1 GB) so it fits next to BgGPT on a 6 GB card.
        return SentenceTransformer(model_name, device="cuda", model_kwargs={"torch_dtype": torch.float16})
    return SentenceTransformer(model_name, device="cpu")


def _apply_prefix(model_name: str, texts: list[str], kind: str) -> list[str]:
    if any(marker in model_name for marker in E5_PREFIX_MODELS):
        prefix = "query: " if kind == "query" else "passage: "
        return [prefix + t for t in texts]
    return texts


def embed_texts(texts: list[str], model_name: str, kind: str = "passage") -> np.ndarray:
    model = _load_model(model_name)
    prefixed = _apply_prefix(model_name, texts, kind)
    embeddings = model.encode(
        prefixed,
        batch_size=8,
        normalize_embeddings=True,
        show_progress_bar=len(prefixed) > 32,
        convert_to_numpy=True,
    )
    return embeddings.astype("float32")
