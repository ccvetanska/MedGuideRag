import json
from pathlib import Path
from typing import Iterator

from medguiderag.config import CHUNKS_PATH


def iter_chunks(path: Path = CHUNKS_PATH) -> Iterator[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - the corpus (chunks.jsonl, documents.jsonl) is expected in data/processed/."
        )
    with path.open(encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def load_chunks(path: Path = CHUNKS_PATH) -> list[dict]:
    return list(iter_chunks(path))
