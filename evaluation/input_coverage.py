"""How much of each chunk the embedding models see within their maximum input length (thesis table tab:truncation).

    python -m evaluation.input_coverage

Uses the models' tokenizers from the local Hugging Face cache; no model weights are loaded.
"""
import glob
import os
import statistics

from tokenizers import Tokenizer

from evaluation.common import CHUNKS

HUB = os.path.expanduser("~/.cache/huggingface/hub")
MAX_INPUT = {
    "BAAI/bge-m3": 8192,
    "intfloat/multilingual-e5-base": 512,
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": 128
}
# The hyperemesis chunk: the term sits deep inside it, past what the short-input models embed.
TARGET = next(c for c in CHUNKS if c["chunk_id"] == "ft-ginekologiq:000007")


def load_tokenizer(name: str) -> Tokenizer:
    path = glob.glob(
        os.path.join(HUB, "models--" + name.replace("/", "--"), "snapshots",
                     "*", "tokenizer.json"))[0]
    tokenizer = Tokenizer.from_file(path)
    tokenizer.no_truncation()
    return tokenizer


def main():
    char_pos = TARGET["text"].lower().find("hyperemesis")
    for name, limit in MAX_INPUT.items():
        tokenizer = load_tokenizer(name)
        prefix = "passage: " if "e5" in name else ""
        lengths = [
            len(tokenizer.encode(prefix + c["text"]).ids) for c in CHUNKS
        ]
        truncated = sum(n > limit for n in lengths)
        share = statistics.mean(min(n, limit) / n for n in lengths)
        term = len(
            tokenizer.encode(prefix + TARGET["text"][:char_pos],
                             add_special_tokens=False).ids)
        print(
            f"{name.split('/')[-1]:40} max input {limit:5} | chunk tokens median {statistics.median(lengths):.0f}, "
            f"max {max(lengths)} | truncated {truncated / len(lengths):.0%} | share embedded {share:.0%} | "
            f"'hyperemesis' at token {term}")


if __name__ == "__main__":
    main()
