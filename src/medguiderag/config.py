from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
CORPUS_DIR = DATA_DIR / "processed"
CHUNKS_PATH = CORPUS_DIR / "chunks.jsonl"
DOCUMENTS_PATH = CORPUS_DIR / "documents.jsonl"
INDEX_DIR = DATA_DIR / "index"
REPORTS_DIR = PROJECT_ROOT / "reports"

OLLAMA_MODEL = "hf.co/INSAIT-Institute/BgGPT-Gemma-3-4B-IT-GGUF:Q4_K_M"

# Generators selectable in the UI, default first; labels are what the sidebar shows.
GENERATOR_MODELS = {
    OLLAMA_MODEL: "BgGPT 4B (Gemma 3)",
    "hf.co/INSAIT-Institute/BgGPT-Gemma-2-2.6B-IT-v1.0-GGUF:Q4_K_M": "BgGPT 2.6B (Gemma 2)",
}

EMBEDDING_MODEL_CANDIDATES = [
    "BAAI/bge-m3",
    "intfloat/multilingual-e5-base",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
]

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
RERANK_CANDIDATES = 20
RERANK_MIN_SCORE = 0.1
# Key-concept check: a question word is specific when its stem occurs in at most this many chunks (~2%).
COVERAGE_MAX_CHUNKS = 30
