from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent

PDP_DATA_DIR = WORKSPACE_ROOT / "Pdp" / "data" / "processed"
CHUNKS_PATH = PDP_DATA_DIR / "chunks.jsonl"
DOCUMENTS_PATH = PDP_DATA_DIR / "documents.jsonl"

DATA_DIR = PROJECT_ROOT / "data"
INDEX_DIR = DATA_DIR / "index"
REPORTS_DIR = PROJECT_ROOT / "reports"

OLLAMA_MODEL = "hf.co/INSAIT-Institute/BgGPT-Gemma-2-2.6B-IT-v1.0-GGUF:Q4_K_M"

EMBEDDING_MODEL_CANDIDATES = [
    "BAAI/bge-m3",
    "intfloat/multilingual-e5-base",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
]

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
RERANK_CANDIDATES = 20
RERANK_MIN_SCORE = 0.1
