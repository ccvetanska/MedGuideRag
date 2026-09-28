import ollama

from medguiderag.config import OLLAMA_MODEL

# After the context since Gemma 2 has no system role; no citation request, as the 2.6B model fabricates sources.
INSTRUCTIONS = (
    "Отговори на въпроса само с информация от горните откъси от национални "
    "фармакотерапевтични ръководства. Не добавяй факти, които не са в откъсите."
)

# 8192 is Gemma 2's native window (Ollama's default 4096 truncates even 5 chunks); temperature 0 keeps answers repeatable.
# Greedy decoding looped on a single drug-list chunk (the same 4 lines until the context filled, ~2 min);
# repeat_penalty 1.2 stopped it without losing facts. num_predict bounds the time if a loop still happens.
GENERATION_OPTIONS = {"num_ctx": 8192, "temperature": 0, "repeat_penalty": 1.2, "num_predict": 1024}


def _format_context(chunks: list[dict]) -> str:
    parts = []
    # Most relevant chunk last, next to the question: the 2.6B model ignores context ~5000 tokens back.
    for c in reversed(chunks):
        source = f"{c['document']}, стр. {c['page_start']}-{c['page_end']}"
        # PDF line breaks arrive as "\n\n" between single words of justified lines.
        text = " ".join(c["text"].split())
        parts.append(f"[{source}]\n{text}")
    return "\n\n".join(parts)


def build_prompt(question: str, chunks: list[dict]) -> str:
    return f"Откъси:\n{_format_context(chunks)}\n\n{INSTRUCTIONS}\n\nВъпрос: {question}"


def generate_answer(question: str, chunks: list[dict], model: str = OLLAMA_MODEL) -> str:
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": build_prompt(question, chunks)}],
        options=GENERATION_OPTIONS,
    )
    answer = response["message"]["content"]
    if response["done_reason"] == "length":
        # Cut off by num_predict: mark it so the reader does not take a partial answer as complete.
        answer += " […]"
    return answer
