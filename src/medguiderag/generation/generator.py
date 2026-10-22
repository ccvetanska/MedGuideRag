import re

import ollama

from medguiderag.config import OLLAMA_MODEL

# After the context since Gemma 2 and Gemma 3 have no system role; no citation request, as BgGPT fabricated
# sources when asked to cite.
INSTRUCTIONS = (
    "Отговори на въпроса само с информация от горните откъси от национални "
    "фармакотерапевтични ръководства. Не добавяй факти, които не са в откъсите."
)
# BgGPT 4B sometimes starts the answer by repeating the instruction (1 of 63 answers in the final evaluation).
_ECHO = re.compile(r"\s*" + r"\s+".join(map(re.escape, INSTRUCTIONS.split())) + r"\s*")

# 8192 holds five chunks and the answer (Ollama's default 4096 truncates even 5 chunks); temperature 0 keeps
# answers nearly repeatable (the GPU may still give two variants).
# Greedy decoding looped on a single drug-list chunk (the same 4 lines until the context filled, ~2 min);
# repeat_penalty 1.2 stopped it without losing facts. num_predict bounds the time if a loop still happens.
GENERATION_OPTIONS = {"num_ctx": 8192, "temperature": 0, "repeat_penalty": 1.2, "num_predict": 1024}


def _format_context(chunks: list[dict]) -> str:
    parts = []
    # Most relevant chunk last, next to the question: BgGPT 2.6B ignored context ~5000 tokens back.
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
    echo = _ECHO.match(answer)
    if echo:
        answer = answer[echo.end():]
    if response["done_reason"] == "length":
        # Cut off by num_predict: mark it so the reader does not take a partial answer as complete.
        answer += " […]"
    return answer
