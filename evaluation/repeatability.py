"""Is temperature-0 generation repeatable? The same prompt 3 times back to back and 3 times interleaved with
another prompt, greedy and with repeat_penalty 1.2, with and without a fixed seed (thesis paragraph "Повторимост").

    python -m evaluation.repeatability [--generator 2.6B]

Uses the archived hybrid-retrieval chunks (results/thesis/dev_contexts_hybrid.json), so no PyTorch model
shares the process's memory with the Ollama runner.
"""
import argparse

import ollama

from evaluation.common import GENERATORS, chunks_by_id, load
from medguiderag.generation.generator import build_prompt

QUESTIONS = ["В каква доза се прилага кломифен цитрат за стимулиране на овулацията?", "Как се овладява екламптичен гърч?"]
OTHER = "Кога се прилага инсулин при захарен диабет тип 2?"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generator", default="2.6B", choices=list(GENERATORS))
    args = parser.parse_args()
    model = GENERATORS[args.generator]
    archived = load("thesis/dev_contexts_hybrid.json")
    contexts = {q: chunks_by_id(archived[q]) for q in QUESTIONS + [OTHER]}

    def ask(q, options):
        prompt = build_prompt(q, contexts[q])
        return ollama.chat(model=model, messages=[{"role": "user", "content": prompt}], options=options)["message"]["content"]

    base = {"num_ctx": 8192, "temperature": 0, "num_predict": 1024}
    for name, options in {"greedy": base, "penalty 1.2": {**base, "repeat_penalty": 1.2},
                          "penalty 1.2 + seed 42": {**base, "repeat_penalty": 1.2, "seed": 42}}.items():
        for q in QUESTIONS:
            back_to_back = [ask(q, options) for _ in range(3)]
            interleaved = []
            for _ in range(3):
                ask(OTHER, options)
                interleaved.append(ask(q, options))
            print(f"{name:22} | {q[:35]:35} | distinct answers: back to back {len(set(back_to_back))}/3, "
                  f"interleaved {len(set(interleaved))}/3, all six {len(set(back_to_back + interleaved))}", flush=True)
            # Unload after each question's 9 calls: the runner keeps every prompt's state in RAM (~0.5 GB for
            # 5,500 tokens) and on a 16 GB machine ran out ("model runner has unexpectedly stopped", std::bad_alloc).
            ollama.generate(model=model, keep_alive=0)


if __name__ == "__main__":
    main()
