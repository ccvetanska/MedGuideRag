"""The thesis charts, drawn from the archived results into docs/figures/ as vector PDFs.

    python -m evaluation.figures

- threshold_calibration.pdf: top reranker score per development and out-of-scope question, with the
  refusal threshold (section 4.3, "Праг за отказ").
- chunk_lengths.pdf: chunk length in tokens against the embedding models' maximum input (section 4.2).
- rank_complementarity.pdf: rank of the gold chunk under bge-m3 and under BM25 with stems (section 4.3).

Needs only the archived results and the tokenizers in the Hugging Face cache; no model is loaded.
"""
import math
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluation.common import CHUNKS, EVAL_DIR, GOLD, load
from evaluation.input_coverage import MAX_INPUT, load_tokenizer
from medguiderag.config import RERANK_MIN_SCORE

FIGURES_DIR = EVAL_DIR.parent / "docs" / "figures"

# Validated categorical slots 1-2 (blue, orange) and text inks of the reference data-viz palette.
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
TEXT_WIDTH = 6.3  # inches, \linewidth of the thesis; figures are drawn at print size

plt.rcParams.update({
    "font.family": "Liberation Sans",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.edgecolor": INK_MUTED,
    "axes.labelcolor": INK,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "xtick.labelcolor": INK,
    "ytick.labelcolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,  # embed TrueType, so the text stays selectable and searchable in the thesis PDF
})


def swarm(xs: list[float], width: float, step: float) -> list[float]:
    """Vertical offsets that keep markers closer than `width` on x from overlapping (a simple beeswarm)."""
    placed, offsets = [], []
    for x in xs:
        k = 0
        while True:
            candidates = [0.0] if k == 0 else [k * step, -k * step]
            free = [o for o in candidates if all(abs(x - px) >= width or abs(o - po) >= step for px, po in placed)]
            if free:
                placed.append((x, free[0]))
                offsets.append(free[0])
                break
            k += 1
    return offsets


def logit(p: float) -> float:
    return math.log(p / (1 - p))


def threshold_calibration():
    data = load("thesis/threshold_calibration.json")
    rows = [
        ("Въпроси извън\nобхвата (10)", [o["top5"][0] for o in data["out_of_scope"]], ORANGE),
        ("Въпроси от набора\nза разработка (24)", [d["top5"][0][1] for d in data["development"]], BLUE),
    ]
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 2.1))
    for y, (_, scores, color) in enumerate(rows):
        scores = sorted(scores)
        offsets = swarm([logit(s) for s in scores], width=0.24, step=0.11)
        ax.scatter(scores, [y + o for o in offsets], s=26, color=color, edgecolors="white", linewidths=0.8, zorder=3)
    ax.set_xscale("logit")
    ticks = [0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlim(0.0006, 0.9994)
    ax.set_ylim(-0.6, 1.6)
    ax.set_yticks([0, 1], [label for label, _, _ in rows])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=GRID, linewidth=0.6, zorder=0)
    ax.set_xlabel("Най-висока оценка от модела за повторно класиране")

    ax.axvline(RERANK_MIN_SCORE, color=INK, linewidth=1, linestyle=(0, (4, 3)), zorder=2)
    ax.text(RERANK_MIN_SCORE, 1.6, f" праг {RERANK_MIN_SCORE}", va="top", ha="left", color=INK)
    ooc_max = max(rows[0][1])
    dev_min = min(rows[1][1])
    ax.annotate(f"{ooc_max:.3f}", (ooc_max, 0), xytext=(0, 9), textcoords="offset points", ha="center", color=INK)
    ax.annotate(f"{dev_min:.3f}", (dev_min, 1), xytext=(0, 9), textcoords="offset points", ha="center", color=INK)
    return fig


def chunk_lengths():
    tokenizer = load_tokenizer("BAAI/bge-m3")
    lengths = [len(tokenizer.encode(c["text"]).ids) for c in CHUNKS]
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 2.6))
    counts, _, _ = ax.hist(lengths, bins=range(0, 1500 + 1, 50), color=BLUE, edgecolor="white", linewidth=0.8,
                           zorder=3)
    top = max(counts) * 1.35  # headroom above the bars for the line labels
    ax.set_xlim(0, 1500)
    ax.set_ylim(0, top)
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax.set_xlabel("Дължина на откъса (токени)")
    ax.set_ylabel("Брой откъси")

    # The short-input labels are broken so that each fits between its line and the next one.
    labels = {128: "paraphrase-\nmultilingual-mpnet-\nbase-v2: 128", 512: "multilingual-e5-base: 512"}
    for name, limit in MAX_INPUT.items():
        if limit > 1500:
            continue
        ax.axvline(limit, color=INK, linewidth=1, linestyle=(0, (4, 3)), zorder=4)
        ax.text(limit + 12, top * 0.97, labels[limit], va="top", ha="left", color=INK)
    bge = MAX_INPUT["BAAI/bge-m3"]
    ax.text(1490, top * 0.97, f"bge-m3: {bge}\n(извън скалата)", va="top", ha="right", color=INK)
    return fig


def rank_complementarity():
    runs = load("thesis/retrieval_rankings.json")
    depth = len(runs["BAAI/bge-m3"][0])  # each method returns this many candidates
    beyond = 90  # where "not among the first `depth`" is drawn on the log axis

    def rank(ranking: list[int], gold: int) -> int:
        return ranking.index(gold) + 1 if gold in ranking else beyond

    pairs = Counter((rank(a, g), rank(b, g)) for a, b, g in zip(runs["BAAI/bge-m3"], runs["bm25-stem6"], GOLD))
    fig, ax = plt.subplots(figsize=(4.2, 3.9))
    for (x, y), n in pairs.items():
        ax.scatter(x, y, s=26 + 14 * (n - 1), color=BLUE, edgecolors="white", linewidths=0.8, zorder=3)
        if n > 1:
            ax.annotate(f"×{n}", (x, y), xytext=(6, 4), textcoords="offset points", color=INK, fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ticks = [1, 2, 5, 10, 20, 50, beyond]
    labels = [str(t) for t in ticks[:-1]] + [f">{depth}"]
    ax.set_xticks(ticks, labels)
    ax.set_yticks(ticks, labels)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlim(0.8, 130)
    ax.set_ylim(0.8, 130)
    ax.grid(color=GRID, linewidth=0.6, zorder=0)
    ax.plot([0.8, 130], [0.8, 130], color=GRID, linewidth=1, zorder=1)

    cut = 20
    for line in (ax.axvline, ax.axhline):
        line(cut, color=INK, linewidth=1, linestyle=(0, (4, 3)), zorder=2)
    def found(in_a: bool, in_b: bool) -> int:
        return sum(n for (x, y), n in pairs.items() if (x <= cut) == in_a and (y <= cut) == in_b)

    ax.text(1.0, 115, f"само семантичното: {found(True, False)}", va="top", color=INK, fontsize=8)
    ax.text(23, 0.9, f"само лексикалното: {found(False, True)}", va="bottom", color=INK, fontsize=8)
    ax.text(1.0, 15, f"и двата метода: {found(True, True)}", va="center", color=INK, fontsize=8)
    ax.set_xlabel("Позиция на верния откъс: семантично търсене (bge-m3)")
    ax.set_ylabel("Позиция на верния откъс: BM25 със съкращаване")
    return fig


def main():
    FIGURES_DIR.mkdir(exist_ok=True)
    for name, draw in (("threshold_calibration", threshold_calibration), ("chunk_lengths", chunk_lengths),
                       ("rank_complementarity", rank_complementarity)):
        fig = draw()
        path = FIGURES_DIR / f"{name}.pdf"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        print("saved", path)


if __name__ == "__main__":
    main()
