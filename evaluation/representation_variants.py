"""Variants of the chunk representation for bge-m3: normalized text, guideline/section titles prepended,
overlapping sub-chunks of 120 or 200 words ranked by their best window (thesis paragraph "Варианти на представянето").

    python -m evaluation.representation_variants

Embeds with bge-m3 served by Ollama on the GPU (`ollama pull bge-m3` once); it gives the same rankings as the
sentence-transformers model in medguiderag but builds the extra indexes in about two minutes each.
"""
import faiss
import numpy as np
import ollama

from evaluation.common import CHUNKS, QUESTIONS, metrics, save

MODEL = "bge-m3"
DEPTH = 50
T = "Фармакотерапевтично ръководство "
TITLES = {
    "Farmako_Derma1.pdf": T + "по кожни и венерически болести",
    "farmakoterap_ruk-neurology__2018-1.pdf": T + "за лечение на неврологичните заболявания",
    "ft-ginekologiq.pdf": T + "по акушерство и гинекология",
    "FTR-urologichni_zaboliavania.pdf": T + "за лечение на урологични заболявания",
    "FTR_Hematology_2018_250418__JT_29.05.2018-J51.pdf": T + "по хематология",
    "FTR_himeoterapia_i_onkohematologia_pri_deca_BZ_final.pdf": T + "по детска клинична хематология и онкология",
    "FTR_Nefrologia.pdf": T + "по нефрология и диализа",
    "FTR_psihichni_zaboliavania.pdf": T + "по психични заболявания",
    "Hepatiti_B_C_-_нов_13.12.2018-1.pdf": T + "за лечение на гастроентерологичните заболявания: хепатит B и C",
    "IBD_new_13.12.2018.pdf": T + "за лечение на гастроентерологичните заболявания: възпалителни болести на червата",
    "neonatologia_13.12.2018.pdf": T + "по неонатология",
    "PharmTher-Nuclear_medicine-25.01.2018.pdf": T + "по нуклеарна медицина",
    "PharmTher-onkologia_2018_dopaln._293_pr._23.08.2018__version-J21.pdf": T + "по медицинска онкология",
    "PharmTher_spravochnik_allergies.pdf": T + "за лечение на алергичните болести",
    "ФТР-Гастроентерологични заболявания.pdf": T + "за лечение на гастроентерологичните заболявания",
    "ФТР-Ендокринология.pdf": T + "по ендокринология и болести на обмяната",
    "ФТР-Очни_Болести.pdf": T + "по очни болести",
}


def vectors(texts: list[str]) -> np.ndarray:
    out = []
    for i in range(0, len(texts), 16):
        out.extend(ollama.embed(model=MODEL, input=texts[i:i + 16])["embeddings"])
    v = np.array(out, dtype="float32")
    faiss.normalize_L2(v)
    return v


def clean(chunk: dict) -> str:
    return " ".join(chunk["text"].split())


def windows(words: list[str], size: int) -> list[str]:
    stride = int(size * 0.75)  # 25% overlap
    if len(words) <= size:
        return [" ".join(words)]
    out = [" ".join(words[s:s + size]) for s in range(0, len(words) - size + 1, stride)]
    if (len(words) - size) % stride:
        out.append(" ".join(words[-size:]))
    return out


def rank(texts: list[str], owner: list[int]) -> list[list[int]]:
    """Rank chunks by their best text (a chunk may own several windows)."""
    index = faiss.IndexFlatIP(1024)
    index.add(vectors(texts))
    _, idx = index.search(vectors(QUESTIONS), DEPTH * 8)
    rankings = []
    for row in idx:
        seen = []
        for j in row:
            if owner[j] not in seen:
                seen.append(owner[j])
        rankings.append(seen[:DEPTH])
    return rankings


def main():
    variants = {
        "raw text": lambda c: c["text"],
        "normalized": clean,
        "guideline title + normalized": lambda c: TITLES[c["document"]] + "\n" + clean(c),
        "guideline + section title + normalized": lambda c: TITLES[c["document"]] + "\nРаздел: " + c["section"] + "\n" + clean(c),
    }
    runs = {name: rank([fn(c) for c in CHUNKS], list(range(len(CHUNKS)))) for name, fn in variants.items()}
    for size in (120, 200):
        for titled in (False, True):
            texts, owner = [], []
            for i, c in enumerate(CHUNKS):
                for w in windows(clean(c).split(), size):
                    texts.append(TITLES[c["document"]] + "\n" + w if titled else w)
                    owner.append(i)
            runs[f"sub-chunks of {size} words{' + title' if titled else ''}"] = rank(texts, owner)

    print(f"{'variant':42} | strict R@1  R@5 R@10  MRR")
    for name, rankings in runs.items():
        s = metrics(rankings)["strict"]
        print(f"{name:42} |       {s['R@1']:.2f} {s['R@5']:.2f} {s['R@10']:.2f} {s['MRR']:.2f}")
    print("saved", save("representation_variants.json", runs))


if __name__ == "__main__":
    main()
