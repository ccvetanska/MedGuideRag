"""Find details in a generated answer that the retrieved chunks do not support: drug names, ATC codes and numbers."""
import re
from collections import Counter
from dataclasses import dataclass

# Cyrillic capitals that look like Latin ones; OCR mixes them inside codes ("АТС: А02ВА02").
_HOMOGLYPHS = str.maketrans("АВСЕНКМОРТХУ", "ABCEHKMOPTXY")
_LOWER_HOMOGLYPHS = str.maketrans("аеорсух", "aeopcyx")
_CYRILLIC = dict(zip("абвгдежзийклмнопрстуфхцчшщъьюя",
                     "a b v g d e zh z i i k l m n o p r s t u f h ts ch sh sht a _ yu ya".split()))
# Bulgarian definite articles and plural endings, longest first: "метоклопрамидът" -> "метоклопрамид".
_BG_ENDINGS = ("овете", "ите", "ове", "ът", "ят", "та", "то", "те", "а", "я", "и")
# WHO INN stems: a Cyrillic word with one of these endings is taken as a drug name even if the corpus lacks it.
_INN_STEMS = ("олол", "прил", "сартан", "дипин", "статин", "празол", "маб", "тиниб", "вир", "цилин", "циклин",
              "флоксацин", "мицин", "платин", "таксел", "парин", "семид", "тиазид", "тидин", "сетрон", "триптан",
              "азепам", "оксетин", "глиптин", "глитазон", "глифлозин", "профен", "низолон")
_WORD = re.compile(r"[A-Za-zА-Яа-я][A-Za-zА-Яа-я\-]*")
_CODE = re.compile(r"(?<![A-Za-zА-Яа-я0-9])[A-ZА-Я][0-9ОOЗ]{2}[A-ZА-Я]{2}[0-9ОOЗ]{2}(?![A-Za-zА-Яа-я0-9])")
# A number, preferring digit groups ("90 000", "25,000") so they are read whole; digits glued to letters
# ("HER2", "5-HT3", "B12") are part of a name, not a quantity.
_NUMBER = re.compile(r"(?<![\w.,])(?:\d{1,3}(?:[  .,]\d{3})+|\d+(?:[.,]\d+)?)(?![\d])")
_PLAIN_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
_ENUMERATOR = re.compile(r"(?m)^[ \t]*(?:[-*][ \t]*)?\d+[.)][ \t]")
_UNIT = re.compile(r" ?(?:%|(?:mg|g|µg|μg|mcg|kg|ml|l|IU|U|mmol|мг|мл|ME)(?:/[A-Za-zа-я0-9²³]+)*|часа|дни|седмици|месеца)(?![A-Za-zА-Яа-я])")
_NON_DRUG_LATIN = {"rec", "inn", "atc", "mg", "kg"}
# "стр. 17" or "стр. 17-19" right before a number: a page reference to the shown source, not a quantity.
_PAGE_REF = re.compile(r"стр\.?\s*(?:\d+\s*[-–]\s*)?$")


@dataclass(frozen=True)
class Issue:
    kind: str    # "наименование", "ATC код" or "число"
    text: str    # as written in the answer, a number with its unit
    reason: str
    start: int
    end: int


def _unmix(word: str) -> str:
    """OCR writes Latin names with stray Cyrillic lookalikes ("Chlоrpromazine"); make such words fully Latin."""
    if re.search("[A-Za-z]", word) and re.search("[а-яА-Я]", word):
        word = word.translate(_LOWER_HOMOGLYPHS).translate(_HOMOGLYPHS)
    return word


def _cyrillic_forms(word: str) -> set[str]:
    w = word.lower()
    return {w} | {w[: -len(e)] for e in _BG_ENDINGS if w.endswith(e) and len(w) - len(e) >= 4}


def _skeletons(word: str) -> set[str]:
    """Spelling-independent keys, so "Metoclopramide" and "метоклопрамида" share one."""
    w = _unmix(word).lower().strip("-")
    if re.search("[а-я]", w):
        forms = {"".join(_CYRILLIC.get(ch, ch) for ch in f).replace("_", "") for f in _cyrillic_forms(w)}
    else:
        w = w.replace("ph", "f").replace("th", "t").replace("ch", "h").replace("qu", "kv").replace("x", "ks")
        w = re.sub(r"c(?=[eiy])", "ts", w).replace("c", "k").replace("y", "i").replace("w", "v")
        forms = {re.sub(r"e$", "", w)}
    # Bulgarian writes an intervocalic s as з (esomeprazole -> езомепразол); doubled letters are dropped.
    return {re.sub(r"(.)\1", r"\1", f.replace("s", "z")) for f in forms}


def _edit_distance(a: str, b: str) -> int:
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, row[0] = row[0], i
        for j, cb in enumerate(b, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (ca != cb))
    return row[-1]


def _matches(keys: set[str], reference: set[str]) -> bool:
    """Exact key match, or a near one: Bulgarian spellings and slips differ from the INN by a letter or two
    ("хидрохлортиазид" / hydrochlorothiazide, "Омалзумаб" / omalizumab), while prednisone and prednisolone stay apart."""
    if keys & reference:
        return True
    for k in keys:
        for r in reference:
            shorter = min(len(k), len(r))
            allowed = 2 if shorter >= 11 else 1 if shorter >= 7 else 0
            if allowed and k[0] == r[0] and abs(len(k) - len(r)) <= allowed and _edit_distance(k, r) <= allowed:
                return True
    return False


def _number_values(text: str) -> set[str]:
    """Every reading of a number: "90,000" may be ninety thousand or ninety with a decimal comma."""
    compact = re.sub(r"[  ]", "", text)
    readings = {compact.replace(",", ".")}
    if re.fullmatch(r"\d{1,3}(?:[.,]\d{3})+", compact) or compact != text:
        readings.add(re.sub(r"[.,]", "", compact))
    values = set()
    for r in readings:
        try:
            values.add(format(float(r), "g"))
        except ValueError:
            pass
    return values


def _normalize_code(code: str) -> str:
    chars = list(code.translate(_HOMOGLYPHS))
    for i in (1, 2, 5, 6):
        chars[i] = {"O": "0", "З": "3"}.get(chars[i], chars[i])
    return "".join(chars)


class GroundingChecker:
    def __init__(self, corpus_texts: list[str]):
        # The guidelines write drug names in Latin (INN) and general terms in Bulgarian. A key seen at least as
        # often in Latin as in Cyrillic is a drug name ("метопролол"); "терапия" is not, despite "therapy" in Latin.
        self.latin_count, self.cyrillic_count = Counter(), Counter()
        for text in corpus_texts:
            for w in _WORD.findall(text):
                if len(w) >= 5:
                    counts = self.cyrillic_count if re.search("[а-яА-Я]", _unmix(w)) else self.latin_count
                    counts.update(_skeletons(w))

    def _is_name(self, word: str) -> bool:
        """Latin words (in these answers: drug names, pathogens, Latin diagnoses) and Cyrillic drug names."""
        word = _unmix(word)
        if len(word) < 5 or word.lower() in _NON_DRUG_LATIN:
            return False
        if not re.search("[а-яА-Я]", word):
            return True
        if any(f.endswith(_INN_STEMS) for f in _cyrillic_forms(word)):
            return True
        return any(self.latin_count[k] and self.latin_count[k] >= self.cyrillic_count[k] for k in _skeletons(word))

    def check(self, answer: str, chunks: list[dict], question: str = "") -> list[Issue]:
        context = " ".join(" ".join(c["text"].split()) for c in chunks)
        reference = context + " " + question
        # The prompt also shows each chunk's file name, which a model may quote; its page numbers are
        # left out here so they cannot vouch for a made-up number (page references are skipped below).
        sources = " ".join(c["document"] for c in chunks)
        # Hyphenated words also count by their parts: the file "PharmTher-onkologia_..." may be quoted as "PharmTher – onkologia".
        ref_words = _WORD.findall(reference + " " + sources)
        ref_keys = {k for w in ref_words + [p for w in ref_words for p in w.split("-")] for k in _skeletons(w)}
        ref_numbers = set()
        for m in list(_NUMBER.finditer(reference)) + list(_PLAIN_NUMBER.finditer(reference)):
            ref_numbers |= _number_values(m.group(0))
        # Which drugs stand right before each code in the chunks ("Famotidine (Rec. INN) АТС: А02ВАОЗ").
        ref_codes = {}
        for m in _CODE.finditer(context):
            # 45 characters cover "• Famotidine (Rec. INN) АТС: " but not the previous list item.
            near = {k for w in _WORD.findall(context[max(0, m.start() - 45): m.start()]) for k in _skeletons(w)}
            ref_codes.setdefault(_normalize_code(m.group(0)), set()).update(near)

        issues, taken = [], []
        for m in _CODE.finditer(answer):
            code = _normalize_code(m.group(0))
            # The drug a code belongs to stands before it, on the same line and after the previous code
            # ("Omalizumab (R03DX05), mepolizumab (R03DX09)").
            start = max(answer.rfind("\n", 0, m.start()) + 1, taken[-1][1] if taken else 0, m.start() - 80)
            taken.append((m.start(), m.end()))
            drugs = [w for w in _WORD.findall(answer[start: m.start()]) if self._is_name(w)]
            if code not in ref_codes:
                issues.append(Issue("ATC код", m.group(0), "липсва в откъсите", m.start(), m.end()))
            elif drugs and not _matches(_skeletons(drugs[-1]), ref_codes[code]):
                issues.append(Issue("ATC код", m.group(0), f"в откъсите не е свързан с {drugs[-1]}", m.start(), m.end()))

        for m in _WORD.finditer(answer):
            if self._is_name(m.group(0)) and not _matches(_skeletons(m.group(0)), ref_keys):
                issues.append(Issue("наименование", m.group(0), "не се споменава в откъсите", m.start(), m.end()))

        taken += [(m.start(), m.end()) for m in _ENUMERATOR.finditer(answer)]
        for m in _NUMBER.finditer(answer):
            if any(s <= m.start() < e for s, e in taken) or _PAGE_REF.search(answer, 0, m.start()):
                continue
            if not (_number_values(m.group(0)) & ref_numbers):
                # A bare "16" means little to the reader; show it with its unit ("16 mg/kg") when one follows.
                unit = _UNIT.match(answer, m.end())
                label = m.group(0) + (unit.group(0) if unit else "")
                issues.append(Issue("число", label, "липсва в откъсите", m.start(), m.end()))
        return sorted(issues, key=lambda i: i.start)


def highlight(answer: str, issues: list[Issue]) -> str:
    """Answer as Streamlit markdown with the unsupported details on a red background."""
    out, pos = [], 0
    for i in sorted(issues, key=lambda i: i.start):
        if i.start < pos:
            continue
        out += [answer[pos:i.start], f":red-background[{answer[i.start:i.end]}]"]
        pos = i.end
    return "".join(out) + answer[pos:]
