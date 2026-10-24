"""Find details in a generated answer that the retrieved chunks do not support: drug names, ATC codes and numbers.

The spelling helpers (WORD, NUMBER, skeletons, ...) are shared with coverage.py and attribution.py.
"""
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
INN_STEMS = ("олол", "прил", "сартан", "дипин", "статин", "празол", "маб", "тиниб", "вир", "цилин", "циклин",
              "флоксацин", "мицин", "платин", "таксел", "парин", "семид", "тиазид", "тидин", "сетрон", "триптан",
              "азепам", "оксетин", "глиптин", "глитазон", "глифлозин", "профен", "низолон")
WORD = re.compile(r"[A-Za-zА-Яа-я][A-Za-zА-Яа-я\-]*")
_CODE = re.compile(r"(?<![A-Za-zА-Яа-я0-9])[A-ZА-Я][0-9ОOЗ]{2}[A-ZА-Я]{2}[0-9ОOЗ]{2}(?![A-Za-zА-Яа-я0-9])")
# A number, preferring digit groups ("90 000", "25,000") so they are read whole; digits glued to letters
# ("HER2", "5-HT3", "B12") are part of a name, not a quantity.
NUMBER = re.compile(r"(?<![\w.,])(?:\d{1,3}(?:[  .,]\d{3})+|\d+(?:[.,]\d+)?)(?![\d])")
_PLAIN_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
_ENUMERATOR = re.compile(r"(?m)^[ \t]*(?:[-*][ \t]*)?\d+[.)][ \t]")
_UNIT = re.compile(r" ?(?:%|(?:mg|g|µg|μg|mcg|kg|ml|l|IU|U|mmol|мг|мл|ME)(?:/[A-Za-zа-я0-9²³]+)*"
                   r"|часа|дни|седмици|месеца)(?![A-Za-zА-Яа-я])")
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


def unmix(word: str) -> str:
    """OCR writes Latin names with stray Cyrillic lookalikes ("Chlоrpromazine"); make such words fully Latin."""
    if re.search("[A-Za-z]", word) and re.search("[а-яА-Я]", word):
        word = word.translate(_LOWER_HOMOGLYPHS).translate(_HOMOGLYPHS)
    return word


def cyrillic_forms(word: str) -> set[str]:
    w = word.lower()
    return {w} | {w[: -len(e)] for e in _BG_ENDINGS if w.endswith(e) and len(w) - len(e) >= 4}


def skeletons(word: str) -> set[str]:
    """Spelling-independent keys, so "Metoclopramide" and "метоклопрамида" share one."""
    w = unmix(word).lower().strip("-")
    if re.search("[а-я]", w):
        forms = {"".join(_CYRILLIC.get(ch, ch) for ch in f).replace("_", "") for f in cyrillic_forms(w)}
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


def number_values(text: str) -> set[str]:
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


def _reference_keys(reference: str, chunks: list[dict]) -> set[str]:
    """Keys of the words in the chunks and the question, and of the chunks' file names, which the prompt shows
    and a model may quote; hyphenated words also count by their parts ("PharmTher-onkologia_...")."""
    words = WORD.findall(reference + " " + " ".join(c["document"] for c in chunks))
    words += [part for w in words for part in w.split("-")]
    return {k for w in words for k in skeletons(w)}


def _reference_numbers(reference: str) -> set[str]:
    """Numbers in the chunks and the question; page numbers are left out, so they cannot vouch for a made-up one."""
    numbers = set()
    for m in list(NUMBER.finditer(reference)) + list(_PLAIN_NUMBER.finditer(reference)):
        numbers |= number_values(m.group(0))
    return numbers


def _codes_with_drugs(context: str) -> dict[str, set[str]]:
    """Each ATC code in the chunks with the keys of the words before it ("Famotidine (Rec. INN) АТС: А02ВАОЗ")."""
    codes = {}
    for m in _CODE.finditer(context):
        # 45 characters cover "• Famotidine (Rec. INN) АТС: " but not the previous list item.
        before = context[max(0, m.start() - 45): m.start()]
        keys = {k for w in WORD.findall(before) for k in skeletons(w)}
        codes.setdefault(_normalize_code(m.group(0)), set()).update(keys)
    return codes


def _number_issues(answer: str, ref_numbers: set[str], skip: list[tuple[int, int]]) -> list[Issue]:
    """Numbers missing from the chunks, except inside the skipped spans, list numbering and page references."""
    skip = skip + [(m.start(), m.end()) for m in _ENUMERATOR.finditer(answer)]
    issues = []
    for m in NUMBER.finditer(answer):
        if any(s <= m.start() < e for s, e in skip) or _PAGE_REF.search(answer, 0, m.start()):
            continue
        if not number_values(m.group(0)) & ref_numbers:
            # A bare "16" means little to the reader; show it with its unit ("16 mg/kg") when one follows.
            unit = _UNIT.match(answer, m.end())
            label = m.group(0) + (unit.group(0) if unit else "")
            issues.append(Issue("число", label, "липсва в откъсите", m.start(), m.end()))
    return issues


class GroundingChecker:
    def __init__(self, corpus_texts: list[str]):
        # The guidelines write drug names in Latin (INN) and general terms in Bulgarian. A key seen at least as
        # often in Latin as in Cyrillic is a drug name ("метопролол"); "терапия" is not, despite "therapy" in Latin.
        self.latin_count, self.cyrillic_count = Counter(), Counter()
        for text in corpus_texts:
            for w in WORD.findall(text):
                if len(w) >= 5:
                    counts = self.cyrillic_count if re.search("[а-яА-Я]", unmix(w)) else self.latin_count
                    counts.update(skeletons(w))

    def _is_name(self, word: str) -> bool:
        """Latin words (in these answers: drug names, pathogens, Latin diagnoses) and Cyrillic drug names."""
        word = unmix(word)
        if len(word) < 5 or word.lower() in _NON_DRUG_LATIN:
            return False
        if not re.search("[а-яА-Я]", word):
            return True
        if any(f.endswith(INN_STEMS) for f in cyrillic_forms(word)):
            return True
        return any(self.latin_count[k] and self.latin_count[k] >= self.cyrillic_count[k] for k in skeletons(word))

    def _code_issues(self, answer: str, ref_codes: dict[str, set[str]]) -> tuple[list[Issue], list[tuple[int, int]]]:
        """Codes missing from the chunks or tied there to another drug, and the spans of all codes in the answer."""
        issues, spans = [], []
        for m in _CODE.finditer(answer):
            code = _normalize_code(m.group(0))
            # The code's drug stands before it, on the same line and after the previous code
            # ("Omalizumab (R03DX05), mepolizumab (R03DX09)").
            start = max(answer.rfind("\n", 0, m.start()) + 1, spans[-1][1] if spans else 0, m.start() - 80)
            spans.append((m.start(), m.end()))
            drugs = [w for w in WORD.findall(answer[start: m.start()]) if self._is_name(w)]
            if code not in ref_codes:
                issues.append(Issue("ATC код", m.group(0), "липсва в откъсите", m.start(), m.end()))
            elif drugs and not _matches(skeletons(drugs[-1]), ref_codes[code]):
                reason = f"в откъсите не е свързан с {drugs[-1]}"
                issues.append(Issue("ATC код", m.group(0), reason, m.start(), m.end()))
        return issues, spans

    def _name_issues(self, answer: str, ref_keys: set[str]) -> list[Issue]:
        return [Issue("наименование", m.group(0), "не се споменава в откъсите", m.start(), m.end())
                for m in WORD.finditer(answer)
                if self._is_name(m.group(0)) and not _matches(skeletons(m.group(0)), ref_keys)]

    def check(self, answer: str, chunks: list[dict], question: str = "") -> list[Issue]:
        context = " ".join(" ".join(c["text"].split()) for c in chunks)
        reference = context + " " + question
        code_issues, code_spans = self._code_issues(answer, _codes_with_drugs(context))
        name_issues = self._name_issues(answer, _reference_keys(reference, chunks))
        number_issues = _number_issues(answer, _reference_numbers(reference), skip=code_spans)
        return sorted(code_issues + name_issues + number_issues, key=lambda i: i.start)


def highlight(answer: str, issues: list[Issue]) -> str:
    """Answer as Streamlit markdown with the unsupported details on a red background."""
    out, pos = [], 0
    for i in sorted(issues, key=lambda i: i.start):
        if i.start < pos:
            continue
        out += [answer[pos:i.start], f":red-background[{answer[i.start:i.end]}]"]
        pos = i.end
    return "".join(out) + answer[pos:]
