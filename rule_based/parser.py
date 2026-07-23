"""
Radiology impression parser — overall normal/abnormal classifier.

Rule-based, finding-cue + negation driven. Designed for free-text Bahasa
Indonesia + Latin/English radiology impressions (USG abdomen, Rontgen dada).

Public API:
    normalize(text)                  -> cleaned lowercase string
    classify_overall(text)           -> "normal" | "abnormal"
    evaluate(y_true, y_pred, pos="abnormal") -> metrics dict (and prints)

Design notes
------------
The decision rule is: an impression is ABNORMAL if any clause contains an
abnormal finding cue that is NOT negated; otherwise NORMAL. "Abnormal wins"
in mixed sentences such as
    "Cor membesar, ginjal kanan dan kiri normal, gallbladder unremarkable."
which is abnormal even though it contains the word "normal".

Negation flips abnormal cues to normal, e.g.
    "cor tidak membesar", "no hepatomegaly", "no pleural effusion",
    "tidak tampak kelainan", "no active lung lesion".
"""

import re

# --- normalization ---------------------------------------------------------

# Leading report-section prefixes that carry no finding information.
_PREFIX_RE = re.compile(r"^\s*(foto\s+thorax|hasil|kesan|impression|usg)\s*:\s*", re.I)

# Trailing recommendation clauses (no finding info) — strip from the end.
_RECO_RE = re.compile(
    r"\b("
    r"saran[^.;]*|disarankan[^.;]*|dianjurkan[^.;]*|"
    r"korelasi\s+klinis[^.;]*|suggest\s+clinical\s+correlation[^.;]*|"
    r"clinical\s+correlation[^.;]*|mohon[^.;]*"
    r")\.?\s*$",
    re.I,
)


def normalize(text):
    """Lowercase, strip section prefixes / trailing recommendations, squash ws."""
    if text is None:
        return ""
    t = str(text).strip().lower()
    t = _PREFIX_RE.sub("", t)
    # A recommendation can be appended after the findings with no separator
    # ("... hepar tidak membesar Saran USG ulang."). Strip it repeatedly.
    prev = None
    while prev != t:
        prev = t
        t = _RECO_RE.sub("", t).strip()
    t = re.sub(r"\s+", " ", t).strip()
    return t


# --- lexicons --------------------------------------------------------------

# Abnormal finding cues. Order/grouping is only for readability; all are ORed.
ABNORMAL_CUES = [
    # enlargement
    r"membesar", r"megali", r"megaly", r"pembesaran", r"enlarged", r"enlargement",
    # stones
    r"lithiasis", r"litiasis", r"kolelitiasis", r"nephrolithiasis",
    r"batu empedu", r"batu ginjal", r"gallstones?", r"kidney stones?",
    # inflammation
    r"cholecystitis", r"kolesistitis", r"cholangitis", r"pyelonephritis",
    # effusion / fluid
    r"efusi", r"effusion", r"asites", r"ascites",
    # mass / lesion / nodule / cyst / tumor
    r"massa", r"\bmass\b", r"lesi", r"lesion", r"nodul", r"nodule",
    r"kista", r"\bcyst\b", r"tumor",
    # liver fat
    r"fatty liver", r"steatosis", r"perlemakan hati",
    # lung
    r"fibrosis", r"infiltrat", r"infiltrate", r"bronchopneumonia",
    r"pneumonia", r"konsolidasi", r"consolidation",
    # renal / biliary dilatation
    r"hidronefrosis", r"hydronephrosis", r"dilatasi",
    # thickening / calcification / stenosis
    r"penebalan", r"thickening", r"kalsifikasi", r"stenosis", r"hipertrofi",
    # generic
    r"\babnormal\b", r"kelainan",
]
_ABNORMAL_RE = re.compile("|".join(ABNORMAL_CUES))

# Negation tokens. If one of these appears just before an abnormal cue (within
# a few tokens, no clause break in between) the cue is treated as negated.
NEGATION = [
    "tidak tampak", "tidak ada", "tidak", "tanpa", "without",
    "no active", "no", "negatif", "bebas", "free of", "tak tampak",
]
_NEG_RE = re.compile(r"\b(" + "|".join(re.escape(n) for n in NEGATION) + r")\b")

# Phrases that look like an abnormal cue but are explicitly normal. These are
# matched and removed before abnormal detection so they cannot trigger it.
NORMAL_OVERRIDES = [
    r"ctr normal",
    r"sinus costophrenicus tajam",  # normal chest sign
    r"lungs clear",
]
_NORMAL_OVR_RE = re.compile("|".join(NORMAL_OVERRIDES))

# Window (in words) a negation may reach forward to suppress an abnormal cue.
_NEG_WINDOW = 4

# Clause separators.
_CLAUSE_SPLIT_RE = re.compile(r"[;,.]| - | dan | tetapi | namun | dengan ")


def _ctr_abnormal(text):
    """Return True if a CTR value > 0.50 is stated (cardiomegaly proxy)."""
    abn = False
    for m in re.finditer(r"ctr\s*:?\s*(\d+(?:\.\d+)?)", text):
        try:
            if float(m.group(1)) > 0.50:
                abn = True
        except ValueError:
            pass
    return abn


def _clause_is_abnormal(clause):
    """True if the clause contains an un-negated abnormal cue."""
    clause = _NORMAL_OVR_RE.sub(" ", clause)
    for m in _ABNORMAL_RE.finditer(clause):
        before = clause[: m.start()]
        words_before = before.split()
        window = " ".join(words_before[-_NEG_WINDOW:])
        if _NEG_RE.search(window):
            continue  # negated -> not an abnormal finding
        return True
    return False


def classify_overall(text):
    """Classify an impression as 'normal' or 'abnormal'."""
    t = normalize(text)
    if not t:
        return "normal"
    if _ctr_abnormal(t):
        return "abnormal"
    for clause in _CLAUSE_SPLIT_RE.split(t):
        clause = clause.strip()
        if clause and _clause_is_abnormal(clause):
            return "abnormal"
    return "normal"


# --- organ-level parsing ---------------------------------------------------

ORGANS = ["cor", "pulmo", "hepar", "ginjal", "vesica", "pleura"]

# Synonyms that explicitly name each organ (Latin / Bahasa / English).
ORGAN_SYNONYMS = {
    "cor":    [r"\bcor\b", r"\bheart\b", r"cardiac", r"\bctr\b", r"jantung",
               r"cardiomeg", r"kardiomeg"],
    "pulmo":  [r"\bpulmo\b", r"\blungs?\b", r"\bparu\b", r"pulmonary"],
    "hepar":  [r"\bhepar\b", r"\bliver\b", r"hepat", r"\bhati\b"],
    "ginjal": [r"\bginjal\b", r"\bkidneys?\b", r"\brenal\b", r"nephro"],
    "vesica": [r"vesica", r"gallbladder", r"empedu", r"cholecyst", r"kolesist",
               r"biliary", r"\bgallstones?\b"],
    "pleura": [r"pleura", r"pleural"],
}

# Findings that imply an organ even when the organ noun is absent.
FINDING_ORGAN = [
    (r"hepatomeg|fatty liver|steatosis|perlemakan hati", "hepar"),
    (r"cardiomeg|kardiomeg|enlarged cardiac|\bctr\b", "cor"),
    (r"nephrolith|hidronefrosis|hydronephrosis|renal cyst|batu ginjal|"
     r"pyelonephritis|kidney stones?", "ginjal"),
    (r"cholelith|kolelitiasis|gallstones?|batu empedu", "vesica"),
    (r"efusi pleura|pleural effusion|penebalan pleura", "pleura"),
    (r"fibrosis|infiltrat|bronchopneumonia|pneumonia|konsolidasi|"
     r"consolidation|lung lesion|lungs clear|\btb\b|tuberculosis", "pulmo"),
]
_ORGAN_SYN_RE = {o: re.compile("|".join(s)) for o, s in ORGAN_SYNONYMS.items()}
_FINDING_ORGAN_RE = [(re.compile(p), o) for p, o in FINDING_ORGAN]


def _organs_in_clause(clause):
    found = set()
    for o, rx in _ORGAN_SYN_RE.items():
        if rx.search(clause):
            found.add(o)
    for rx, o in _FINDING_ORGAN_RE:
        if rx.search(clause):
            found.add(o)
    return found


def parse_organs(text):
    """Return {organ: 'normal'|'abnormal'|'not_mentioned'} for the 6 organs.

    Clause-based: each clause's organs inherit that clause's status. Abnormal
    wins (an organ flagged abnormal in any clause stays abnormal).
    """
    t = normalize(text)
    status = {o: "not_mentioned" for o in ORGANS}
    if not t:
        return status
    for clause in _CLAUSE_SPLIT_RE.split(t):
        clause = clause.strip()
        if not clause:
            continue
        orgs = _organs_in_clause(clause)
        if not orgs:
            continue
        abn = _clause_is_abnormal(clause) or _ctr_abnormal(clause)
        for o in orgs:
            if abn:
                status[o] = "abnormal"
            elif status[o] == "not_mentioned":
                status[o] = "normal"
    return status


def classify_overall_from_organs(text):
    """Derive the overall label from organ-level findings (abnormal if any)."""
    st = parse_organs(text)
    return "abnormal" if "abnormal" in st.values() else "normal"


# --- vocabulary holdout (for generalization / held-out-cue experiments) ----
# Base copies captured at import so a holdout can be applied and then reverted.
# Removing cues here is how we simulate a finding whose vocabulary was NEVER
# seen: the classifier literally cannot match it. configure_vocabulary() does an
# EXACT removal from ABNORMAL_CUES (this is what drives classify_overall) plus a
# best-effort prune of the organ-implication maps so parse_organs stays honest.
_BASE_ABNORMAL_CUES = list(ABNORMAL_CUES)
_BASE_FINDING_ORGAN = list(FINDING_ORGAN)
_BASE_ORGAN_SYNONYMS = {o: list(s) for o, s in ORGAN_SYNONYMS.items()}


def _alt_excluded(alt, ex):
    """True if a regex alternative overlaps (either direction) an excluded cue."""
    return any(alt == e or alt in e or e in alt for e in ex)


def _drop_alts(pattern, ex):
    """Drop overlapping '|'-alternatives from an organ-map regex pattern."""
    return "|".join(a for a in pattern.split("|") if not _alt_excluded(a, ex))


def configure_vocabulary(exclude_cues=()):
    """Rebuild every compiled regex with ``exclude_cues`` held out.

    Pass a list of raw cue strings (entries of ABNORMAL_CUES). Call with no
    arguments, or reset_vocabulary(), to restore the full vocabulary.
    """
    global ABNORMAL_CUES, _ABNORMAL_RE
    global FINDING_ORGAN, _FINDING_ORGAN_RE, ORGAN_SYNONYMS, _ORGAN_SYN_RE
    ex = set(exclude_cues)
    ABNORMAL_CUES = [c for c in _BASE_ABNORMAL_CUES if c not in ex]
    _ABNORMAL_RE = re.compile("|".join(ABNORMAL_CUES))
    FINDING_ORGAN = [(pruned, o) for p, o in _BASE_FINDING_ORGAN
                     for pruned in [_drop_alts(p, ex)] if pruned]
    _FINDING_ORGAN_RE = [(re.compile(p), o) for p, o in FINDING_ORGAN]
    ORGAN_SYNONYMS = {o: [a for a in alts if not _alt_excluded(a, ex)]
                      for o, alts in _BASE_ORGAN_SYNONYMS.items()}
    _ORGAN_SYN_RE = {o: re.compile("|".join(s))
                     for o, s in ORGAN_SYNONYMS.items() if s}


def reset_vocabulary():
    """Restore the full vocabulary after a configure_vocabulary() holdout."""
    configure_vocabulary(())


# --- baseline (for comparison) ---------------------------------------------

_NAIVE_NORMAL_RE = re.compile(
    r"(normal|dalam batas normal|tidak tampak kelainan|no active|unremarkable)"
)


def classify_naive(text):
    """Old heuristic: 'normal' if the text mentions a normal keyword anywhere."""
    t = str(text).lower()
    return "normal" if _NAIVE_NORMAL_RE.search(t) else "abnormal"


# --- evaluation ------------------------------------------------------------

def evaluate(y_true, y_pred, pos="abnormal", title="", show_errors=False, texts=None):
    """Compute and print accuracy / precision / recall / F1 + confusion matrix.

    Returns a dict with the metrics. Positive class defaults to 'abnormal'.
    """
    y_true = list(y_true)
    y_pred = list(y_pred)
    n = len(y_true)
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == pos and p == pos)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != pos and p == pos)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == pos and p != pos)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t != pos and p != pos)
    acc = (tp + tn) / n if n else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    if title:
        print(f"=== {title} (n={n}) ===")
    print(f"accuracy={acc:.3f}  precision={prec:.3f}  recall={rec:.3f}  f1={f1:.3f}")
    print(f"confusion: TP={tp} FP={fp} FN={fn} TN={tn}  (positive='{pos}')")
    if show_errors and texts is not None:
        errs = [(tx, t, p) for tx, t, p in zip(texts, y_true, y_pred) if t != p]
        print(f"-- {len(errs)} misclassified --")
        for tx, t, p in errs:
            print(f"   gold={t:8s} pred={p:8s} | {tx}")
    return {"n": n, "accuracy": acc, "precision": prec, "recall": rec,
            "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}
