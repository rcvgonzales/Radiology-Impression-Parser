"""
Three-way comparison for overall normal/abnormal classification:

  1. Rule baseline      — classify_overall (regex + negation, hand-written)
  2. TF-IDF + LogReg    — genuinely TRAINED ML (scikit-learn .fit())
  3. Transformer        — fine-tuned IndoBERT / XLM-R (see train_transformer.py)

Evaluated on two axes:
  1. In-distribution: the leakage-free test split (distinct texts, seed=42) —
     near-saturated for all models on this closed synthetic vocabulary.
  2. Vocabulary-held-out: effusion + stones cues removed from BOTH the rule
     vocabulary and the trained models' data, then scored on impressions that
     use them (validate_generalization.heldout_split). The sole-signal subset
     is the honest generalization number — rules collapse to 0, learned models
     recover some. Writes generalization.json for the report page/PDF.

Run: python train_models.py            # rules + tf-idf (fast)
Transformer rows (in-dist + held-out) are merged in from transformer_results.json
and transformer_heldout.json if train_transformer.py has been run.
"""

import os
import sys
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path[:0] = [os.path.join(_ROOT, "shared"),       # data_utils
                os.path.join(_ROOT, "rule_based")]   # parser, validate_generalization
_OUT = os.path.join(_HERE, "outputs")
os.makedirs(_OUT, exist_ok=True)

import data_utils as D
import parser as P
from validate_generalization import OOD, HELDOUT_CUES, heldout_split  # same sets

POS = "abnormal"

gold = D.load_gold()
g = D.split_by_text(gold, seed=42)
tr, te = g[g.split == "train"], g[g.split == "test"]
ood_X = [t for t, _ in OOD]
ood_y = [y for _, y in OOD]


def score(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[POS], average="binary", pos_label=POS,
        zero_division=0)
    return {"model": name, "accuracy": round(float(acc), 3),
            "precision": round(float(p), 3), "recall": round(float(r), 3),
            "f1": round(float(f), 3)}


rows = []

# 1) RULE BASELINE ----------------------------------------------------------
rule_test = te["text"].map(P.classify_overall)
rule_ood = [P.classify_overall(t) for t in ood_X]
rows.append({**score("Rule baseline (regex+negation)", te["rad_label"], rule_test),
             "ood_acc": round(accuracy_score(ood_y, rule_ood), 3), "trained": "no"})

# 2) TF-IDF + LOGISTIC REGRESSION (TRAINED) ---------------------------------
clf = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2),
                              min_df=1, sublinear_tf=True)),
    ("lr", LogisticRegression(max_iter=2000, class_weight="balanced")),
])
clf.fit(tr["text"], tr["rad_label"])          # <-- actual training
tfidf_test = clf.predict(te["text"])
tfidf_ood = clf.predict(ood_X)
rows.append({**score("TF-IDF + LogReg (trained)", te["rad_label"], tfidf_test),
             "ood_acc": round(accuracy_score(ood_y, tfidf_ood), 3), "trained": "yes"})

# persist the OOD predictions so we can inspect ML failure modes
json.dump(
    {"ood": [{"text": t, "gold": y, "tfidf": p}
             for t, y, p in zip(ood_X, ood_y, tfidf_ood)]},
    open(os.path.join(_OUT, "tfidf_ood_preds.json"), "w"), indent=2)

# 3) TRANSFORMER (merged if available) --------------------------------------
tdata = None
try:
    tdata = json.load(open(os.path.join(_OUT, "transformer_results.json")))
    rows.append({"model": f"{tdata['model_name']} (fine-tuned)",
                 **{k: tdata[k] for k in ["accuracy", "precision", "recall", "f1"]},
                 "ood_acc": tdata["ood_acc"], "trained": "yes"})
except FileNotFoundError:
    print("(transformer_results.json not found — run train_transformer.py to add row 3)\n")

# ---------------------------------------------------------------------------
# VOCABULARY-HELD-OUT — the clean, supervisor-endorsed generalization test.
# effusion + stones cues are held out of BOTH the rule vocabulary and the
# trained models' data; all three are then scored on impressions using them.
# sole_signal = abnormals whose ONLY signal is a held-out cue (rules cannot
# catch them once unseen) — this subset is the real generalization contrast.
# ---------------------------------------------------------------------------
ho_tr, ho_ev = heldout_split(gold)
ho_true = ho_ev["rad_label"]
sole = ho_ev[ho_ev["sole_signal"]]


def ho_scores(pred_ev, pred_sole):
    acc = accuracy_score(ho_true, pred_ev)
    _, rec, _, _ = precision_recall_fscore_support(
        ho_true, pred_ev, labels=[POS], average="binary", pos_label=POS,
        zero_division=0)
    sole_rec = float((np.asarray(pred_sole) == POS).mean()) if len(pred_sole) else 0.0
    return {"heldout_acc": round(float(acc), 3),
            "heldout_recall": round(float(rec), 3),
            "sole_signal_recall": round(sole_rec, 3)}


# rule — with the held-out cues removed from its vocabulary
P.configure_vocabulary(exclude_cues=HELDOUT_CUES)
ho_rule = ho_scores(ho_ev["text"].map(P.classify_overall),
                    sole["text"].map(P.classify_overall))
P.reset_vocabulary()

# tf-idf — retrained on the corpus WITHOUT any held-out-cue impression
clf_ho = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2),
                              min_df=1, sublinear_tf=True)),
    ("lr", LogisticRegression(max_iter=2000, class_weight="balanced")),
])
clf_ho.fit(ho_tr["text"], ho_tr["rad_label"])
ho_tfidf = ho_scores(clf_ho.predict(ho_ev["text"]),
                     clf_ho.predict(sole["text"]) if len(sole) else [])

# transformer — read the held-out retrain if the user has run it
try:
    thd = json.load(open(os.path.join(_OUT, "transformer_heldout.json")))
    ho_transformer = {k: thd[k] for k in
                      ("heldout_acc", "heldout_recall", "sole_signal_recall")}
except FileNotFoundError:
    ho_transformer = None
    print("(transformer_heldout.json not found — run "
          "`train_transformer.py --heldout` to add the transformer contrast)\n")

# attach held-out columns to the comparison rows (by trained flag / order)
rows[0].update(heldout_acc=ho_rule["heldout_acc"], heldout_recall=ho_rule["heldout_recall"])
rows[1].update(heldout_acc=ho_tfidf["heldout_acc"], heldout_recall=ho_tfidf["heldout_recall"])
if tdata is not None and ho_transformer is not None:
    rows[2].update(heldout_acc=ho_transformer["heldout_acc"],
                   heldout_recall=ho_transformer["heldout_recall"])

# REPORT --------------------------------------------------------------------
print("=" * 96)
print(f"{'model':34s} {'test_acc':>9s} {'f1':>6s} {'OOD':>6s} "
      f"{'HO_acc':>7s} {'HO_rec':>7s} {'sole_rec':>9s} {'trained':>8s}")
print("-" * 96)
_ho = {rows[0]["model"]: ho_rule, rows[1]["model"]: ho_tfidf}
if tdata is not None:
    _ho[rows[2]["model"]] = ho_transformer
for r in rows:
    h = _ho.get(r["model"]) or {}
    hoacc = f"{h['heldout_acc']:7.3f}" if h else f"{'--':>7s}"
    horec = f"{h['heldout_recall']:7.3f}" if h else f"{'--':>7s}"
    sole_r = f"{h['sole_signal_recall']:9.3f}" if h else f"{'--':>9s}"
    print(f"{r['model']:34s} {r['accuracy']:9.3f} {r['f1']:6.3f} {r['ood_acc']:6.3f} "
          f"{hoacc} {horec} {sole_r} {r['trained']:>8s}")
print("=" * 96)
print("In-distribution (test) is near-saturated for all — closed synthetic vocab.")
print("HO_* = vocabulary-held-out (effusion+stones unseen). sole_rec on the")
print("truly-unseen subset is the honest generalization number: rules -> 0,")
print("learned models recover some. THAT contrast is the story.")
json.dump(rows, open(os.path.join(_OUT, "model_comparison.json"), "w"), indent=2)

# generalization.json — everything the report page/PDF reads for the contrast
gen = {
    "heldout_cues": HELDOUT_CUES,
    "families": ["effusion", "stones"],
    "eval_total": int(len(ho_ev)),
    "eval_abnormal": int((ho_true == POS).sum()),
    "eval_normal": int((ho_true != POS).sum()),
    "sole_signal_n": int(len(sole)),
    "train_n": int(len(ho_tr)),
    "heldout": {"rule": ho_rule, "tfidf": ho_tfidf, "transformer": ho_transformer},
    "in_distribution": {
        "rule": {"acc": rows[0]["accuracy"], "recall": rows[0]["recall"],
                 "ood_acc": rows[0]["ood_acc"]},
        "tfidf": {"acc": rows[1]["accuracy"], "recall": rows[1]["recall"],
                  "ood_acc": rows[1]["ood_acc"]},
        "transformer": ({"acc": tdata["accuracy"], "recall": tdata["recall"],
                         "ood_acc": tdata["ood_acc"]} if tdata is not None else None),
    },
}
json.dump(gen, open(os.path.join(_OUT, "generalization.json"), "w"), indent=2)
print("\nwrote outputs/model_comparison.json, tfidf_ood_preds.json, generalization.json")
