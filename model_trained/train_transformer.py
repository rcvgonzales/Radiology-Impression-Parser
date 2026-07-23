"""
Fine-tune a multilingual transformer (IndoBERT / XLM-R) for overall
normal/abnormal classification — a genuinely TRAINED model to sit alongside
the rule baseline and TF-IDF+LogReg in train_models.py.

Same leakage-free split (distinct texts, seed=42). Writes transformer_results.json
which train_models.py merges into the 3-way comparison.

--heldout runs the supervisor-endorsed vocabulary-held-out test: train with the
effusion + stones impressions removed (those cues are then NEVER seen) and
evaluate on exactly those held-out-cue impressions. Writes transformer_heldout.json,
which train_models.py merges to complete the rule-vs-TF-IDF-vs-transformer contrast.

Notes
-----
- Tiny corpus (~607 distinct texts) -> a transformer is NOT expected to beat the
  rules in-distribution; the point is a fair, real comparison incl. OOD + held-out.
- Defaults to IndoBERT (indobenchmark/indobert-base-p1). Override with --model
  e.g. xlm-roberta-base. Trains on CPU/MPS; a few epochs, minutes.

Run: python train_transformer.py            # in-distribution + OOD
     python train_transformer.py --heldout   # vocabulary-held-out contrast
     python train_transformer.py --model xlm-roberta-base --epochs 4
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, set_seed)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path[:0] = [os.path.join(_ROOT, "shared"),       # data_utils
                os.path.join(_ROOT, "rule_based")]   # validate_generalization
_OUT = os.path.join(_HERE, "outputs")
os.makedirs(_OUT, exist_ok=True)

import data_utils as D
import parser as P  # noqa: F401  (kept for parity / potential reuse)
from validate_generalization import OOD, HELDOUT_CUES, heldout_split  # noqa: F401

LABELS = ["normal", "abnormal"]
L2I = {l: i for i, l in enumerate(LABELS)}
POS = "abnormal"


class DS(Dataset):
    def __init__(self, texts, labels, tok, maxlen=64):
        self.enc = tok(list(texts), truncation=True, padding="max_length",
                       max_length=maxlen)
        self.labels = [L2I[l] for l in labels]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.labels[i])
        return item


def metrics(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[1], average="binary", pos_label=1, zero_division=0)
    return dict(accuracy=round(accuracy_score(y_true, y_pred), 3),
                precision=round(float(p), 3), recall=round(float(r), 3),
                f1=round(float(f), 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="indobenchmark/indobert-base-p1")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--heldout", action="store_true",
                    help="Vocabulary-held-out mode: train with effusion+stones "
                         "impressions removed, evaluate on the held-out-cue set.")
    a = ap.parse_args()
    set_seed(a.seed)

    gold = D.load_gold()
    g = D.split_by_text(gold, seed=42)
    tr, dv, te = (g[g.split == s] for s in ["train", "dev", "test"])
    if a.heldout:
        tr, ho_ev = heldout_split(gold)   # held-out cues never appear in tr

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=2)

    ds_tr = DS(tr["text"], tr["rad_label"], tok)
    # eval_dataset is unused during training (no eval strategy); in held-out mode
    # the in-dist dev split would leak held-out vocab, so reuse the train set.
    ds_dv = ds_tr if a.heldout else DS(dv["text"], dv["rad_label"], tok)

    args = TrainingArguments(
        output_dir=os.path.join(_HERE, "_tf_out"), num_train_epochs=a.epochs,
        per_device_train_batch_size=16, per_device_eval_batch_size=32,
        learning_rate=2e-5, weight_decay=0.01, logging_steps=20,
        report_to=[], seed=a.seed,
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds_tr, eval_dataset=ds_dv)
    trainer.train()

    def predict(texts):
        enc = tok(list(texts), truncation=True, padding=True, max_length=64,
                  return_tensors="pt")
        model.eval()
        with torch.no_grad():
            logits = model(**{k: v.to(model.device) for k, v in enc.items()}).logits
        return logits.argmax(-1).cpu().numpy()

    if a.heldout:
        # evaluate on impressions using the now-unseen effusion+stones cues
        ev_pred = predict(ho_ev["text"])
        ev_true = [L2I[l] for l in ho_ev["rad_label"]]
        m = metrics(ev_true, ev_pred)
        sole = ho_ev[ho_ev["sole_signal"]]
        sole_rec = 0.0
        if len(sole):
            sole_pred = predict(sole["text"])
            sole_rec = round(float((sole_pred == L2I[POS]).mean()), 3)
        misses = [{"text": t, "gold": y, "pred": LABELS[p]}
                  for t, y, p in zip(ho_ev["text"], ho_ev["rad_label"], ev_pred)
                  if L2I[y] != p]
        out = {"model_name": a.model.split("/")[-1],
               "heldout_acc": m["accuracy"], "heldout_recall": m["recall"],
               "sole_signal_recall": sole_rec, "sole_signal_n": int(len(sole)),
               "heldout_misses": misses, "epochs": a.epochs}
        json.dump(out, open(os.path.join(_OUT, "transformer_heldout.json"), "w"),
                  indent=2)
        print("\n=== transformer (vocabulary-held-out: effusion+stones unseen) ===")
        print(json.dumps({k: out[k] for k in
                          ["model_name", "heldout_acc", "heldout_recall",
                           "sole_signal_recall", "sole_signal_n"]}, indent=2))
        print("wrote transformer_heldout.json  "
              "(re-run train_models.py to merge into the comparison)")
        return

    # test split
    te_pred = predict(te["text"])
    te_true = [L2I[l] for l in te["rad_label"]]
    m = metrics(te_true, te_pred)

    # OOD
    ood_pred = predict([t for t, _ in OOD])
    ood_true = [L2I[y] for _, y in OOD]
    ood_acc = round(accuracy_score(ood_true, ood_pred), 3)
    ood_misses = [{"text": t, "gold": y, "pred": LABELS[p]}
                  for (t, y), p in zip(OOD, ood_pred) if L2I[y] != p]

    out = {"model_name": a.model.split("/")[-1], **m, "ood_acc": ood_acc,
           "ood_misses": ood_misses, "epochs": a.epochs}
    json.dump(out, open(os.path.join(_OUT, "transformer_results.json"), "w"), indent=2)
    print("\n=== transformer (fine-tuned) ===")
    print(json.dumps({k: out[k] for k in
                      ["model_name", "accuracy", "precision", "recall", "f1", "ood_acc"]},
                     indent=2))
    print(f"OOD misses: {len(ood_misses)}")
    print("wrote transformer_results.json")


if __name__ == "__main__":
    main()
