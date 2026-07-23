"""Compute all portfolio-page numbers from the real synthetic dataset.

Outputs:
  - overall classifier metrics (train/dev/test)  [from earlier work]
  - organ-mention frequencies (for the bar chart)
  - per-organ normal/abnormal/not_mentioned distribution
  - consistency: overall-derived-from-organs vs rad_label
  - headline corpus stats (impressions, distinct, normal rate, languages, organs)
Writes organ_stats.json for the page content.
"""

import os
import sys
import json
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "shared"))  # shared/data_utils
_OUT = os.path.join(_HERE, "outputs")
os.makedirs(_OUT, exist_ok=True)

import data_utils as D
import parser as P

gold = D.load_gold()
g = D.split_by_text(gold, seed=42)

# --- overall classifier (recap) ---
print("=" * 64)
print("OVERALL normal/abnormal classifier (positive='abnormal')")
split_acc = {}
for s in ["train", "dev", "test"]:
    sub = g[g.split == s]
    r = P.evaluate(sub["rad_label"], sub["text"].map(P.classify_overall), title=s)
    split_acc[s] = round(float(r["accuracy"]), 3)

# --- organ-level parse over the full labeled corpus ---
print("\n" + "=" * 64)
print("ORGAN-LEVEL parse over", len(gold), "impressions")
organ_status = gold["text"].map(P.parse_organs)
mentions = Counter()
per_organ = {o: Counter() for o in P.ORGANS}
for st in organ_status:
    for o in P.ORGANS:
        per_organ[o][st[o]] += 1
        if st[o] != "not_mentioned":
            mentions[o] += 1

print("\nOrgan mentions (impressions naming/implying the organ):")
for o, n in mentions.most_common():
    print(f"  {o:8s} {n}")

print("\nPer-organ status distribution:")
for o in P.ORGANS:
    print(f"  {o:8s} {dict(per_organ[o])}")

# --- consistency check: overall-from-organs vs gold rad_label ---
print("\n" + "=" * 64)
derived = gold["text"].map(P.classify_overall_from_organs)
agree = (derived.values == gold["rad_label"].values).mean()
print(f"overall-derived-from-organ-parse vs rad_label: {agree:.1%} agreement")
m = P.evaluate(gold["rad_label"], derived, title="organ-derived overall (full corpus)")

# --- headline corpus stats ---
def lang(t):
    t = str(t).lower()
    id_m = ["dalam batas", "tidak tampak", "tampak", "ginjal", "hepar", "membesar"]
    en_m = ["unremarkable", "lungs", "liver", "kidney", "normal limits", "lesion"]
    i = any(x in t for x in id_m); e = any(x in t for x in en_m)
    return "mixed" if i and e else "id" if i else "en" if e else "other"

langs = Counter(gold["text"].map(lang))
stats = {
    "impressions": int(len(gold)),
    "distinct_texts": int(gold["text"].nunique()),
    "normal_rate": round(float((gold["rad_label"] == "normal").mean()), 3),
    "organs_tracked": len(P.ORGANS),
    "organ_mentions": dict(mentions.most_common()),
    "languages": dict(langs),
    "test_accuracy": split_acc["test"],  # computed on the held-out test split
    "organ_derived_overall_agreement": round(float(agree), 3),
}
print("\n" + "=" * 64)
print("HEADLINE STATS:", json.dumps(stats, indent=2))
json.dump(stats, open(os.path.join(_OUT, "organ_stats.json"), "w"), indent=2)
print("\nwrote outputs/organ_stats.json")
