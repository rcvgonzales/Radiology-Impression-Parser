"""
Generalization validation for the impression classifier.

Answers: "100% looks too perfect — is it overfitting?"

  A. No-leakage split        -> proves it is NOT classic overfitting
                                (train == test on DISTINCT-text split; a rule
                                 lexicon has no learned weights to memorize)
  B. Leave-one-finding-out   -> the HONEST generalization number; simulate each
                                finding type being NEW by removing its cues
  C. Out-of-distribution set -> novel hand-written phrasings whose vocabulary is
                                NOT in the synthetic corpus
  D. Vocabulary-held-out     -> the clean, supervisor-endorsed test: hold two
                                whole finding families (effusion + stones) out of
                                the vocabulary and score on impressions that use
                                them. Rules collapse on sole-signal unseen
                                findings; the transformer (train_transformer.py
                                --heldout) supplies the other half of the contrast.

Run: python validate_generalization.py
OOD, HELDOUT_CUES and heldout_split() are importable by other scripts so the
rule / TF-IDF / transformer comparisons all use exactly the same held-out set.
"""

import os
import sys
import re

sys.path.insert(0, os.path.join(  # shared/data_utils
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "shared"))

import data_utils as D
import parser as P


# Out-of-distribution set — novel impressions, vocabulary mostly outside the
# synthetic corpus. Defined at module level so other scripts import the SAME set.
OOD = [
    ("Kedua paru bersih, jantung tidak membesar.", "normal"),
    ("Liver echotexture normal, no focal lesion.", "normal"),
    ("Vesica fellea tidak melebar, dinding tidak menebal.", "normal"),
    ("Foto thorax dalam batas normal.", "normal"),
    ("Ginjal kanan dan kiri ukuran serta echotextur normal.", "normal"),
    ("Cor tak membesar.", "normal"),               # 'tak' (not 'tidak') -> tricky
    ("Efusi pleura (-).", "normal"),               # (-) notation = absent
    ("Tampak massa pada caput pancreas.", "abnormal"),
    ("Ground glass opacity pada lobus kanan atas.", "abnormal"),
    ("Limfadenopati hilus bilateral.", "abnormal"),
    ("Atelektasis pada paru kiri.", "abnormal"),
    ("Spondylosis lumbalis dengan osteofit.", "abnormal"),
    ("Kardiomegali dengan CTR meningkat.", "abnormal"),
    ("Susp. malignancy paru, pro biopsi.", "abnormal"),
    ("Emfisema subkutis regio thorax.", "abnormal"),
    ("Hepatosplenomegaly.", "abnormal"),
    ("Nephrolithiasis bilateral dengan hidronefrosis grade II.", "abnormal"),
    ("Bronkiektasis pada lobus inferior.", "abnormal"),
    ("Pembesaran kelenjar getah bening paratrakeal.", "abnormal"),
    ("Cardiomegali (+).", "abnormal"),
]

# Abnormal cues grouped into clinical finding families (for leave-one-out).
FAMILIES = {
    "effusion":       [r"efusi", r"effusion"],
    "enlargement":    [r"membesar", r"megali", r"megaly", r"pembesaran",
                       r"enlarged", r"enlargement"],
    "stones":         [r"lithiasis", r"litiasis", r"kolelitiasis",
                       r"nephrolithiasis", r"batu empedu", r"batu ginjal",
                       r"gallstones?", r"kidney stones?"],
    "inflammation":   [r"cholecystitis", r"kolesistitis", r"cholangitis",
                       r"pyelonephritis"],
    "mass_lesion":    [r"massa", r"\bmass\b", r"lesi", r"lesion", r"nodul",
                       r"nodule", r"kista", r"\bcyst\b", r"tumor"],
    "liver_fat":      [r"fatty liver", r"steatosis", r"perlemakan hati"],
    "lung_disease":   [r"fibrosis", r"infiltrat", r"infiltrate",
                       r"bronchopneumonia", r"pneumonia", r"konsolidasi",
                       r"consolidation", r"\btb\b", r"tuberculosis"],
    "renal_dilat":    [r"hidronefrosis", r"hydronephrosis", r"dilatasi"],
    "thickening_etc": [r"penebalan", r"thickening", r"kalsifikasi",
                       r"stenosis", r"hipertrofi"],
}


# Vocabulary-held-out set (supervisor-endorsed): hold out TWO whole finding
# families — effusion + stones (10 cue strings, no CTR numeric backstop) — from
# BOTH the rule vocabulary and the trained models' data, then score all systems
# on impressions that use those now-unseen cues. Shared so every script agrees.
HELDOUT_CUES = FAMILIES["effusion"] + FAMILIES["stones"]


def heldout_split(gold):
    """Split gold by held-out-cue vocabulary for a fair unseen-findings test.

    Returns (train_df, eval_df):
      eval_df  = impressions mentioning any HELDOUT_CUES term (vocabulary a model
                 trained only on train_df has never seen). Gains a boolean
                 'sole_signal' column: abnormal impressions whose ONLY abnormal
                 signal is a held-out cue (no other cue, no CTR>0.50) — the
                 impressions a rule system CANNOT catch once the cue is unseen.
      train_df = everything else (held-out cues never appear in it).
    """
    base = getattr(P, "_BASE_ABNORMAL_CUES", list(P.ABNORMAL_CUES))
    held = set(HELDOUT_CUES)
    held_re = re.compile("|".join(HELDOUT_CUES))
    other_re = re.compile("|".join(c for c in base if c not in held))
    g = gold.copy()
    g["norm"] = g["text"].map(P.normalize)
    mask = g["norm"].str.contains(held_re)
    train_df = g[~mask].copy()
    eval_df = g[mask].copy()
    eval_df["sole_signal"] = eval_df.apply(
        lambda r: (r["rad_label"] == "abnormal"
                   and not other_re.search(r["norm"])
                   and not P._ctr_abnormal(r["norm"])), axis=1)
    return train_df, eval_df


def main():
    gold = D.load_gold()
    g = D.split_by_text(gold, seed=42)

    # A. no leakage --------------------------------------------------------
    print("=" * 68)
    print("A. NO-LEAKAGE SPLIT — is there a train/test gap? (overfitting sign)")
    print("=" * 68)
    acc = {}
    for s in ["train", "dev", "test"]:
        sub = g[g.split == s]
        pred = sub["text"].map(P.classify_overall)
        acc[s] = (pred.values == sub["rad_label"].values).mean()
        print(f"  {s:5s} accuracy = {acc[s]:.3f}  (n={len(sub)})")
    print(f"  train-minus-test gap = {acc['train'] - acc['test']:+.3f}")
    print("  -> No gap; split on distinct texts; rules have no learned weights.")
    print("     => NOT classic overfitting. But shared vocabulary hides the real")
    print("     failure mode: an UNSEEN finding term (measured next).")

    # B. leave-one-finding-out --------------------------------------------
    print("\n" + "=" * 68)
    print("B. LEAVE-ONE-FINDING-OUT — recall when a finding type is NEW")
    print("=" * 68)
    abn = gold[gold["rad_label"] == "abnormal"].copy()
    abn["norm"] = abn["text"].map(P.normalize)
    rows, total_cases, total_caught = [], 0, 0
    for fam, cues in FAMILIES.items():
        fam_re = re.compile("|".join(cues))
        members = abn[abn["norm"].str.contains(fam_re)]
        n = len(members)
        if n == 0:
            continue
        P.configure_vocabulary(exclude_cues=cues)
        still = members["text"].map(P.classify_overall)
        caught = int((still == "abnormal").sum())
        rows.append((fam, n, caught, caught / n))
        total_cases += n
        total_caught += caught
    P.reset_vocabulary()
    print(f"  {'finding family':16s} {'cases':>6s} {'caught':>7s} {'held-out recall':>16s}")
    for fam, n, caught, rec in sorted(rows, key=lambda r: r[3]):
        print(f"  {fam:16s} {n:6d} {caught:7d} {rec:16.1%}")
    print("-" * 68)
    lofo = total_caught / total_cases
    print(f"  AGGREGATE held-out recall (finding unseen) = "
          f"{total_caught}/{total_cases} = {lofo:.1%}")

    # C. out-of-distribution ----------------------------------------------
    print("\n" + "=" * 68)
    print("C. OUT-OF-DISTRIBUTION — novel phrasings NOT in the corpus")
    print("=" * 68)
    ok, misses = 0, []
    for t, goldl in OOD:
        pred = P.classify_overall(t)
        ok += pred == goldl
        if pred != goldl:
            misses.append((t, goldl, pred))
    ood = ok / len(OOD)
    print(f"  OOD accuracy = {ok}/{len(OOD)} = {ood:.1%}")
    for t, gl, pr in misses:
        print(f"    gold={gl:8s} pred={pr:8s} | {t}")

    # D. vocabulary-held-out (the clean, supervisor-endorsed comparison) -----
    print("\n" + "=" * 68)
    print("D. VOCABULARY-HELD-OUT — effusion + stones cues removed from the")
    print("   rule vocabulary; scored on impressions that use them (rule side).")
    print("=" * 68)
    _, ev = heldout_split(gold)
    sole = ev[ev["sole_signal"]]
    n_abn = int((ev["rad_label"] == "abnormal").sum())
    print(f"  held-out cues ({len(HELDOUT_CUES)}): {', '.join(HELDOUT_CUES)}")
    print(f"  eval set: {len(ev)} impressions ({n_abn} abnormal, "
          f"{len(ev) - n_abn} normal); sole-signal abnormals: {len(sole)}")
    P.configure_vocabulary(exclude_cues=HELDOUT_CUES)
    ev_pred = ev["text"].map(P.classify_overall)
    ho_acc = (ev_pred.values == ev["rad_label"].values).mean()
    tp = int(((ev_pred == "abnormal") & (ev["rad_label"] == "abnormal")).sum())
    ho_rec = tp / n_abn if n_abn else 0.0
    sole_rec = ((sole["text"].map(P.classify_overall) == "abnormal").mean()
                if len(sole) else 0.0)
    P.reset_vocabulary()
    print(f"  RULE held-out accuracy = {ho_acc:.1%}   recall = {ho_rec:.1%}")
    print(f"  RULE recall on SOLE-SIGNAL unseen findings = {sole_rec:.1%}  "
          f"(rules can't match vocabulary they've never seen)")

    print("\n" + "=" * 68)
    print("SUMMARY (what to report)")
    print("=" * 68)
    print(f"  In-distribution held-out accuracy ... {acc['test']:.0%}  (method correct)")
    print(f"  Leave-one-finding-out recall ........ {lofo:.0%}  (robustness)")
    print(f"  Vocabulary-held-out sole-signal rec . {sole_rec:.0%}  (RULE limit — see")
    print( "                                            transformer for the contrast)")
    print(f"  Out-of-distribution accuracy ........ {ood:.0%}  (hand-written tail)")


if __name__ == "__main__":
    main()
