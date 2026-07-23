"""Load the labeled radiology gold set and make a leakage-free train/dev/test split."""

import glob
import os
import pandas as pd

# shared/ lives one level below the project root; data is at <root>/data/...
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_ROOT, "data", "patient_service_results")


def load_gold(data_dir=DATA_DIR):
    """Return a DataFrame of labeled impression rows: text, client, rad_label."""
    frames = []
    for path in sorted(glob.glob(os.path.join(data_dir, "client_syn_*.xlsx"))):
        if os.path.basename(path).startswith("~$"):
            continue  # skip Excel lock/temp files
        df = pd.read_excel(path)
        df["client"] = os.path.basename(path).split("_")[2]
        frames.append(df)
    results = pd.concat(frames, ignore_index=True)

    results["text"] = results["word_value"].astype(str).str.strip()
    lab = results["rad_label"].astype(str).str.strip().str.lower()
    gold = results[
        results["word_value"].notna()
        & (results["text"] != "")
        & lab.isin(["normal", "abnormal"])
    ].copy()
    gold["rad_label"] = gold["rad_label"].str.strip().str.lower()
    return gold[["text", "client", "rad_label"]].reset_index(drop=True)


def split_by_text(gold, seed=42, frac=(0.6, 0.2, 0.2)):
    """Split on DISTINCT texts so no impression appears in two splits.

    Returns gold with an added 'split' column in {train, dev, test}.
    """
    texts = sorted(gold["text"].unique())
    # deterministic shuffle
    order = pd.Series(texts).sample(frac=1.0, random_state=seed).tolist()
    n = len(order)
    n_tr = int(n * frac[0])
    n_dev = int(n * frac[1])
    split_of = {}
    for i, t in enumerate(order):
        split_of[t] = "train" if i < n_tr else "dev" if i < n_tr + n_dev else "test"
    gold = gold.copy()
    gold["split"] = gold["text"].map(split_of)
    return gold
