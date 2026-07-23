# Radiology Impression Parser (Dashlabs P8)

Classify multilingual (Bahasa Indonesia + Latin/English) radiology impressions as
**normal / abnormal** and extract **per-organ findings**, scored against the human
`rad_label` ground truth. Two methods, kept in separate folders:

- **`rule_based/`** — hand-written regex + clause-level negation (no training)
- **`model_trained/`** — genuinely trained ML (TF-IDF+LogReg and a fine-tuned IndoBERT)

## Folder layout

```
Internship Dashlabs/
├── data/patient_service_results/   # 13 client .xlsx (inputs, read-only)
├── shared/
│   └── data_utils.py               # load_gold() + leakage-free split_by_text()  (used by both)
├── rule_based/                     # ── METHOD 1: rules ──
│   ├── parser.py                   # normalize, classify_overall, parse_organs, lexicons, evaluate
│   ├── compute_stats.py            # organ-mention stats + chart numbers
│   ├── validate_generalization.py  # overfitting / honesty checks (+ shared OOD set)
│   ├── impression_classifier.ipynb # full story: load → split → classify → evaluate
│   └── outputs/                    # predictions.csv, organ_stats.json
├── model_trained/                  # ── METHOD 2: trained ML ──
│   ├── train_models.py             # rule baseline + TF-IDF+LogReg + merges transformer → comparison
│   ├── train_transformer.py        # fine-tune IndoBERT / XLM-R
│   └── outputs/                    # model_comparison.json, generalization.json, transformer_results.json,
│                                   #   transformer_heldout.json, tfidf_ood_preds.json
└── report/
    ├── report.html                 # editable source of the PDF
    ├── Radiology_Impression_Parser_Report.pdf
    └── portfolio_content.ts         # drop-in page content (projects.ts + project-details.ts)
```

`model_trained/` reuses `rule_based/parser.py` (for the rule baseline) and the OOD
set in `rule_based/validate_generalization.py`; both read data via `shared/data_utils.py`.
Scripts add the right folders to `sys.path` automatically — just run them from their own folder.

## Results

Two axes. In-distribution (closed synthetic vocabulary) every method saturates, so
the **generalization** columns are the story. The clean test holds two whole finding
families — effusion + stones (10 cues) — out of BOTH the rule vocabulary and the
models' training data, then scores all three on impressions that use them.

| Model | Test acc | F1 | OOD acc | Held-out acc | Sole-signal recall | Trained |
|---|---|---|---|---|---|---|
| Rule baseline (regex+negation) | 1.000 | 1.000 | 0.55 | 0.69 | **0.00** | no |
| TF-IDF + LogReg | 0.908 | 0.875 | 0.75 | 0.78 | **0.58** | yes |
| IndoBERT (fine-tuned) | 0.992 | 0.990 | 0.90 | 0.89 | **0.93** | yes |

**Sole-signal recall** — recall on abnormals whose ONLY signal is a held-out cue — is
the honest generalization number. Rules score **0** (they can't match a word they've
never seen); a trained model recovers it — TF-IDF 0.58, and the fine-tuned IndoBERT
**0.93**. The 100% is real but on a closed vocabulary, so it is reported as
**secondary** context. See
`validate_generalization.py` (sections A–D) and `model_trained/outputs/generalization.json`.

> Honesty note: overall accuracy barely moves when a single finding is unseen (such
> impressions are rare in the set) — which is exactly why accuracy hides the failure
> and recall on the unseen finding is the number to trust. A fully real figure still
> needs a radiologist-labeled sample of real reports.

## Reproduce

```bash
pip install pandas openpyxl scikit-learn jupyter        # rule-based + tf-idf
pip install torch transformers accelerate               # transformer only

# rule-based
cd rule_based
python compute_stats.py
python validate_generalization.py          # sections A–D incl. vocabulary-held-out
jupyter nbconvert --to notebook --execute --inplace impression_classifier.ipynb

# trained models
cd ../model_trained
python train_models.py                      # rules + TF-IDF (fast); writes generalization.json
python train_transformer.py --epochs 5      # IndoBERT: in-distribution + OOD
python train_transformer.py --heldout --epochs 5   # IndoBERT: vocabulary-held-out contrast
python train_models.py                      # re-run to merge the transformer held-out row

# report
cd ../report
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --disable-gpu \
  --no-pdf-header-footer --print-to-pdf="Radiology_Impression_Parser_Report.pdf" "file://$PWD/report.html"
```
# Radiology-Impression-Parser
