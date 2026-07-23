/**
 * Radiology Impression Parser — portfolio page content.
 *
 * Drop-in content for the Dashlabs portfolio template. All numbers are REAL,
 * computed on the 679 labeled synthetic impressions (gold column = rad_label):
 *   - corpus / organ stats ....... rule_based/outputs/organ_stats.json
 *   - model + generalization ..... model_trained/outputs/generalization.json
 *
 * Framing follows supervisor feedback: the HEADLINE is generalization to
 * UNSEEN findings; the 100% closed-vocabulary result is secondary context.
 *
 * All three models' held-out numbers are final (fine-tuned IndoBERT via
 *   `python train_transformer.py --heldout` → merged by train_models.py).
 *
 * NOTE: field names below follow the brief's described shape (ProjectMeta +
 * six detail sections + analysis.charts of {label, value}). After you clone the
 * forked repo, align these keys with the exact TS types in
 *   src/content/projects.ts          (ProjectMeta)
 *   src/content/project-details.ts   (the six sections)
 * — rename fields if the repo's interfaces differ; the prose/values are final.
 */

// ── src/content/projects.ts ────────────────────────────────────────────────
export const radiologyMeta = {
  slug: "radiology-impression-parser",
  title: "Radiology Impression Parser",
  tagline: "Parse multilingual radiology impressions into structured organ-level findings.",
  difficulty: "Intermediate",
  techniques: [
    "Regex Extraction",
    "Negation-aware Rules",
    "Clinical NLP",
    "TF-IDF + LogReg",
    "Fine-tuned IndoBERT",
    "Generalization testing",
  ],
  keyMetric: {
    label: "Recall on truly-unseen findings (held-out cues)",
    value: "0% → 93%",
    // rules → fine-tuned IndoBERT, on the sole-signal held-out set. Hand-written
    // rules score 0% on findings whose vocabulary they never saw; learned models
    // recover it (TF-IDF 58%, IndoBERT 93%). In-distribution every model hits
    // ~100% on the CLOSED synthetic vocabulary — secondary context, NOT the headline.
  },
};

// ── src/content/project-details.ts ─────────────────────────────────────────
export const radiologyDetails = {
  // 01 -----------------------------------------------------------------------
  businessProblem: {
    heading: "Can we structure free-text radiology impressions — and will it hold up on findings it has never seen?",
    body:
      "Thirteen synthetic diagnostic-lab clients store 679 imaging impressions " +
      "(USG abdomen and chest X-ray) as free text — Bahasa Indonesia mixed " +
      "freely with Latin and English organ terms. A human-entered rad_label " +
      "column marks each as normal or abnormal (~60% normal). A parser can turn " +
      "each narrative into a structured per-organ finding plus an overall " +
      "normal/abnormal flag. The harder, more honest question is generalization: " +
      "on a CLOSED synthetic vocabulary every method looks near-perfect, so the " +
      "project deliberately measures what happens when a finding's vocabulary is " +
      "UNSEEN — where a hand-written rule set and a trained model part ways.",
    stats: [
      { label: "Problem Type", value: "Clinical NLP — Multilingual" },
      { label: "Methods", value: "Rules · TF-IDF+LogReg · IndoBERT" },
      { label: "Impressions", value: "679" },
      { label: "Normal Rate", value: "60%" },
    ],
  },

  // 02 -----------------------------------------------------------------------
  dataSources: [
    {
      table: "patient_service_results",
      role: "PRIMARY",
      columns: [
        "word_value — free-text impression",
        "service_name — Ultrasound / USG Abdomen / Rontgen Dada",
        "rad_label — human normal/abnormal ground truth",
      ],
    },
    {
      table: "patient_services",
      role: "context",
      columns: ["service_name", "status", "created_at"],
    },
  ],

  // 03 -----------------------------------------------------------------------
  methodology: {
    heading: "Rules built around negation — then stress-tested for generalization.",
    steps: [
      {
        n: "01",
        title: "Corpus Analysis",
        body:
          "Catalog phrasings, synonyms (gallbladder / vesica fellea / batu " +
          "empedu), and the many negation forms (tidak tampak, no, unremarkable).",
      },
      {
        n: "02",
        title: "Normalize + clause/negation rules",
        body:
          "Lowercase, strip section prefixes and trailing recommendations; split " +
          "into clauses; flag abnormal only on an un-negated finding cue. " +
          "“Abnormal wins” on mixed sentences.",
      },
      {
        n: "03",
        title: "Train real models",
        body:
          "Alongside the rules, genuinely train TF-IDF + LogReg and fine-tune " +
          "IndoBERT on the same leakage-free split (distinct texts, seed 42) — a " +
          "fair three-way comparison, not just the rules.",
      },
      {
        n: "04",
        title: "Vocabulary-held-out generalization test",
        body:
          "Hold two whole finding families — effusion + stones (10 cues) — out of " +
          "BOTH the rule vocabulary AND the models' training data, then score all " +
          "three on impressions that use those now-unseen cues. The sole-signal " +
          "subset (unseen cue is the only signal) is the honest generalization number.",
      },
    ],
  },

  // 04 -----------------------------------------------------------------------
  analysis: {
    heading: "Structured findings extracted — and generalization measured honestly.",
    note:
      "Computed on the labeled synthetic dataset (679 impressions). Held-out " +
      "test: effusion + stones cues removed from both the rule vocabulary and " +
      "the models' training data (405 train / 274 eval, of which 40 are " +
      "sole-signal abnormals).",
    metrics: [
      { label: "In-distribution accuracy", value: "~100%", sub: "closed synthetic vocab (secondary)" },
      { label: "Unseen-finding recall — Rules", value: "0%", sub: "sole-signal held-out" },
      { label: "Unseen-finding recall — IndoBERT", value: "93%", sub: "sole-signal held-out" },
      { label: "Impressions", value: "679", sub: "labeled (rad_label)" },
      { label: "Held-out eval set", value: "274", sub: "40 sole-signal abnormals" },
    ],
    // analysis.charts — renders as bars (label / value pairs)
    charts: [
      {
        title: "Recall on truly-unseen findings (sole-signal held-out)",
        subtitle: "effusion + stones held out of vocabulary AND training — the honest generalization number",
        data: [
          { label: "Rule (regex+negation)", value: 0 },
          { label: "TF-IDF + LogReg", value: 58 },
          { label: "IndoBERT (fine-tuned)", value: 93 },
        ],
      },
      {
        title: "Generalization to novel phrasings — OOD accuracy",
        subtitle: "20 hand-written impressions whose vocabulary is outside the corpus (all three models)",
        data: [
          { label: "Rule (regex+negation)", value: 55 },
          { label: "TF-IDF + LogReg", value: 75 },
          { label: "IndoBERT (fine-tuned)", value: 90 },
        ],
      },
      {
        title: "Organ Mentions Across Impressions",
        subtitle: "Impressions naming or implying each organ",
        data: [
          { label: "hepar", value: 325 },
          { label: "pulmo", value: 314 },
          { label: "ginjal", value: 295 },
          { label: "vesica", value: 284 },
          { label: "cor", value: 273 },
          { label: "pleura", value: 230 },
        ],
      },
    ],
  },

  // 05 -----------------------------------------------------------------------
  insights: [
    {
      title: "Rules can't generalize to unseen vocabulary — learned models can",
      body:
        "When effusion + stones cues are held out of everything, the rule system " +
        "scores 0% recall on impressions where an unseen cue is the only signal — " +
        "it simply cannot match a word it has never seen. A trained TF-IDF model " +
        "recovers 58% from surrounding context, and the fine-tuned IndoBERT 93%. " +
        "That contrast — not the 100% — is the real story.",
    },
    {
      title: "The 100% is on a closed vocabulary, and we say so",
      body:
        "In-distribution the data is synthetic from a fixed phrase set, so the " +
        "held-out test shares its vocabulary and every model saturates (~100%). " +
        "A leave-one-cue-out probe and the vocabulary-held-out test above make " +
        "that limitation explicit: the closed-vocab score reflects method " +
        "correctness, not real-world generalization.",
    },
    {
      title: "Negation is the biggest lever",
      body:
        "A naive “mentions normal” heuristic scored only 63.8% and missed most " +
        "abnormals — findings like “cor membesar” sit right beside the word " +
        "“normal”. Handling “no / tidak tampak” at clause level is what lifts the " +
        "rules to 100% in-distribution.",
    },
    {
      title: "Synonyms bridge languages",
      body:
        "One organ surfaces many ways — gallbladder / vesica fellea / batu " +
        "empedu; ginjal / kidney / renal. A synonym map makes the same rules work " +
        "across Bahasa, Latin, and English in a single sentence.",
    },
  ],

  // 06 -----------------------------------------------------------------------
  futureImprovements: [
    {
      title: "Widen the held-out sweep",
      body: "More finding families and an XLM-R comparison to map where the generalization gap opens.",
    },
    {
      title: "Radiologist-labeled real reports",
      body: "The only fully honest generalization figure needs a sample of real, human-labeled impressions.",
    },
    {
      title: "Per-organ gold labels",
      body: "Hand-label organ findings to score the structured extraction directly, not just the overall flag.",
    },
    {
      title: "PACS integration & trends",
      body: "Link findings to images; track organ findings per patient across visits.",
    },
  ],
};
