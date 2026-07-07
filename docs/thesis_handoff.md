# Thesis Handoff — Violin Expressiveness → Emotion Recognition

**Purpose of this document:** complete context so work can continue in a fresh chat. Covers (1) the original project, (2) all progress made recently, (3) how the workflow is now set up, and (4) what comes next.

> **Doc revision note:** This version corrects several facts that had drifted from the code:
> the rating scale is **0–6** (not 1–6); the questionnaire-cleaning results are from the current
> 0–6-corrected run (**118 screened, 3 flagged**); MERT averages hidden layers **[5, 6, 7]**
> (three mid-stack layers, not "last 4"); stimulus screening now lives in a **separate**
> `excerpt_cleaning.py`; and Supervisor Notes 1, 2, 3, 4, 7 are now **implemented in the notebook**.
> A new **§9 Open Issues** section lists the interpretation/robustness items to resolve before
> writing the results chapters.

---

## 1. Project Overview

**Thesis question:** Do large-model SSL embeddings outperform hand-crafted low-level acoustic features for predicting perceived emotion from expressive solo violin? Does combining both outperform either alone? And more fundamentally — **which performance techniques are associated with which emotions?**

**Dataset:**
- 60 audio excerpts = 20 pieces × 3 performance conditions
  - **MEC** = mechanical (no expression, baseline)
  - **EXP** = expressive (natural)
  - **EXG** = exaggerated (over-the-top expressive)
- **118 participants** across 5 listening sessions rated each excerpt on **valence (0–6)**, **arousal (0–6)**, and tagged perceived emotions.
  - **Scale is 0–6 (7 points, integer 0…6, midpoint 3.0).** `0` is a genuine "lowest" rating (verified against the raw forms — a `0` co-occurs with normal arousal values and filled-in tags, e.g. valence=0 on the sad Schindler's List excerpts). It is **not** a skip/NA marker. `VA_MID = 3.0` in the pipeline is therefore correct.
- 8 emotion labels derived from valence/arousal circumplex quadrants: Tenderness, Nostalgia, Peacefulness, Power, Joyful Activation, Tension, Sadness, Neutral.

**Environment:** Kaggle notebook, P100 GPU. PyTorch pinned to 2.3.1 (last version supporting P100's sm_60 compute capability).

---

## 2. Pipeline Architecture (cell-by-cell)

The notebook `thesis_pipeline.ipynb` runs top to bottom:

1. **Install / Imports / Config** — environment setup, PyTorch 2.3.1 pin
2. **Parse Questionnaire CSVs** (§4) — reads 5 session CSVs into long-format participant responses (keeps `0` ratings; no 1–6 clamp)
3. **Aggregate per excerpt + derive labels** (§5) — collapses to one row per excerpt; means/SDs for VA; places each excerpt in a VA quadrant (split at `VA_MID = 3.0`) → emotion labels. MEC excerpts are assigned `Neutral` by rule.
4. **Label Matrix** (§6) — builds `Y`, the 60×8 binary label matrix
5. **§6A Statistical Emotion Analysis** — participant-only descriptives + Friedman + post-hoc (see 3.8)
6. **§6B Lift Analysis** — emotion–condition binding (see 3.8)
7. **MERT** (§7) — SSL embedding, 768-dim, **averaged over hidden layers [5, 6, 7]** (three mid-stack transformer layers). The primary "black box" feature set.
8. **CREPE** (§8) — pitch tracker → 8 interpretable features: vibrato rate/depth, portamento count/extent, F0 range/CV/mean, voiced fraction. The interpretable "low-level" set.
9. **§8A Dominant-frequency demo** — single-excerpt CREPE F0 vs loudest partial vs spectral centroid (Supervisor Note 1)
10. **Skip stubs** (§9–12) — Madmom, DTW, Essentia DSP, MusiCNN (validation mode)
11. **Feature assembly** (§13) — three matrices: A=MERT (60×768), B=CREPE (60×8), C=Combined (60×776)
12. **§13A Exploratory Data Analysis** — label structure, VA distribution, CREPE sanity, MERT PCA/t-SNE/silhouette (Supervisor Note 7)
13. **Classification** (§14) — 8 classifiers × 3 feature sets, Leave-One-Out Cross-Validation (LOOCV), multi-label
14. **VA Regression** (§15) — Ridge/SVR/MLP predict continuous valence & arousal (currently MERT only)
15. **MFCC baseline** (§16) — generic 13-MFCC + spectral features + SVM
16. **Ablation + Wilcoxon** (§17) — results table + significance tests across feature sets
17. **§17A Expanded Classifier Metrics** — per-label precision/recall/F1 + Dummy baselines (Supervisor Note 3)
18. **Analyses A–D** (§18–21) — per-condition F1, quadrant accuracy, Krippendorff's α, embedding-distance attribution
19. **§E1 Condition-Delta (Path B)** + **§E2 SHAP importance (Path A)** — technique–emotion association (see 3.7)
20. **§E3 MERT ↔ CREPE Correspondence** — correlation map + linear probe (Supervisor Note 2, MERT side)
21. **§E4 Feature Selection** — leakage-free k-sweep + stability (Supervisor Note 4)
22. **Figures + Save** (§22–23) — all plots and CSV exports

---

## 3. Progress Made Recently (chronological)

### 3.1 — Removed all librosa fallbacks
The original pipeline had `try/except` blocks that silently fell back to librosa if a specialized library (crepe, madmom, essentia) wasn't installed. These were removed in favor of **hard imports that fail loudly**. Rationale: silent degradation produced wrong-dimensional features and made debugging impossible. Now if a library is missing, the notebook stops with a clear error instead of producing bad data.

### 3.2 — Diagnosed the essentia TF-models warning
The `⚠️ essentia TF models not available — mel fallback active` message was traced to its root cause: `TensorflowPredictMusiCNN` lives only in the separate `essentia-tensorflow` package, not the DSP-only `essentia` package. Fix was adding `pip install essentia-tensorflow --no-deps` (the `--no-deps` flag is critical to avoid pulling a TensorFlow that conflicts with the pinned PyTorch). *Note: MusiCNN is currently skipped anyway (see 3.4), so this matters only when the full pipeline is re-enabled.*

### 3.3 — Reduced to validation mode (MERT + CREPE only)
To validate the pipeline end-to-end without the confounding complexity of all five feature extractors, the pipeline was cut down to **one SSL feature (MERT)** and **one low-level feature (CREPE)**. The four other extractors — **Madmom** (timing/rhythm), **DTW** (alignment), **Essentia DSP** (timbre/loudness), **MusiCNN** (2nd DNN embedding) — were replaced with skip stubs. They are NOT deleted; re-enabling is just restoring those four cells and expanding the feature-assembly lists. Feature sets became A=MERT, B=CREPE, C=Combined.

### 3.4 — Removed the Neutral Gate
Originally the 20 MEC excerpts were excluded from classification and auto-assigned Neutral. This gate was **removed** — all 60 excerpts now go through classification, MEC keeps its VA-derived Neutral label like any other excerpt. This raised N from 40 → 60 for LOOCV (statistically better) and made the task richer (distinguishing expressive from non-expressive playing). *Caveat: because MEC excerpts are still labelled Neutral by rule, the `Neutral` label remains perfectly confounded with the MEC condition — see §9.*

### 3.5 — Fixed the classification crash (SafeMultiOutputClassifier)
`ValueError: The number of classes has to be greater than one` was crashing the classifiers. Cause: in LOOCV with rare emotion labels, some training folds had all-zero columns for a label (0 positive examples), and SVM requires ≥2 classes. Fix: wrote **`SafeMultiOutputClassifier`**, which fits one binary classifier per label but predicts the majority class (0) for any degenerate single-class column instead of crashing. This is scientifically correct (no positive evidence → predict absent) and can be described as "degenerate folds handled by majority-class imputation."

### 3.6 — Split cleaning into two standalone scripts (participant + excerpt)
Screening is now **two** scripts under `scripts/`:

**`questionnaire_cleaning.py` — participant screening.** Screens all 118 participants for response quality using 5 methods:
- **Robust Mahalanobis Distance (MCD)** — PRIMARY, supervisor-recommended. Each participant is a point in 40-dim space (20 valence + 20 arousal ratings). Uses Minimum Covariance Determinant to estimate the "typical" response from the cleanest 75% of participants, avoiding the masking effect. Threshold: √χ²(40, 0.001) = 8.57.
- **Classical MD**, **Person-Total Correlation (leave-one-out)**, **IRV** (straight-lining detection), **LongString** (consecutive identical responses).
- **Combined flag** — excluded only if flagged by **≥2 independent methods** (robust MD, person-r, IRV, LongString; classical MD is excluded from the vote as redundant with robust MD).

**`excerpt_cleaning.py` — stimulus/excerpt screening.** Replaces v1's mathematically-invalid per-column "ICC" with correct diagnostics: rater coverage, dispersion (SD/IQR), rWG agreement, tag/valence entropy, bimodality (Hartigan dip test if `diptest` installed, else Sarle's BC), **one-way panel ICC(1)/ICC(1,k)** (correct model — each excerpt is rated by a different random set of listeners, so raters are not crossed with targets), and a per-piece **manipulation check** (does mean arousal follow MEC ≤ EXP ≤ EXG?). It **flags, never auto-removes** — disagreement is often the finding.

**Results on the real data (0–6-corrected run):**
- **Participants:** 118 screened. Per method: robust MD 30, classical MD 1, person-r (LOO) 4, IRV 1, LongString 4. **3 flagged by ≥2 methods:**
  - `S5_P012` — MD-robust, MD-std, person-r (MD_rob=18.21, person_r=−0.171)
  - `S5_P021` — MD-robust, LongString (LongString=9)
  - `S1_P006` — IRV, LongString (IRV=2.57, LongString=8)
  - Flagged participants are **NOT auto-removed**; decision pending review with supervisor. The pipeline currently runs on **all 118** (no exclusions applied).
- **Excerpts:** Panel reliability is high — valence ICC(1,k)=0.963, arousal ICC(1,k)=0.952 (single-rater ICC(1)=0.396/0.335; avg k≈39). **46 of 60 excerpts** flagged low-agreement/high-dispersion (diagnostic only — high dispersion is expected for emotion ratings, not grounds for removal). **3 arousal inversions** (EXG not more arousing than MEC): Bach_Adagio (−0.53), Mendelssohn_VC_Mvt2 (−0.17), La_Vita_E_Bella (−0.13). **12 of 20 pieces** show no statistically distinguishable arousal across conditions (Kruskal–Wallis p>0.05) — see §9.

### 3.6b — Fixed the 0–6 scale bug in the cleaning scripts
Both cleaning scripts previously assumed a **1–6** scale (`VA_SCALE_MIN = 1`) and silently converted every `0` rating to `NaN`, deleting ~3–4% of real data concentrated on low-valence (sad) excerpts and bottom-of-scale participants. The excerpt script additionally used `N_SCALE_POINTS = 6` in the rWG uniform-null variance (should be 7), which understated the null and pushed all rWG values down. **Both fixed:** `VA_SCALE_MIN = 0`, `N_SCALE_POINTS = 7`, histogram bins start at 0. Effect: the flagged-participant list changed (only `S5_P012` is stable across the old and new runs), contested-excerpt count dropped 54 → 46, and inversions dropped 4 → 3. **The main analysis pipeline was never affected** — its parse always kept `0` ratings and `VA_MID = 3.0` was already correct.

### 3.7 — Added Technique–Emotion Association analysis (the core thesis question)
Two complementary analyses plus figures (TE1–TE3):

**Path A — SHAP Feature Importance (§E2):** For each of the 8 emotions, trains a RandomForest (400 trees, balanced weights) on the 8 CREPE features across all 60 excerpts, then uses SHAP TreeExplainer to compute how much each technique drives each emotion. Produces an 8×8 emotion×technique importance matrix, ranked bar charts, and a beeswarm plot. Saved: `shap_technique_importance.csv`.

**Path B — Condition-Delta Analysis (§E1):** For each of the 20 pieces, computes how much each CREPE technique feature changed between MEC→EXP and EXP→EXG, and correlates those deltas with the corresponding ΔValence/ΔArousal from listeners (within-subject evidence). Produces Pearson-r correlation matrices with significance stars. Saved: `analysis_condition_deltas.csv`. *Caveat: currently uncorrected for multiple comparisons — see §9.*

**Why both:** Path B is the performer's side ("when I added vibrato, did listeners feel it?"), Path A is the model's side ("which features predict each emotion?"). Convergence between them = evidence.

### 3.8 — Added Statistical Emotion Analysis + Lift (§6A, §6B)
Two participant-only analyses (no classifiers):

**§6A — Statistical Emotion Analysis:** Mean/SD valence & arousal per condition; emotion-label frequency per condition; **Friedman test** on piece-level condition means (repeated-measures) for valence and arousal, with Kendall's W effect size; post-hoc Wilcoxon signed-rank tests with Bonferroni correction. **Result:** valence Friedman **n.s.** (p=0.39); arousal significant (p=0.001) driven by the MEC↔EXG contrast. Saved: `participant_stats_by_condition.csv`, `fig_va_by_condition.png`.

**§6B — Lift Analysis:** `lift = P(emotion|condition) / P(emotion overall)`, using **raw participant tags** (not VA-derived labels, to avoid circularity). Threshold-based incidence (≥30% of raters), with a sensitivity sweep [0.20, 0.30, 0.40] and a threshold-free mean-share robustness check. **Result:** only the Neutral↔MEC binding is threshold-stable; the expressive-condition bindings are threshold-sensitive. Saved: `lift_matrix.csv`, `fig_lift_heatmap.png`.

### 3.9 — Implemented the remaining Supervisor Notes in the notebook
Since the last handoff, Notes 1, 2, 3, 4, and 7 have been implemented (see §6 for status): §8A (dominant frequency), §13A (EDA), §17A (expanded metrics + Dummy baselines), §E3 (MERT↔CREPE correspondence via correlation map + linear probe), §E4 (leakage-free feature selection + stability).

---

## 4. Current Pipeline Status

| Component | Status |
|---|---|
| MERT embeddings (SSL, 768-dim, layers [5,6,7]) | ✅ Active, all 60 excerpts |
| CREPE features (low-level, 8-dim) | ✅ Active, interpretable |
| Label matrix Y (60×8) | ✅ All 60 excerpts, no neutral gate |
| LOOCV classification (8 classifiers × 3 sets) | ✅ Fixed (SafeMultiOutputClassifier) |
| VA regression (Ridge/SVR/MLP) | ✅ Active (MERT only — add CREPE/Combined/baseline, §9) |
| MFCC baseline | ⚠️ Active but degenerate (F1-macro 0.0 — see §9) |
| Participant cleaning (`questionnaire_cleaning.py`) | ✅ 0–6-corrected; 3 flagged; decision pending |
| Excerpt/stimulus QC (`excerpt_cleaning.py`) | ✅ 0–6-corrected; 46 contested (diagnostic), 3 inversions |
| §6A Statistical analysis (participants only) | ✅ Done |
| §6B Lift analysis | ✅ Done |
| §8A Dominant-frequency demo (Note 1) | ✅ Done |
| §13A EDA (Note 7) | ✅ Done |
| §17A Expanded classifier metrics (Note 3) | ✅ Done |
| §E1 Condition-Delta (Path B) | ✅ Done (needs multiple-comparison correction, §9) |
| §E2 SHAP importance (Path A) | ✅ Done |
| §E3 MERT↔CREPE correspondence (Note 2) | ✅ Done |
| §E4 Feature selection (Note 4) | ✅ Done |
| Madmom / DTW / Essentia DSP / MusiCNN | ⏭ Skip stubs — re-enable after validation |

**Headline results (validation run, all 118 participants, no exclusions):**
- MERT F1-macro ≈ **0.74** (SVM-Linear) ≫ CREPE ≈ **0.32** (NaiveBayes); Combined ≈ MERT (0.73). Wilcoxon across classifiers: MERT>CREPE and Combined>CREPE significant; MERT vs Combined n.s.
- VA regression (MERT, Ridge): valence r ≈ 0.86, arousal r ≈ 0.86.
- Quadrant accuracy (Ridge VA → circumplex): 0.70.

---

## 5. Workflow Setup (how work is now done)

**Version control is GitHub-based.**

- **Repository:** `makris2003/violin-emotion-thesis` (private, GitHub)
- **Single source of truth:** GitHub. Kaggle is the execution environment (GPU + data); GitHub stores the code.

**Repository structure:**
```
violin-emotion-thesis/
├── CLAUDE.md              ← project context, auto-read by Claude Code
├── README.md
├── .gitignore            ← excludes audio, .npz, outputs/
├── notebook/
│   └── thesis_pipeline.ipynb
├── scripts/
│   ├── questionnaire_cleaning.py   ← participant screening
│   └── excerpt_cleaning.py         ← stimulus/excerpt QC
├── data/
│   ├── raw/              ← audio (gitignored)
│   └── forms/            ← questionnaire CSVs (gitignored)
├── outputs/             ← generated figures/tables (gitignored)
└── docs/
    └── handoff_notes.md
```

**Tools:**
- **Local machine:** Windows, VS Code at `C:\VScodeProjects\violin-emotion-thesis`, repo cloned via GitHub Desktop
- **Claude Code** (CLI + VS Code extension) for editing the notebook and scripts locally. Reads `CLAUDE.md` automatically at session start.
- **Kaggle ↔ GitHub linked** via Kaggle's built-in integration (File → Link to GitHub).

**Standard loop:**
1. Edit notebook/scripts locally with Claude Code
2. `git add . → git commit → git push` to GitHub
3. On Kaggle: import/pull latest from GitHub, run with GPU
4. Commit results back from Kaggle to GitHub

**Known gotcha:** Kaggle's "Pull from GitHub" overwrites the editor and its OAuth token can go stale ("Failed to fetch file content"). Fix = unlink/relink GitHub in Kaggle settings, or use File → Import Notebook → GitHub tab. Kaggle exposes no SSH; local VS Code is for editing only, GPU work runs on Kaggle.

---

## 6. Supervisor's Notes — Status

Seven notes from the meeting, sequenced by dependency. **All seven now have implementations**; the remaining work is validation and write-up (see §9).

**Phase 1 — Foundation**
- **Note 7 — EDA:** ✅ **DONE** (§13A). Label structure/prevalence, trivial baselines, VA distribution, quadrant occupancy, CREPE sanity vs physical priors, MERT PCA (22 comps ≥90% var) / t-SNE / silhouette (≈0 by condition → no unsupervised separation).
- **Note 1 — Dominant frequency in violin playing:** ✅ **DONE** (§8A). Demo showing the loudest partial ≠ fundamental (e.g. F0 220 Hz vs loudest partial 445 Hz = 2nd harmonic vs centroid ~1.9 kHz).

**Phase 2 — Human data analysis (classifier-independent, most defensible)**
- **Note 5 — Statistics from participants:** ✅ **DONE** (§6A).
- **Note 6 — Lift:** ✅ **DONE** (§6B).

**Phase 3 — Feature understanding**
- **Note 2 — Which MERT/CREPE characteristics correlate with which technique:** ✅ **DONE** (CREPE side via §E1/§E2; MERT side via §E3). §E3 linear probe result: MERT strongly encodes F0 Mean (R²=0.97) and F0 CV (0.80), weakly encodes vibrato depth (0.11), and does **not** linearly recover F0 Range or Vibrato Rate.
- **Note 3 — Check classifier metrics:** ✅ **DONE** (§17A). Per-label precision/recall/F1 + Dummy baselines (most_frequent/stratified/uniform). Documents F1-macro as the headline metric.

**Phase 4 — Feature decision**
- **Note 4 — Pick the features to use:** ✅ **DONE** (§E4). Leakage-free within-fold `MLSelectKBest` k-sweep (MERT peaks around k≈50), selection-stability across folds, and a block test showing the two MERT-blind CREPE features add nothing (−0.007 F1) — which explains Combined ≈ MERT.

---

## 7. Immediate Next Steps

1. **Review the 3 flagged participants** with supervisor; decide exclusions; re-run the pipeline cleaned (wire `excluded_participant_ids.csv` into the §4 parse).
2. **Resolve the §9 open issues** before drafting results chapters — especially the label-structure framing and the §E1 multiple-comparison correction.
3. **Fix the two quick modelling issues:** drop/tune SVM-RBF (currently collapses to majority prediction) and replace the degenerate MFCC baseline (F1-macro 0.0) with a linear-kernel baseline + the §17A Dummy floors + a condition-detection floor.
4. **Extend VA regression** to CREPE and Combined feature sets plus a mean-predictor baseline (currently MERT only).
5. **Verify the MERT `pos_conv_embed` weight load** (see §9) and pin `transformers`/`torch` to a known-good pair.
6. **Then re-enable the 4 skipped extractors** (Madmom, DTW, Essentia DSP, MusiCNN) for full-feature results.

---

## 8. Key Conventions / Variables (for whoever continues)

- Rating scale: **0–6** (7 points, midpoint 3.0). `VA_MID = 3.0`.
- Excerpt IDs: `PieceName_ConditionCode` (e.g. `Bach_Adagio_S1_MEC`)
- Participant IDs: `questionnaire_{n}_P{idx:03d}` (alias `S{session}_P{idx:03d}`)
- `df_all` / `df_agg` — per-excerpt aggregated dataframe (condition, piece, valence_mean, arousal_mean, emotion_labels)
- `df_crepe` — (60,8) CREPE features indexed by excerpt_id
- `emb_mert` — dict {excerpt_id: np.array(768)}; averaged over layers [5,6,7]
- `Y` — (60,8) binary label matrix; `mlb.classes_` = the 8 emotion names
- `X_A / X_B / X_C` — MERT / CREPE / Combined feature matrices
- `df_delta` — piece-level technique & VA deltas (20 pieces, §E1)
- `df_shap_importance` — (8 emotions × 8 techniques) mean |SHAP| (§E2)
- `df_probe` — MERT→technique linear-probe CV R² per feature (§E3)
- `df_ks` — feature-selection k-sweep results (§E4)
- `SafeMultiOutputClassifier` — the LOOCV-safe multi-label wrapper
- `MLSelectKBest` — leakage-safe within-fold multilabel univariate selector (§E4)
- `OUTPUT_DIR` — where all figures/CSVs are saved
- The 4 skipped extractors are skip-stub cells at §9 (madmom), §10 (DTW), §11 (Essentia DSP), §12 (MusiCNN)

---
