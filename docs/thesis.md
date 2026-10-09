# Thesis Handoff — Violin Expressiveness → Emotion Recognition

**Purpose:** complete standalone context to continue work in a fresh conversation. Covers (1) scope, (2) current pipeline architecture, (3) progress, (4) findings, (5) open issues, (6) technical report status, (7) conventions, (8) workflow.

> **Revision status (current): Phase B applied — results pending the next Kaggle run.** Phase B (descriptor repairs + analysis upgrades, §10) is in the notebook (still 101 cells). Phase A was applied and run on Kaggle (2026-10-09, GPU T4, 101 cells, 0 errors); §4 records those verified results, which are **pre-Phase-B** wherever they depend on CREPE / Essentia / timing features or on the default classifier (all of Parts 4–6, §2.3c). Part 1 and the §1.x statistics are unchanged by Phase B. Every Part 4/5/6 predictive number from runs before Phase A is pre-audit and superseded.
>
> **Earlier revision note:** This supersedes the older "validation-mode (MERT+CREPE only)" handoff. The pipeline has since gained **per-rater normalization** (new ground truth), an **active Essentia block** (dynamics/timbre/tonal), a **shared leakage-free forwarding gate** for every tool, a full **Part 5 attribution suite** and **Part 6 tool-characterization**, and the **MFCC baseline was removed**. The interpretable-feature narrative has moved **from vibrato to dynamics/timbre + pitch**. **DTW (timing/rubato) is implemented and wired end-to-end (§2.4/§2.4b)** and a **systematic ablation study** (§4.4b/c/d + §5.6, one shared engine, two CV protocols, FDR-corrected) was added; **both were executed in the Kaggle run of 2026-08-16**. Notebook: `notebook/thesis_pipeline.ipynb`, **95 cells**, Kaggle P100.

---

## 1. Project Overview

### 1.1 Original thesis brief (authoritative scope)

> Σκοπός: να μελετηθούν **τεχνικές που προσδίδουν εκφραστικότητα σε μουσική ερμηνεία** (βιμπράτο, χρονικές/τονικές παραλλαγές, διανθίσματα κλπ) και η **επίδρασή τους στο ακουστικό αποτέλεσμα** και κατ' επέκταση **στο κοινό**.
>
> Ζητείται **κατηγοριοποίηση των εργαλείων** εντοπισμού τεχνικών (signal processing, DNNs, αλγόριθμοι, κ.ά.) και **κατηγοριοποίηση των modalities** μέτρησης της επίδρασης στο κοινό (valence-arousal, annotations, κ.ά).

The brief defines the spine **technique → acoustic → audience** plus two categorisation deliverables (tools; modalities). The framing is a **categorisation + interpretable-attribution** study, **not** a leaderboard. MERT enters as *one tool category* (the deep/SSL exemplar), not as the protagonist.

### 1.2 Working questions (current framing)

1. Which expressive techniques does the performer escalate across MEC → EXP → EXG, and which does the audience register (as valence/arousal)?
2. Can perceived emotion be **attributed interpretably** to technique, and what is the **cost of explainability** relative to the black-box representation?
3. Where do the interpretable acoustic features and the SSL representation agree/complement each other?

*(The earlier "does MERT beat CREPE?" benchmarking framing is deliberately de-emphasised — see §5.)*

### 1.3 Dataset

- **60 excerpts = 20 pieces × 3 conditions** — `MEC` mechanical / `EXP` expressive / `EXG` exaggerated. **Recorded by the thesis author playing violin.**
- **118 participants across 5 linked questionnaire forms.** Ratings: **valence 0–6**, **arousal 0–6**, plus multi-select emotion tags.
- **Ring (linked) form design:** 20 excerpts/form; **8 anchors shared with each neighbouring form, 4 unique**, so the five forms connect cyclically through **40 anchor excerpts**. This linking underpins the normalization validation (§3/§1.2b).
- **Scale 0–6** (7 points, midpoint **3.0**). `0` is a genuine lowest rating (verified in raw forms), **not** NA. `VA_MID = 3.0` — correct, do not change.
- **Ground truth = 5 classes** (4 circumplex quadrants + Neutral), derived from **normalized** VA means (see §1.1b flow). `NEUTRAL_RADIUS = 0.75` (un-tuned — see open issues).

**Environment:** Kaggle, P100 GPU, PyTorch 2.3.1 (last sm_60 build).

---

## 2. Pipeline Architecture (current, 101 cells — Phase B applied)

Runs top-to-bottom. `§` = the in-notebook section banners.

| Part | § | Content |
|---|---|---|
| **0 Setup** | 0.1–0.4 | install / imports / **config** (`VA_MID=3.0`, `NEUTRAL_RADIUS=0.75`, `MERT_LAYERS=[5,6,7]`, `QUADRANT_EMOTIONS`, tag maps) / audio manifest |
| **1 Audience ground truth** | 1.1 | parse 5 CSVs → `df_long` (keeps `0`) |
| | **1.1b** | **per-rater normalization → new ground truth** (per-dim z, σ-floor, rescale preserving grand-mean + between-excerpt SD) |
| | 1.2 | aggregate → `df_agg`; derive 5-class labels from **normalized** VA; raw kept in parallel |
| | **1.2b** | **normalization validation** (anchor cross-form Δ, quadrant stability, corr, variance) |
| | 1.3 | label matrix `Y` (60×N), `mlb`, `df_expr`, `df_all` |
| | 1.4 | rating stats — Friedman / Kendall W / Wilcoxon |
| | 1.5 | lift analysis (emotion↔condition, raw tags) |
| **2 Feature extraction** | 2.1 / **2.1b** | **MERT** (768-dim, layers 5–7; **positional-conv weight-norm tensors restored from the checkpoint + asserted**, cache `emb_mert_v2.npz`) / **controlled MERT forwarding** (PCA-95, unsupervised) |
| | 2.2 / **2.2b** | **CREPE** — Phase B: tracks cached once (`crepe_tracks_v2.npz`), octave correction, note segmentation, then vibrato (B8) / portamento (B9) / F0 (B10) descriptors from the tracks; helpers in the `# >>> HELPERS:descriptors` block / **controlled CREPE forwarding** |
| | 2.3 / **2.3b** | **Essentia** (dynamics/timbre/tonal) — Phase B: flux on L1-normalised spectra (B11), silence-gated dynamic range (B12) / **controlled Essentia forwarding** |
| | **2.3c** | **recording-level sanity check** (report only) — peak/clipping, LUFS robust z, within-piece LUFS, **spectral-flatness noise floor** "recording-chain change" test (B-NF), optional session map |
| | 2.4 / **2.4b** | **timing + DTW (ACTIVE)** — reference-free `timing_*` family (CREPE note onsets + 151-frame tempogram with log-normal prior, B13) + within-piece `dtw_*` warp-path family / **controlled forwarding**. madmom / MusiCNN remain skip stubs |
| | 2.5 | assemble `X_A/B/C/D/F`, `X_MERT_FULL`, regression twins `X_C_REG/X_D_REG/X_F_REG` (B-MERTFULL); `STRICT_BLOCKS` guard; no pre-Phase-B cache name |
| **3 Validation & EDA** | 3.1–3.4 | class floors · VA/coverage · CREPE sanity · **3.3b DTW timing sanity + manipulation check** · MERT structure (PCA scree) |
| **4 Prediction** | **4.0** | **evaluation protocol & CV engine** — `get_cv` (piece = PRIMARY, loo = secondary), **nested model selection default** (`cv_predict_nested_select`, B-NMS; parallel == serial guard), `cv_predict_multilabel`, `nested_ridge_predict`, `oos_r2`/`rmse`, `bh_fdr`, Freedman–Lane + split-half helpers, re-seeded Dummy floors (mean/SD/P95), equivalence guard |
| | 4.1 | multi-label classification — NestedSelect (★default) + **8-classifier sensitivity table** (not used for selection; NaiveBayes = Phase A default), both protocols; balanced + inner-CV C for SVM/LogReg (A5) |
| | 4.2 | valence/arousal regression (fixed models + `Ridge-nested · MERT-full`), both protocols, r · R² · RMSE |
| | 4.4 | feature-set table + Wilcoxon (four hand-picked sets), per protocol |
| | **4.4b** | **ablation engine + block lattice** — all 15 subsets of {MERT, CREPE, Essentia, DTW}; unique vs marginal contribution per tool |
| | **4.4c** | **configuration ablation** — `MERT_PCA_VAR` · Essentia curation · DTW family · shared-gate `corr_max` · **`4D-balancing`** (nested selection = reference; NB priors / SVM balancing, descriptive). `4A-crepe-portamento` removed in Phase B |
| | **4.4d** | **target-side ablation** — `NEUTRAL_RADIUS` 0.5/0.75/1.0 (macro-F1 with *and* without Neutral) · normalized vs raw ground truth |
| | 4.5 / **4.5b** | per-class P/R/F1 (default on X_C) / **DummyClassifier floors** (the official floor; mean, SD, P95 over 200 re-seeded repeats) |
| | 4.6 | per-condition F1 (MEC/EXP/EXG), default on X_C |
| | 4.7 | quadrant accuracy from predicted VA (+ VA scatter, circumplex), both protocols |
| | 4.8 | prediction–ground-truth agreement (Krippendorff α), default on X_C |
| | **4.9** | **within-piece prediction** — piece-centred targets & features, piece CV only, Condition-only reference, within-piece permutation p; **B-INC** Freedman–Lane incremental test (Condition + Features vs Condition-only); **B-NC** split-half noise ceiling |
| **5 Attribution** | 5.1 | **condition-delta** — technique change vs rating change (within-piece); **BH-FDR over all 174 tests** |
| | **5.1b** | **dose-response** — Page's trend test over ordered MEC<EXP<EXG |
| | 5.2 | **SHAP** technique importance per class |
| | 5.3 | **embedding-distance** attribution |
| | 5.4 | **interpretable linear read-out** (signed technique→emotion) |
| | 5.5 | **cost of explainability** (interpretable vs representation, same CV, both protocols; headline max(MERT-PCA, MERT-full) + the two separate gaps; handles a negative CoE) |
| | **5.6** | **sub-family ablation** — decomposes the CoE across Pitch/Vibrato/Portamento/Dynamics/Timbre/Tonal/Timing; writes the consolidated `ablation_master.csv` |
| **6 Tool characterization** | 6.1 / 6.1b | MERT-dim ↔ feature correlation / linear probe (does MERT encode each feature?) — **GroupKFold(5) over pieces, α nested, grid to 1e4** |
| | 6.2 / 6.2b | **tool complementarity** — does each tool add signal on top of MERT? (default classifier, both protocols) / guard vs the lattice |
| **7 Synthesis** | 7.1 | final summary + save all outputs |

### 2.1 The shared forwarding gate (key mechanism)

Every tool is **extracted in full** (nothing lost on disk), then **forwarded** through one shared, **unsupervised** selector `unsup_forward_select(candidates, corr_max=0.90, name, eps, order)`: near-constant prune + correlation prune, in an a-priori preference order. It reads **only the feature matrix, never `Y`** — this is what makes a once-up-front selection safe to feed LOOCV (a supervised selection would leak held-out folds). `StandardScaler` stays **inside** every CV pipeline. MERT (§2.1b), CREPE (§2.2b), Essentia (§2.3b) and DTW (§2.4b) all use this same gate → `MERT_FORWARD`, `CREPE_FORWARD`, `ESSENTIA_FORWARD`, `DTW_FORWARD`.

### 2.2 What enters the models vs what is kept aside

`X_A=MERT-fwd`, `X_B=CREPE-fwd`, `X_C=MERT+CREPE+Essentia+DTW (all fwd)`, `X_D=MERT+Essentia`, `X_F=MERT+DTW` (the §6.2 complementarity contrast). **`X_MERT_FULL`** (768-dim) serves the *representational* analyses (§3.4 structure, §5.3 distances, §6.1/§6.1b correspondence), which must see all dimensions, and — since Phase B (B-MERTFULL, `MERT_REG_REPR='full'`) — the **regression heads**: `MERT_REG` and `X_C_REG/X_D_REG/X_F_REG` are the same matrices with the PCA-forwarded MERT block replaced by `X_MERT_FULL`. The classification heads keep `X_A`–`X_F`. Diagnostic-only columns (`CREPE_DIAGNOSTIC`, `ESSENTIA_DIAGNOSTIC`, `TIMING_DIAGNOSTIC`) never enter forwarding, `df_interp` or the gate sweeps. `STRICT_BLOCKS=('MERT','Essentia')` → a missing block raises rather than silently zero-filling (per the "hard imports, no fallbacks" policy).

### 2.3 Evaluation protocol (§4.0, Phase A) — one engine for every cross-validated number

- **Protocols:** `PRIMARY_PROTOCOL='piece'` (leave-one-piece-out on `PIECE_GROUPS`) and `loo` (optimistic secondary), always both, piece first, with the optimism gap; `get_cv(protocol)` is the only splitter builder (`'piece5'` = GroupKFold(5) on pieces, for the §6.1b probe only).
- **Default classifier (Phase B, B-NMS): nested model selection** (`DEFAULT_CLF_NAME='NestedSelect'`, `cv_predict_default` = `cv_predict_nested_select`). Inside every outer training fold an inner GroupKFold(5) over the training pieces scores the 13 `SELECT_CONFIGS` (NB · SVM-Linear balanced · LogReg balanced, C ∈ logspace(−3, 2, 6)) by pooled inner out-of-fold macro-F1; ties → NB, then the smallest C; the winner is refitted on the outer training fold. Choosing the classifier on the CV that reports it would be selection bias; nesting makes the reported score an unbiased estimate of the select-then-fit procedure. A content-hash memo returns byte-identical calls; outer folds run in parallel (joblib, `threadpool_limits(1)` per fold) and §4.0 asserts parallel == serial exactly. Every call prints how often each config was chosen. **GaussianNB** (`NB_CLF_NAME`, `make_nb_clf`) — the Phase A pre-declared default — failed under `piece` and stays as a reported row. §4.1's 8 classifiers are a sensitivity table only; per-class / per-condition / α analyses use the default on `X_C`.
- **Engine:** `cv_predict_multilabel` (manual binary relevance, `SafeMultiOutputClassifier` semantics, optional per-label inner-CV tuning) — asserted identical to the old pipeline for GaussianNB; `nested_ridge_predict` (α by piece-grouped inner GroupKFold(5), replaces every `RidgeCV`); `NestedRidgeOperator` (closed-form twin, for permutation refits); `oos_r2` (baseline = training-fold mean), `rmse`, `bh_fdr`, `within_piece_permutation`.
- **Chance floors:** Dummy seeded per fold × label, 200 repeats → mean / SD / P95; "beats chance" = above the max P95 across strategies.

---

## 3. Progress since the previous handoff

- **Phase B of the Oct-2026 audit applied (2026-10-10) — results pending the next Kaggle run.** Nested model selection as the default classifier (B-NMS) · Freedman–Lane incremental test (B-INC) and split-half noise ceiling (B-NC) in §4.9 · MERT-full regression heads (B-MERTFULL) · CREPE tracks cached + octave correction + note segmentation (B-shared) · repaired vibrato (B8), portamento (B9), F0 range (B10) · normalised flux (B11) · silence-gated dynamic range (B12) · `timing_*` family + 151-frame tempogram with log-normal prior (B13) · flatness noise floor + optional session map (B-NF) · `_v2` caches (B-cache). See §10.

- **Phase A of the Oct-2026 audit applied and run on Kaggle (2026-10-09; results in §4).** §4.0 CV engine · re-seeded Dummy floors · leave-one-piece-out primary everywhere · nested ridge + R²/RMSE · §4.9 within-piece prediction · MERT pos-conv weight restore (cache v2) · GaussianNB default + best-of selection removed · class balancing + inner-CV C (+ tier `4D-balancing`) · group-aware MERT probe · BH-FDR in §5.1 · §2.3c recording-level check. See §10.

- **Per-rater normalization added (§1.1b/§1.2b) and made the new ground truth.** Per-dimension z with σ-floor 0.5, rescaled to preserve the raw grand mean and between-excerpt SD (so VA_MID and quadrant geometry are untouched — only rater scale-use bias is removed). **Validated via the ring:** the 40 anchors are each rated by two independent pools, so the cross-form discrepancy before/after is the honest test. **Functional with or without participant exclusions** (all constants derived at runtime). Raw retained in parallel.
- **Essentia re-enabled (§2.3/§2.3b)** — dynamics (integrated loudness, dynamic/loudness range), timbre (spectral flux/centroid/rolloff, dissonance, log-attack time), tonal (key strength, HPCP entropy). This closed the dynamics/timbre gap and populates the signal-processing tool category.
- **Shared leakage-free forwarding** unified across all tools (§2.1).
- **Part 5 attribution suite** built out: condition-delta, dose-response (Page), SHAP, embedding-distance, interpretable read-out, cost of explainability.
- **Part 6 tool characterization** (correspondence probe + complementarity).
- **MFCC baseline removed** — it was degenerate (F1≈0, indistinct from the Dummy floor) and carried out-of-topic leaderboard framing. The **DummyClassifier floors (§4.5b)** are the official chance baseline.
- **Cleanup pass (partial):** MFCC gone; some duplicate/out-of-topic figures still pending removal (see open issues).
- **Technical report drafted** — Abstract + Introduction + Methods (IEEEtran), following the Chowdhury/Widmer interpretable-MER backbone (see §6).
- **DTW (timing/rubato) implemented and integrated (§2.4/§2.4b).** Two families: **(1) reference-free** — local-tempo variability, IOI CV, onset rate, pulse clarity; computed per excerpt with no reference, so all 60 (MEC included) are graded and the family is condition-INdependent. **(2) within-piece DTW deviation** — chroma sequences of EXP/EXG aligned to the same piece's MEC with `librosa.sequence.dtw`; warp-path deviation, local-slope variability, tempo/path ratios. MEC is the alignment reference, so its family-(2) values are the **self-alignment identity (0 deviation / 1 ratio) by construction** — meaningful for attribution but it means family (2) partly encodes the condition. §2.4b splits them: family (1) alone feeds the predictive matrices (`DTW_MODEL_COLS`, `DTW_DEV_IN_MODELS=False`), both feed the attribution frame (`df_dtw_attr`, `DTW_DEV_IN_INTERP=True`). DTW is wired into §2.5 (X_C, new X_F), §3.3b, §5.1, §5.1b, §5.2, §5.4, §5.5, §6.1, §6.1b, §6.2. **Executed in the 2026-08-16 run** (13-col block; 11 forwarded = 8 ref-free + 3 dev; the models receive the 8 ref-free).
- **Systematic ablation study added (§4.4b/c/d + §5.6 + the §6.2b guard).** One engine `ablate()` scores every tier under one protocol (SVM-Linear → F1-macro, RidgeCV → V/A, scaler inside the CV pipeline — the same protocol as §5.5/§6.2, so all numbers are comparable by construction), caches per-excerpt predictions, and consolidates into `ablation_master.csv`. It closes the asymmetry that §4.4 compared only four hand-picked sets while §6.2 could only ever *add* to MERT: the **full 15-subset lattice** now yields each tool's **unique** (full − full∖tool) and **marginal** (alone − chance floor) contribution, MERT's included. Every row is scored under **both** LOOCV and leave-one-piece-out, with bootstrap CIs, paired permutation p-values and BH-FDR. §5.6 decomposes the single cost-of-explainability number by expressive family, giving Part 5 a third and *predictive* line of evidence alongside dose-response and SHAP. Three asserts keep it honest (the lattice must rebuild `X_C/X_D/X_F` exactly, the §4.4d label re-derivation must reproduce `Y`, and §6.2's additive numbers must equal the lattice rows). **Executed in the 2026-08-16 run** (112 scored rows, 9 tiers; all three asserts passed) — those numbers are pre-audit (SVM-Linear + RidgeCV-GCV protocol) and superseded by Phase A.

---

## 4. Current findings

> **Phase A results (Kaggle run of 2026-10-09, GPU T4) — pre-Phase-B.** All verified from that run's outputs. Phase B changes every CREPE / Essentia / timing feature and the default classifier, so every number below that depends on them (classification, regression on feature sets, CoE, §4.9, §5.x, §6.x, §2.3c floors, the dose-response headline) is **pre-Phase-B** and will be superseded by the next run; the normalization / variance-structure / cleaning facts are unchanged. Pre-audit LOOCV numbers (e.g. CoE +0.112 / +0.051 / +0.145) are superseded.

**Guards.** All passed: §4.0 engine equivalence, §4.4b lattice rebuild, §4.4d label reproduction, §6.2b, §4.9 closed-form permutation equivalence, §5.5↔§5.4 cross-check. 101 cells, 0 errors.

**Dummy floors fixed.** Uniform mean 0.272 (loo) / 0.2715 (piece); stratified 0.183 / 0.161; strict P95 0.320 (loo) / 0.327 (piece).

**MERT (A3) — false alarm.** The restore ran, but the embeddings are identical to the pre-fix run (the §6.1 correspondence table and every MERT-based F1 match to 4 decimals). The transformers "newly initialized" warning was spurious; pre-audit MERT numbers were valid. The assert stays.

**Classification, piece (PRIMARY).** The pre-declared default GaussianNB is **below chance on every set** (F1-macro A 0.112, B 0.287, C 0.168, D 0.113); on X_C three classes collapse (HA_HV, LA_HV, Neutral recall 0); quadrant accuracy 0.25; Krippendorff α ≤ 0 for all labels except LA_LV (0.41). LOOCV for comparison: C 0.686. **Tier 4D (piece):** SVM-Linear balanced + inner-CV C beats NB deployed on X_A (+0.143, q=0.023), X_C (+0.156, q=0.005), Interp (+0.151, q=0.023); Interp + SVM balanced = 0.385 [0.327, 0.434] — the only configuration above the strict chance P95; NB balanced priors: no effect. → Decision (Phase B): nested model selection replaces the single default classifier; NB stays reported as "pre-declared Phase A default, failed under piece".

**Regression, piece (nested ridge).** Arousal R² — X_C 0.50, CREPE+Essentia 0.52, Essentia 0.40; valence R² ≈ 0 (max 0.23, CREPE+DTW). MERT-PCA: R²v −0.17, R²a −0.25; MERT-full: R²a 0.45 (Δ vs PCA +0.70, q=0.0015) → PCA-95 discards the arousal-relevant variance and keeps piece identity (a tool-taxonomy finding).

**Cost of explainability, piece (negative = interpretable wins).** Headline ΔR valence −0.215, ΔR arousal −0.054, ΔF1 −0.127; vs MERT-PCA ΔR² arousal −0.756.

**§4.9 within-piece (piece-centred, LOPO).** Δarousal — Condition-only R² 0.41, Interpretable 0.47 (p=0.002), MERT-full 0.38, MERT-PCA 0.27. Δvalence — Condition-only 0.047 (p=0.034), Interpretable 0.20 (p=0.004), MERT-full 0.175, MERT-PCA 0.134. Within-piece SD: V 0.234, A 0.285 (scale 0–6). The "beats condition" difference was not yet tested (→ Phase B B-INC).

**§5.1 BH.** 0 of 174 tests have q < .05 (12 raw p < .05 vs 8.7 expected) → the condition-delta correlations are **not evidence; do not cite them as findings**.

**§6.1b grouped probe.** Well represented across pieces — F0 mean 0.80, spectral centroid 0.73, HPCP entropy 0.73, pulse clarity 0.59; loudness only 0.08; most reference-free timing features ≤ 0. α hit the grid edge (10000) in 32 folds (= shrink to the mean; harmless for those targets).

**§2.3c recording level.** 0 LUFS outliers, 0 clipping, 21 noise-floor flags (15 excerpts inconclusive). EXG conclusive floor mean −66.2 dBFS vs MEC −71.4 / EXP −72.5; EXG LUFS +4.2 LU vs MEC. Facts from the author: 3 recording sessions with position markers; in EXG the performer deliberately played louder. Equality of gain across sessions is **not independently verified** (limitation); the 5th-percentile floor estimator is unreliable when an excerpt has no silence (→ Phase B B-NF). A session map is not available yet.

**Variance structure (audit).** 95% of valence and 91% of arousal variance across the 60 excerpts is **between-piece**; the condition explains **1% (V) / 4% (A)**. LOOCV (which keeps a piece's other two renditions in training) therefore mostly measures piece recognition; leave-one-piece-out and the within-piece analysis (§4.9) are the honest tests.

**Headline 2 — escalation via dynamics & timbre, not vibrato (contrarian) — pre-Phase-B, obtained with the OLD (broken) vibrato descriptor and the raw (loudness-confounded) flux; both were replaced in Phase B (B8, B11), so this headline must be re-read from the next run.** Dose-response (Page's trend, 20/20 complete pieces): **spectral flux z=+5.38\*\*\***, **loudness z=+4.74\*\*\*** rise; **rolloff z=−2.85\*\*** falls; **vibrato rate z=+1.26 n.s.** Audience side: **arousal z=+4.11\*\*\*** vs **valence z=+1.26 n.s.** ⚠️ Note the honest nuance: the mean EXG−MEC for vibrato is *positive* (rate +0.234, depth +1.903) — vibrato does rise on average, but **non-monotonically/inconsistently** (only 10–30% of pieces monotone), which is why it is n.s. The correct claim is "vibrato change is weaker and less consistent **with the current mean descriptor**," not "vibrato does nothing." A richer vibrato descriptor (coverage/extent, note-level) may recover it — open question.

**Normalization validation.** Raw grand mean V=2.97 / A=3.40 (arousal legitimately skews high); between-excerpt SD V=1.065 / A=0.956 (preserved); anchor cross-form |Δ| baseline **V=0.334 / A=0.289**; raw↔norm Pearson r = **0.996 (V) / 0.993 (A)**; **10/60 excerpts change class, 4/60 change quadrant** (the earlier "0/60 flips" claim was wrong). Per-rater min SD 0.51/0.54 → the σ-floor **never fires** (insurance, not active). **Conclusion (unchanged): normalization is a defensibility/robustness step — it confirms the signal is real, not a rater artifact; no material change in results.**

**Stimulus level.** The stimuli played to participants were **not loudness-normalized** (confirmed by the author): level differences between excerpts are part of the stimulus. §2.3c (Phase A) only checks that no recording was made at a clearly different gain/microphone distance.

**Cleaning.** 3 participants flagged (≥2 methods) — **not yet removed**; decision pending. Excerpt QC: valence ICC(1,k)≈0.96, arousal ≈0.95; a few arousal inversions among the 20 pieces (manipulation-check candidates).

---

## 5. Open issues

1. **"8 emotions" is mechanically 5 classes.** `QUADRANT_EMOTIONS` maps each quadrant to a *set*, so pairs (Tenderness≡Peacefulness, Power≡Joyful, Sadness≡Nostalgia) are byte-identical columns. **Decision taken: report honestly as 5 classes** (4 quadrants + Neutral). Alternative (rebuild `Y` from raw multi-label tags) remains available. Good supervisor topic.
2. ~~**Neutral radius un-tuned.**~~ **Now measured in §4.4d** — the 0.5/0.75/1.0 sweep runs on both X_A and X_C, reports macro-F1 **with and without** Neutral, and prints class support plus how many excerpts each radius relabels. Remaining decision: read the table once it has run and keep 0.75 unless another radius wins on *both* columns.
3. **Figure dedup still pending** (partial cleanup only): a duplicate VA scatter still sits alongside the circumplex; check for any remaining per-class-F1 heatmap/radar and ablation-figure redundancy and prune (keep circumplex, P/R/F1 bars, PCA scree).
4. **DTW block size / DTW-dev in `df_interp`.** In the 2026-08-16 run the §2.4b gate forwarded 11 columns (8 ref-free + 3 dev) and the "outside 3–10" warning fired — trim `DTW_CANDIDATES` if that matters. Open: whether letting the DTW-deviation family into `df_interp` moves the §5.5 cost-of-explainability numbers, since those columns are 0/1 at MEC by construction (`DTW_DEV_IN_INTERP=False` gives a strictly condition-independent CoE).
5. **Participant exclusions not applied** — pipeline runs on all 118; wire the 3 flagged in once decided (normalization is already robust to this).
6. **Statistics hygiene:** the ablation study (§4.4b) carries bootstrap CIs, paired permutation tests and **BH-FDR within each tier × protocol** (now over F1, r and R²); **§5.1 now applies BH-FDR over all its tests** (Phase A); §5.1b (dose-response) still wants the same treatment or a pre-registered hypothesis set — not yet approved. Keep phrasing classifier-set comparisons as "across the classifiers tried," not population inference. **The lattice must stay descriptive** — selecting the winning subset on the same CV and then reporting its score is exactly the nested-selection error.
7. **Feature hygiene — repaired in Phase B (pending the run).** The Phase A descriptors were broken: `crepe_portamento_count` was a raw count; `crepe_vibrato_rate_hz` was bandpass-bounded to [4,8] Hz *and argmax-ed inside that same band* (it could never report "no vibrato"); vibrato/portamento ran on concatenated voiced frames. **Pre-Phase-B vibrato nulls** (the n.s. dose-response) are therefore evidence about the descriptor only. Phase B (B8/B9) analyses time-contiguous notes, searches [3,10] Hz with a presence test, splits extent from coverage and reports portamento as rates/shares — a vibrato null from the next run is a **musical result at this N**, with the remaining caveats (CREPE 10 ms resolution, only notes ≥ 800 ms, declared median substitutions counted in §2.2). B9 detectability limit: a glide is only segmented if its trend moves > 70 cents within 50 ms (≈ 1.4 cents/ms).
8. ~~**LOOCV leaks piece identity.**~~ **Addressed in Phase A:** leave-one-piece-out is now the PRIMARY protocol everywhere (§4.0), LOOCV is the optimistic secondary, and §4.9 removes the piece entirely (within-piece prediction). Quote the `piece` number for any claim about unseen material.
9. **`LA_HV` has n=5 from only 2 pieces** (La_Vita ×3, Meditation_Thais MEC+EXP): under `piece` it is learned from one piece at a time, and inner-CV tuning is skipped for it in many folds.
10. **Recording sessions.** The session map is **not provided yet** (`SESSION_MAP_CSV`, columns `excerpt_id, session`; §2.3c prints "session map not provided" and skips until it is uploaded). Gain equality across the 3 recording sessions is **not independently verified** (the performer reports playing louder in EXG on purpose).
11. **Phase C (deferred idea) — augmentation.** The supervisor's suggestion: augment the minority positive-valence classes (pitch shift ±1 semitone, tempo ±5%, never combined), with augmented copies used **only in training folds, and only when their source piece is itself in that training fold** (test = originals only, gate/PCA fitted on originals, scoring/permutations on the 60 originals). Caveats: it cannot add piece diversity (LA_HV comes from only 2 pieces); it perturbs studied variables (F0 mean, tempo, onset rate) while keeping labels; the copies' labels were never rated; phase-vocoder artefacts affect Essentia timbre features. **Decision:** evaluate class balancing (A5, tier `4D-balancing`) first; augmentation later as an ablation tier.
12. **Not approved yet:** SHAP fitted in-sample (§5.2) · §5.3 embedding-distance sign · §1.4 mixed model · BH in §5.1b.
13. **Known, not fixed (out of scope):** Fig 5 VA scatter (§4.7) never draws the MEC points (`cc` holds only EXP/EXG); README points to `docs/thesis_handoff.md`, the file is `docs/thesis.md`. §5.5's headline CoE is a best-of over two black boxes (max(MERT-PCA, MERT-full)) — conservative; the two separate gaps are in `coe_gaps.csv`.
14. **Parameter ablations still pending:** CREPE step_ms (re-extraction), the CREPE voicing threshold and the B8/B9 descriptor constants (now cheap from `crepe_tracks_v2.npz`, not approved yet), Essentia frame sizes, the tempogram window / tempo prior, per-MERT-layer.
15. **Runtime.** Nested selection (full 13-config grid, ~115 distinct calls after the memo) needs ≈ 3 h of serial CPU work by local measurement (≈ 40 s per serial `piece` call, ≈ 3× that for `loo`); with parallel outer folds that is an estimated **≈ 45 min extra on 4 cores, ≈ 1.5 h on 2 cores** — check §4.0's printed `os.cpu_count()` and the total run time.

---

## 6. Technical report status

First draft written for **Abstract, Introduction, Methods** (IEEEtran, English). Structural decisions (from a survey of the closest interpretable-MER / expressive-performance literature):

- **Backbone = Chowdhury/Widmer interpretable-MER skeleton:** interpretable layer → black-box baseline → an explicit **"cost of explainability"** results section.
- **The two categorisations are their own sections** (tools; modalities), *before* Methods; Methods **instantiates** their members with cross-references (no re-description).
- **Aims woven into prose, not numbered research questions**; the two taxonomies appear as content, never flagged as questions or "axes."
- **Abstract carries no counts** (only the three levels MEC/EXP/EXG) + the two headline findings.
- **MERT positioned as a frozen/probed representation** (deep-family exemplar), explicitly not a benchmark competitor.
- **Predictive validation (LOOCV) framed as supportive → appendix**, not a leaderboard.

Methods subsections drafted: study design & graded manipulation · stimuli/corpus · participants & questionnaire (ring) · emotion measurement & ground-truth · **response normalization** · feature extraction (CREPE / Essentia / MERT as parallel blocks; **the DTW timing block now needs writing up as a fourth parallel block, not as "planned"** — including the honest note that its within-piece family is referenced to the MEC rendition) · analysis pipeline (deltas, dose-response, read-out, cost of explainability). Unknowns (how the three conditions were produced in detail, piece list, audio specs, demographics, exact tag set, author block) are marked as visible placeholders in the `.tex`.

---

## 7. Key conventions / variables (current)

- **Scale 0–6**, `VA_MID=3.0`, `NEUTRAL_RADIUS=0.75`, `MERT_LAYERS=[5,6,7]`.
- `df_long` — long-format per-rating (parse, §1.1); gains `valence_norm/arousal_norm` in §1.1b.
- `df_agg` / `df_all` — per-excerpt aggregated. **`valence_mean/arousal_mean` are now the NORMALIZED means** (ground truth); `valence_mean_raw/arousal_mean_raw` kept in parallel.
- `emotion_labels` — VA-quadrant-derived (from normalized means); **this becomes `Y`**. `top_tags` — raw tags, descriptive only (§1.5).
- `Y` (60×N), `mlb.classes_` = class names; `df_expr` — modelled-excerpt frame.
- **Forwarding:** `unsup_forward_select(...)`; `MERT_FORWARD` / `CREPE_FORWARD` / `ESSENTIA_FORWARD`; forwarded frames `emb_mert_fwd`, `df_crepe_fwd`, `df_ess_fwd`. Full blocks on disk: `df_mert`, `df_crepe`, `df_essentia`.
- **Assembly:** `FEAT_A/B/E/DTW/C/D/F`, `X_A/X_B/X_C/X_D/X_F`, `X_MERT_FULL`; `STRICT_BLOCKS=('MERT','Essentia')` (DTW non-strict but a missing row warns loudly); `bfv(...)` builder.
- **Interpretable frame & labels** used by Part 5: `df_interp` + `INTERP_FEAT_LABELS` (CREPE + forwarded Essentia + forwarded DTW, built in §5.1; §5.1b / §5.2 / §5.4 / §5.5 all derive their technique set from `INTERP_FEAT_LABELS ∩ df_interp.columns`).
- Part 5 outputs: `df_dose` (§5.1b) · SHAP importances (§5.2) · read-out coefficients (§5.4) · CoE table (§5.5).
- Excerpt IDs `PieceName_ConditionCode`; participant IDs `questionnaire_{n}_P{idx:03d}` (alias `S{n}_P{idx:03d}`).
- `OUTPUT_DIR` — all figures/CSVs. Skip stubs at §2.4: madmom, MusiCNN (DTW is active).
- **Timing/DTW constants:** `DTW_SR=22050`, `DTW_HOP=512`, **`DTW_TG_WIN=151`** (≈ 3.5 s), log-normal tempo prior centred on 100 BPM within 40–200 BPM, interior tempogram frames only (≥ 20), chroma-CQT alignment; `DTW_REFFREE` (= the `timing_*` family) / `DTW_DEV` (the `dtw_*` family) name the two families; `TIMING_DIAGNOSTIC=['timing_octave_jump_frac']`.
- **Phase B constants (§0.3 / §2.2 / §2.3 / §2.3c / §4.0):** `NC_N_SPLITS=200`, `SELECT_C_GRID=logspace(-3,2,6)` → 13 `SELECT_CONFIGS`, `MERT_REG_REPR='full'`, `SESSION_MAP_CSV`, `OLD_CACHE_NAMES`; CREPE descriptors: `OCT_WIN=10`, `OCT_DEV_RANGE=(1000,1400)`, `TREND_WIN=15`, `BOUNDARY_LAG=5`, `BOUNDARY_CENTS=70`, `NOTE_MIN_MS_TIMING=60`, `VIB_MIN_NOTE_MS=800` (sensitivity 600/800/1000), `VIB_BAND=(3,10)`, `VIB_NFFT=4096`, `VIB_PEAK_HALFWIDTH=0.5`, `VIB_PRESENCE_MIN=0.30` (empirical noise of the ratio ≈ 0.15–0.17, P95 ≈ 0.27 at 0.8 s), `VIB_MIN_EXTENT=5`, `GLIDE_DUR_MS=(40,400)`, `GLIDE_INTERVAL=(80,2000)`, `GLIDE_SIGN_MIN=0.8`, `GLIDE_RISE_SCALE=1.25`; Essentia `DR_ABS_GATE_DB=-60`, `DR_REL_GATE_DB=40`; §2.3c `FLAT_MIN=0.2` (0.1/0.2/0.3), `FLAT_BAND_HZ=(50,8000)`, `FLAT_RMS_PCTL=20`, `FLAT_MIN_FRAMES=20`, `SESSION_CONFINED_FRAC=0.80`; §4.0 `NMS_N_JOBS=os.cpu_count()`, `NMS_TIE_TOL=1e-12`.
- **Caches (Phase B, `_v2`):** `emb_mert_v2.npz`, `crepe_tracks_v2.npz`, `feat_crepe_v2.csv`, `essentia_features_v2.csv/.npz`, `rms_frames_v2.npz`, `dtw_features_v2.csv`; §2.5 asserts no `CACHE_*` path carries a name in `OLD_CACHE_NAMES`.
- **Phase A constants:** `SEED=42`, `N_DUMMY_REPEATS=200`, `WP_N_PERM=500` (§0.3); `PRIMARY_PROTOCOL='piece'`, `PROTOCOLS=('piece','loo')`, `RIDGE_ALPHAS=logspace(-2,4,13)`, inner `GroupKFold(5)`, `DEFAULT_CLF_NAME='NestedSelect'` since Phase B (`NB_CLF_NAME='NaiveBayes'` = the Phase A default) (§4.0); `TUNE_GRID_C={'C': logspace(-3,2,6)}` (§4.1); probe `ALPHAS=[1,10,100,1000,10000]` (§6.1b); MERT cache `emb_mert_v2.npz`; `OUTPUT_DIR='/kaggle/working/pipeline_outputs'`.
- **Result containers:** `all_recs` (with `protocol`; NestedSelect rows carry `selected` / `chosen`), `DEFAULT_RECS[(set, protocol)]` (NestedSelect), `NB_RECS[(set, protocol)]`, `REG_RES[protocol]` (`reg_res` = primary; incl. `REG_MERTFULL`), `df_wp_ceiling` / `WP_CEILING`, ablation rows gain `selected` / `n_configs_selected` / `n_features_reg`, `PRED_QUADS`, `INTERP_CV_R[protocol][dim]`, `INTERP_CV_F1[protocol]`, `COE[protocol]`, `BLOCK_F1[protocol]`, `df_within_piece`, `df_delta_tests`, `df_reclevel`, `df_inner_skips`; ablation rows gain `r2_val/r2_aro/rmse_val/rmse_aro/beats_chance/classifier`.

---

## 8. Workflow

**Repo:** `makris2003/violin-emotion-thesis` (private). GitHub = source of truth; Kaggle = GPU execution. `CLAUDE.md` at repo root is auto-read by Claude Code. Loop: edit locally (VS Code) → commit/push → Kaggle pulls → run on GPU → commit results back. Claude Code **cannot run** the notebook (no audio/GPU) → surgical edits + static checks + local synthetic tests only (the §4.0 `# >>> HELPERS:cv_engine` block is self-contained for that purpose). The notebook is stored as single-line JSON, so review changes cell by cell (e.g. nbdime) rather than with a plain `git diff`. Kaggle "Pull from GitHub" can stale its OAuth token (unlink/relink to fix).

---

## 9. Quick answers

- **"Why is valence weak?"** Intrinsic to MER (needs harmony/context). Arousal is clean with the same features → it's a finding. To *explain* (not rescue) it, add 1–2 harmonic features (mode / tonal tension) so you can say "valence is governed by harmony, which the graded expressivity holds fixed."
- **"Did normalization change results?"** Not materially — raw↔norm r = 0.996 (V) / 0.993 (A); 10/60 excerpts change class (mostly across the Neutral radius), 4/60 change quadrant. That's the point: it confirms the findings aren't a rater artifact (defensibility, not correction).
- **"Why remove MFCC?"** Degenerate (F1≈0), indistinct from the Dummy floor, out-of-topic. Dummy floors are the official baseline.
- **"Is it a leaderboard (MERT vs CREPE)?"** No — that framing is de-emphasised. It's a categorisation + interpretable-attribution study; cost of explainability is the comparison, not "who wins."
- **"Why forward features up front unsupervised?"** To keep the CV leakage-free — a supervised selection would leak held-out folds. The gate never sees `Y`.
- **"Why leave-one-piece-out as the primary protocol?"** 95% (V) / 91% (A) of the excerpt variance is between pieces; LOOCV keeps a held-out excerpt's two sibling renditions in training, so it mostly measures piece recognition.
- **"Why nested model selection as the default (and not GaussianNB any more)?"** Phase A pre-declared GaussianNB so it could not be a post-hoc pick; under leave-one-piece-out it fell below chance on every feature set. Adopting the best fixed classifier from the 4D table afterwards would be a post-hoc pick on the same CV, so Phase B moves the choice *inside* every outer training fold (inner GroupKFold over the training pieces): the reported score is then an unbiased estimate of the whole select-then-fit procedure. NB stays visible as "pre-declared Phase A default — failed under piece".

---

## 10. Audit (Oct 2026)

A full audit of the 2026-08-16 run's outputs produced three phases of corrections.

- **Phase A — pipeline corrections (APPLIED and run on 2026-10-09; results in §4; the A3 MERT issue turned out to be a false alarm).** A-shared §4.0 CV engine · **A1** DummyClassifier floors were re-created with one seed in every fold, so `stratified`/`uniform` predicted the same constant everywhere (F1-macro = 0); now seeded per fold × label, 200 repeats, mean/SD/P95 · **A2** leave-one-piece-out PRIMARY everywhere, nested piece-grouped ridge instead of RidgeCV, §5.5 handles a negative CoE, new §4.9 within-piece prediction · **A3** MERT's positional-convolution weight-norm tensors were never loaded (checkpoint `weight_g/_v` vs model `parametrizations.weight.original0/1`) — the conv ran with random weights; now restored + asserted, cache `emb_mert_v2.npz`; **every MERT-based number from before is superseded** · **A4** GaussianNB as the single pre-declared default; every best-of selection removed (§4.5 headline, §4.6 best overall, §4.8, §7.1) · **A5** class balancing + per-label inner-CV C for SVM/LogReg, balanced RF/ET, XGBoost `scale_pos_weight`; tier `4D-balancing` · **A6** out-of-sample R² and RMSE next to every r, R² permutation p in the ablation BH family · **A7** MERT probe with GroupKFold(5) over pieces and nested α (grid to 1e4) · **A14** BH-FDR over all 174 §5.1 tests, stars from q · **A-QC** §2.3c recording-level sanity check (report only).
- **Key fact behind A2:** 95% of valence variance and 91% of arousal variance across the 60 excerpts is between-piece; the condition explains 1% (V) / 4% (A).
- **Phase B — descriptor repairs + analysis upgrades (APPLIED 2026-10-10; results pending the next Kaggle run).** Analysis upgrades: **B-NMS** nested model selection replaces the single default classifier · **B-INC** Freedman–Lane incremental test in §4.9 (ΔR² = R²(Condition + Features) − R²(Condition-only); reduced model = OLS on the centred condition one-hot; residuals permuted within piece; closed-form refits; BH over the 8 tests; columns `r2_cond_plus`, `d_r2_incremental`, `p_incremental`, `q_incremental`) · **B-NC** split-half noise ceiling (200 random rater halves, centred within piece, Spearman–Brown; uncentred for context; `r2_over_ceiling`) · **B-MERTFULL** regression heads on MERT-full (`ablate(X_reg=…)`, regression rebuild assert, §4.2 `Ridge-nested · MERT-full` row; `4A-mert-pca` unchanged). Descriptor repairs: **B-shared** CREPE tracks cached, `fix_octave_errors`, `segment_notes` (never across unvoiced gaps) · **B8** vibrato · **B9** portamento (+ `4A-crepe-portamento` removed) · **B10** F0 range P95 − P5 on the corrected track · **B11** L1-normalised flux · **B12** dynamic-range silence gate + 3 × 3 sweep · **B13** `timing_*` family (note onsets from CREPE, 151-frame tempogram × log-normal prior, `timing_tempo_logsd`), `dtw_global_tempo_ratio` → `dtw_duration_ratio` · **B-NF** spectral-flatness noise floor + FLAT_MIN sensitivity + optional session map · **B-cache** `_v2` names + assert.
- **Phase C (deferred idea):** augmentation (see §5 item 11).

### Phase B — old → new feature names

| Old (Phase A) | New (Phase B) |
|---|---|
| `crepe_vibrato_rate_hz` (argmax inside the [4,8] Hz band) | `crepe_vibrato_rate_hz` (per note ≥ 800 ms, [3,10] Hz, parabolic peak, duration-weighted median over vibrato notes) |
| `crepe_vibrato_depth_cents` | `crepe_vibrato_extent_cents` (half-extent) + `crepe_vibrato_coverage` |
| `crepe_portamento_count` | `crepe_portamento_rate` (glides / voiced s) + `crepe_portamento_share` (glides / transitions) |
| `crepe_portamento_mean_ext_cents` | same name, new glide detector |
| `crepe_f0_range_cents` / `crepe_f0_cv` / `crepe_f0_mean_hz` | same names, on the octave-corrected track; range = P95 − P5 |
| — | `crepe_vibrato_n_notes` (diagnostic) |
| `essentia_spectral_flux_mean` / `_std` | `essentia_spectral_flux_norm_mean` / `_std` (+ diagnostic `essentia_spectral_flux_raw_mean`) |
| `essentia_dynamic_range_db_mean` | same name, silence-gated |
| `dtw_local_tempo_mean_bpm` | `timing_local_tempo_bpm` |
| `dtw_tempo_variability_std` / `_cv` | dropped → `timing_tempo_logsd` |
| `dtw_onset_rate` | `timing_note_rate` |
| `dtw_ioi_mean_s` / `dtw_ioi_std_s` / `dtw_ioi_cv` | `timing_ioi_mean_s` / `timing_ioi_sd_s` / `timing_ioi_cv` |
| `dtw_pulse_clarity` | `timing_pulse_clarity` |
| — | `timing_octave_jump_frac` (diagnostic) |
| `dtw_global_tempo_ratio` | `dtw_duration_ratio` ("Duration ratio vs MEC (>1 = slower)") |
| `make_default_clf`, `DEFAULT_CLF_NAME='NaiveBayes'` | `make_nb_clf`, `NB_CLF_NAME`; `DEFAULT_CLF_NAME='NestedSelect'` |

Variable names (`DTW_REFFREE`, `df_dtw`, `DTW_*`) and the lattice block name `DTW` were kept. Deviations from the literal Phase B spec (stated in the cells): vibrato notes ≥ 800 ms instead of 400 (presence resolution at fs = 100 Hz); tempogram statistics over interior frames only (edge frames ramp to zero and fake tempo variability); the raw diagnostic flux is computed on the gated frames as well; `feat_crepe_v2.csv` is recomputed from the cached tracks every run; the timing features raise if an excerpt has < 3 notes; §6.2 takes its lattice-equivalent blocks from the §4.4b stacks (identical arrays, so the §6.2b equality guard is not exposed to float32 rounding flipping a discrete inner-fold choice).

### What to check after the Phase B Kaggle run

- [ ] §2.2 octave corrections per excerpt; vibrato rate has **> 20 distinct values**; coverage / half-extent ranges plausible (half-extent typically ~10–40 cents); **declared vibrato substitution count** (no note ≥ 800 ms: ___ / 60 — ⚠️ fires above 10; no vibrato note: ___) and the 600/800/1000 ms sensitivity; F0 range max (expected well below the old 5343 c); segmentation diagnostic.
- [ ] §2.2 portamento extents within 80–2000 c and glides not on every transition.
- [ ] §2.3 r(flux, LUFS) raw vs normalised; DR max and % gated; DR sensitivity ρ.
- [ ] §2.3c flatness distribution, flatness-floor flags and their sensitivity; session map still "not provided".
- [ ] §3.3b timing manipulation check on the new `timing_*` features.
- [ ] §4.0 `os.cpu_count()`, parallel == serial guard; nested-selection config frequencies per call.
- [ ] Does NestedSelect beat the strict chance P95 under `piece` (§4.1 / §4.5b / §7.1)?
- [ ] MERT-full in the lattice regression heads (`n_features_reg`), regression rebuild assert.
- [ ] §4.9 B-INC incremental p / q for Δarousal and Δvalence; noise ceilings and R² / ceiling.
- [ ] §5.1b dose-response for the repaired vibrato / timing features; §5.6 vibrato family; §6.1b probe for the new CREPE descriptors (re-check the `CREPE_UNIQUE2` pair in §6.2).
- [ ] Total runtime (Phase A: 75.5 min; nested selection adds an estimated ≈ 45 min on 4 cores / ≈ 1.5 h on 2 cores).

### Phase A Kaggle run (2026-10-09) — checklist outcome

- [x] §2.1 restore message printed; the embeddings turned out identical to the pre-fix run (false alarm).
- [x] §2.3c: 0 LUFS outliers, 0 clipping, 21 noise-floor flags (15 inconclusive) — estimator replaced in Phase B (B-NF).
- [x] §4.0 equivalence assert passed; Dummy floors vary across folds (uniform 0.272 / stratified 0.183 under loo).
- [x] §4.5b: no default-classifier row beats the strict P95 under `piece` (NB below chance everywhere).
- [x] §4.4b lattice, §4.4d label-reproduction, §6.2b guards passed.
- [x] Tier 4D: balanced SVM + inner-CV C beats NB under `piece` on all three matrices.
- [x] §4.9: Interpretable beats Condition-only descriptively for both targets (untested → B-INC).
- [x] §5.5: CoE negative under `piece` (interpretable wins).
- [x] §6.1b: α hit the grid edge in 32 folds.
- [x] Runtime: 45.3 → 75.5 min (+30 min), dominated by §4.1 inner-CV tuning and the 4D tier.