# Thesis Handoff — Violin Expressiveness → Emotion Recognition

**Purpose:** complete standalone context to continue work in a fresh conversation. Covers (1) scope, (2) current pipeline architecture, (3) progress, (4) findings, (5) open issues, (6) technical report status, (7) conventions, (8) workflow, (9) quick answers, (10) audit, **(11) listening study — final facts (N = 116)**, **(12) diploma thesis (LaTeX) — writing status**.

> **Latest (2026-11-03):** the **Phase B run on N = 118 is done** (Kaggle, Tesla T4, papermill 2026-10-09 23:45 → 2026-10-10 01:22 UTC, ≈ 97 min, 101 cells, 0 errors) — §4 now records those results, labelled **N = 118 / pre-block-weighting**. Applied in the notebook since then (now **104 cells**), not yet run: the participant exclusions in §1.1 (N = 118 → 116), **equal-share block weighting** of the multi-block MERT-full regression heads (`REG_BLOCK_WEIGHTING`, tier `4E-block-weighting`) and the **temporary §2.6 manual descriptor audit**. **Next Kaggle run:** interactively up to §2.6 first (inspect the audit), then the full run. The diploma thesis (NTUA template, LaTeX) has Chapters 1–7 written, 73 of a 120-page cap (§12).

> **Latest (2026-11-10): Phase B3 — APPLIED, NOT RUN** (§10, "Phase B3"). The B2 notebook was run in full on Kaggle by the author (N = 116; outputs not committed). That run showed that the B2 **re-articulation splits followed the vibrato amplitude modulation** (MEC 24 / EXP 51 / EXG 68 splits for the same scores) and inflated the note-based timing features; B3 **disables the detector for every feature** (`REART_ENABLED = False`, kept as a report-only negative result), re-checks the 80 c range after a plateau merge, rejects vibrato windows whose best rate is on the 4 / 8 Hz grid edge, renames the descriptor caches to `feat_crepe_v4.csv` / `dtw_features_v4.csv`, and adds **§4.4e** (MERT generalisation: train vs test, validation curve, nearest-neighbour identity — capacity overfitting vs piece-identity shortcut) and **§5.6b** (family overlap: pair-removal tier `3-subfamily-pairs`, cross-family probes, correlations, a pre-declared triangulation rule). Notebook 108 cells. **Next:** interactive run up to §2.6 (A7 acceptance criteria), then the full run.
>
> **Earlier (2026-11-05): Phase B2 — descriptor repairs after the manual audit — APPLIED (since run in full)** (§10, "Phase B2"). The committed notebook carries an **N = 116 run of 2026-11-03** (papermill 15:26 → 17:11 UTC, with block weighting and the §2.6 audit); its §2.6 outputs were audited by eye and ear on 2026-11-05 and showed four descriptor faults (Viterbi-ramp portamento false alarms, notes merged with slides, real narrow vibrato rejected, repeated notes merged). B2 replaces the note segmentation (plateaus + re-articulation splits from a 10 ms RMS envelope), the glide rule (Viterbi-aware) and the vibrato detector (windowed sinusoid fit); caches `feat_crepe_v3.csv`, `dtw_features_v3.csv`, `rms10_v3.npz`. The 2026-11-03 run's Part 2–7 numbers are not recorded here and are superseded. **Next:** interactive run up to §2.6 (calibration grids + AUTHOR VALIDATION V1–V6), then the full run.
>
> **Revision status (2026-11-03): Phase B run on N = 118; N = 116 + block weighting + §2.6 audit applied (since run on 2026-11-03, see above).** Phase A was applied and run on Kaggle (2026-10-09); its numbers (kept in §10 as history) are superseded by the Phase B run. Every Part 4/5/6 predictive number from runs before Phase A is pre-audit and superseded.
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
- **118 responses across 5 linked Google Forms; 116 after screening** (exclude `questionnaire_5_P012`, `questionnaire_5_P020`; §11 — applied in notebook §1.1 via `EXCLUDED_PARTICIPANTS` since 2026-11-03). Ratings: **valence 0–6**, **arousal 0–6**, plus 1–3 emotion tags per excerpt (7 GEMS terms + Neutral).
- **Ring (linked) form design:** every form holds all 20 pieces, one rendition each (8 MEC, 8 EXP, 4 EXG). Renditions follow a **cyclic rotation of (MEC, MEC, EXG, EXP, EXP)** over F1–F5, shifted by one form for each of 5 groups of 4 pieces (§11). So every MEC/EXP excerpt is in two adjacent forms (**8 anchors shared with each neighbouring form**, F5↔F1) and every EXG excerpt in one (**4 unique per form**): the five forms connect cyclically through **40 anchor excerpts**. This linking underpins the normalization validation (§3/§1.2b). Each listener hears each piece once.
- **Scale 0–6** (7 points, midpoint **3.0**). `0` is a genuine lowest rating (verified in raw forms), **not** NA. `VA_MID = 3.0` — correct, do not change.
- **Ground truth = 5 classes** (4 circumplex quadrants + Neutral), derived from **normalized** VA means (see §1.1b flow). `NEUTRAL_RADIUS = 0.75` (un-tuned — see open issues).

**Environment:** Kaggle GPU, PyTorch 2.3.1 — pinned for the P100 (last sm_60 build); the Phase A and Phase B runs used a Tesla T4, which 2.3.1 also supports, so the pin is kept.

---

## 2. Pipeline Architecture (current, 108 cells — Phase B + N = 116 exclusions + block weighting + temporary §2.6 audit + B2 / B3 descriptors + §4.4e / §5.6b diagnostics)

Runs top-to-bottom. `§` = the in-notebook section banners.

| Part | § | Content |
|---|---|---|
| **0 Setup** | 0.1–0.4 | install / imports / **config** (`VA_MID=3.0`, `NEUTRAL_RADIUS=0.75`, `MERT_LAYERS=[5,6,7]`, `QUADRANT_EMOTIONS`, tag maps) / audio manifest |
| **1 Audience ground truth** | 1.1 | parse 5 CSVs → `df_long` (keeps `0`) → **drop `EXCLUDED_PARTICIPANTS`** (screening v3; asserts 116 participants / 2,320 rows / 60 excerpts, prints 118 → 116) |
| | **1.1b** | **per-rater normalization → new ground truth** (per-dim z, σ-floor, rescale preserving grand-mean + between-excerpt SD) |
| | 1.2 | aggregate → `df_agg`; derive 5-class labels from **normalized** VA; raw kept in parallel |
| | **1.2b** | **normalization validation** (anchor cross-form Δ, quadrant stability, corr, variance) |
| | 1.3 | label matrix `Y` (60×N), `mlb`, `df_expr`, `df_all` |
| | 1.4 | rating stats — Friedman / Kendall W / Wilcoxon |
| | 1.5 | lift analysis (emotion↔condition, raw tags) |
| **2 Feature extraction** | 2.1 / **2.1b** | **MERT** (768-dim, layers 5–7; **positional-conv weight-norm tensors restored from the checkpoint + asserted**, cache `emb_mert_v2.npz`) / **controlled MERT forwarding** (PCA-95, unsupervised) |
| | 2.2 / **2.2b** | **CREPE** — Phase B / **B2** / **B3**: tracks cached once (`crepe_tracks_v2.npz`) + 10 ms RMS envelope (`rms10_v3.npz`, report-only since B3), octave correction, **plateau note segmentation** (B3: **no re-articulation split** — `REART_ENABLED = False`; the merge re-checks the 80 c range), then **sinusoid-fit vibrato (B8b; B3 band-edge rule)** / **Viterbi-aware portamento (B9b)** / F0 (B10) descriptors from the tracks (`feat_crepe_v4.csv`), old vs deployed segmentation counts, the report-only re-articulation evidence table (detector off vs on, 3 / 6 / 9 dB), the note-count condition-independence check (Friedman), MEC negative-control calibration grids (report only); helpers in the `# >>> HELPERS:descriptors` block / **controlled CREPE forwarding** |
| | 2.3 / **2.3b** | **Essentia** (dynamics/timbre/tonal) — Phase B: flux on L1-normalised spectra (B11), silence-gated dynamic range (B12) / **controlled Essentia forwarding** |
| | **2.3c** | **recording-level sanity check** (report only) — peak/clipping, LUFS robust z, within-piece LUFS, **spectral-flatness noise floor** "recording-chain change" test (B-NF), optional session map |
| | 2.4 / **2.4b** | **timing + DTW (ACTIVE)** — reference-free `timing_*` family (CREPE note onsets — B3: no re-articulation split — + 151-frame tempogram with log-normal prior, B13; `dtw_features_v4.csv`) + within-piece `dtw_*` warp-path family / **controlled forwarding**. madmom / MusiCNN remain skip stubs |
| | 2.5 | assemble `X_A/B/C/D/F`, `X_MERT_FULL`, regression twins `X_C_REG/X_D_REG/X_F_REG` (B-MERTFULL, stored raw); `STRICT_BLOCKS` guard; no pre-Phase-B cache name |
| | **2.6** | **TEMPORARY manual descriptor audit** (between `# >>> AUDIT` / `# <<< AUDIT`; `RUN_MANUAL_AUDIT`, `AUDIT_EXCERPTS` = Méditation + Mozart VC4 × MEC/EXP/EXG; `_aud_*` names; writes only `manual_audit/`): recomputes the 30 interpretable features with the pipeline's own (B2 / B3) code on the deployed path (no re-articulation split) and asserts equality and detail consistency (windows → vibrato features, transitions → portamento); audio players, overview + zoom figures (vibrato windows + fitted sinusoids, band-edge windows purple, 10 ms envelope for reference), notes / windows (with `edge` / `reason`) / transitions / key / LUFS tables, direction checks, checklist template, **AUTHOR VALIDATION V1–V7** (PASS / FAIL / INFO, never raised; B3: V4 INFO, V2 / V3 need ≥ 80 % span coverage, V7 = note-count Friedman). Removable |
| **3 Validation & EDA** | 3.1–3.4 | class floors · VA/coverage · CREPE sanity · **3.3b DTW timing sanity + manipulation check** · MERT structure (PCA scree) |
| **4 Prediction** | **4.0** | **evaluation protocol & CV engine** — `get_cv` (piece = PRIMARY, loo = secondary), **nested model selection default** (`cv_predict_nested_select`, B-NMS; parallel == serial guard), `cv_predict_multilabel`, `nested_ridge_predict` (+ optional `block_sizes`: `block_weights` / `BlockScaler`, equal-share block weighting; guard on `X_C_REG`), `oos_r2`/`rmse`, `bh_fdr`, Freedman–Lane + split-half helpers, re-seeded Dummy floors (mean/SD/P95; prevalences printed from `Y`), equivalence guard |
| | 4.1 | multi-label classification — NestedSelect (★default) + **8-classifier sensitivity table** (not used for selection; NaiveBayes = Phase A default), both protocols; balanced + inner-CV C for SVM/LogReg (A5) |
| | 4.2 | valence/arousal regression (fixed models + `Ridge-nested · MERT-full`), both protocols, r · R² · RMSE |
| | 4.4 | feature-set table + Wilcoxon (four hand-picked sets), per protocol |
| | **4.4b** | **ablation engine + block lattice** — all 15 subsets of {MERT, CREPE, Essentia, DTW}; unique vs marginal contribution per tool; multi-block MERT-full regression heads use the deployed block weighting (`_abl_reg_kw`, rows record `block_weighting` + α grid-edge hits) |
| | **4.4c** | **configuration ablation** — `MERT_PCA_VAR` · Essentia curation · DTW family · shared-gate `corr_max` · **`4D-balancing`** (nested selection = reference; NB priors / SVM balancing, descriptive) · **`4E-block-weighting`** (the 7 multi-block MERT-full regression heads, unweighted vs equal-share weighted; Δ, paired permutation, BH over the regression metrics; descriptive). `4A-crepe-portamento` removed in Phase B |
| | **4.4d** | **target-side ablation** — `NEUTRAL_RADIUS` 0.5/0.75/1.0 (macro-F1 with *and* without Neutral) · normalized vs raw ground truth |
| | **4.4e** | **MERT generalisation diagnostics (B3)** — B1 train vs test F1 (per-fold NestedSelect configs from the §4.0 memo, refitted) and train vs out-of-sample R² for MERT-PCA25, MERT-full, CREPE, Essentia, DTW, the 30-feature layer and a 25-column N(0,1) capacity control, both protocols · B2 validation curve (`piece`, SVM-Linear at fixed C) · B3 nearest-neighbour identity (same piece / same condition vs chance, 10 000 label shuffles, silhouettes) · B4 pre-declared rules (capacity overfitting / piece-identity shortcut); `mert_diagnostics/` |
| | 4.5 / **4.5b** | per-class P/R/F1 (default on X_C) / **DummyClassifier floors** (the official floor; mean, SD, P95 over 200 re-seeded repeats) |
| | 4.6 | per-condition F1 (MEC/EXP/EXG), default on X_C |
| | 4.7 | quadrant accuracy from predicted VA (+ VA scatter, circumplex), both protocols |
| | 4.8 | prediction–ground-truth agreement (Krippendorff α), default on X_C |
| | **4.9** | **within-piece prediction** — piece-centred targets & features, piece CV only, Condition-only reference, within-piece permutation p; **B-INC** Freedman–Lane incremental test (Condition + Features vs Condition-only); **B-NC** split-half noise ceiling |
| **5 Attribution** | 5.1 | **condition-delta** — technique change vs rating change (within-piece); **BH-FDR over all 180 tests** (30 features × 2 targets × 3 transitions) |
| | **5.1b** | **dose-response** — Page's trend test over ordered MEC<EXP<EXG |
| | 5.2 | **SHAP** technique importance per class |
| | 5.3 | **embedding-distance** attribution |
| | 5.4 | **interpretable linear read-out** (signed technique→emotion) |
| | 5.5 | **cost of explainability** (interpretable vs representation, same CV, both protocols; headline max(MERT-PCA, MERT-full) + the two separate gaps; handles a negative CoE) |
| | **5.6** | **sub-family ablation** — decomposes the CoE across Pitch/Vibrato/Portamento/Dynamics/Timbre/Tonal/Timing; B3: chance lines + q < .05 markers in the figure, commonality summary (Σ alone / Σ loss / shared), the pre-declared timing reading; writes the consolidated `ablation_master.csv` |
| | **5.6b** | **family overlap (B3)** — C1 pair removal (tier `3-subfamily-pairs`, interaction I(A,B), `piece`) · C2 cross-family probes (nested ridge, raw and within-piece) · C3 Spearman ρ heatmaps (raw / within-piece) + family mean \|ρ\| · C4 pre-declared triangulation (≥ 2 of 3) · re-exports `ablation_master.csv` |
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

- **Phase B3 (2026-11-10, applied, not run).** Changed cells (outputs of changed code cells cleared): Part 0 intro (md), §0.3 (`OLD_CACHE_NAMES` += the two v3 caches), §2.2 (md + code: `REART_ENABLED = False`, `reart_enabled` switch, merge range check, band-edge rule, `vibrato_window_reason`, banner, report-only re-articulation evidence table, note-count Friedman, `feat_crepe_v4.csv`), §2.4 (md + code: `dtw_features_v4.csv`, comments), §2.5 (cache asserts), §2.6 (md + both code cells: disabled path, band-edge display, V4 INFO, ≥ 80 % span rule, V7), §4.4c (one print), §5.6 (md + code: chance lines, q markers, commonality summary, `q_loss_r2_val`, timing reading), output reference (md). New cells: §4.4e (md + code, after §4.4d) and §5.6b (md + code, after §5.6). Details in §10, "Phase B3".
- **The B2 notebook was run in full on Kaggle (N = 116)** by the author; its outputs are not committed (the repo notebook keeps the 2026-11-03 outputs of the unchanged cells). The numbers that motivated B3 are in §10, "Phase B3", evidence (a)–(e).
- **Phase B2 — descriptor repairs after the manual audit (2026-11-05, applied, since run).** Changed cells (outputs cleared): §0.3 (`OLD_CACHE_NAMES` += the two v2 descriptor caches), §2.2 (md + code: B2 segmentation, B8b vibrato, B9b portamento, Viterbi-source check, `rms10_v3.npz`, old vs new note counts, REART sensitivity, calibration grids), §2.4 (md + code: `env_db` threaded into the timing onsets, `dtw_features_v3.csv`), §2.5 (cache assert), §2.6 (md + both code cells: B2 tables / figures, detail asserts, AUTHOR VALIDATION V1–V6), §4.4c (one "still pending" print), §5.6 (md + the vibrato caveat print), output reference. Details in §10.
- **The N = 116 run (2026-11-03, papermill 15:26 → 17:11 UTC)** of the edits below is in the committed notebook; its §2.6 outputs are the audit evidence; its Part 2–7 numbers are not recorded and are superseded by B2.
- **Notebook edits for the next run (2026-11-03, since run).** (1) Participant exclusions wired into §1.1 (`EXCLUDED_PARTICIPANTS` in §0.3; asserts 116 participants / 2,320 rows / 60 excerpts; §1.4 prints the exclusions, the old `df_long_clean` branch is gone; §4.0 prints the prevalences from `Y` instead of the hardcoded N = 118 ones). (2) **Equal-share block weighting** of the multi-block MERT-full regression heads (`block_weights`, `BlockScaler`, `REG_BLOCK_WEIGHTING = 'equal'`; §4.0 guard; §4.4b/§4.4d heads; new tier `4E-block-weighting`). (3) **Temporary §2.6 manual descriptor audit** (6 excerpts). (4) Descriptive text made accurate (diploma thesis / NTUA, thesis framing and questions, torch pin = P100 / runs = T4, stale markdown in §2.5, §4.1, §5.2, §5.4, §6.1, §6.2, output reference). The notebook is now 104 cells.
- **Phase B run on N = 118 (papermill 2026-10-09 23:45 → 2026-10-10 01:22 UTC, T4, ≈ 97 min, 101 cells, 0 errors).** Results in §4. It surfaced the MERT-dominance problem of the multi-block regression heads (→ block weighting) and several descriptor questions (→ §2.6 audit, §5).
- **Listening-study QC re-validated and finalised (2026-10-21).** The months-old cleaning scripts were checked against the five CSVs; the participant screening was rewritten (`questionnaire_cleaning_v3.py`, four bugs fixed — §11) → 2 exclusions. Excerpt QC re-run at N = 116. Demographics and durations scripts written. The normalization and ground truth were reproduced offline: at N = 118 the reproduction matches the notebook exactly, so the N = 116 numbers in §11 are what the notebook will produce once the exclusions are wired in.
- **Diploma thesis Chapters 1–7 written (2026-10-18 → 2026-10-30)** — front matter, Introduction, background Chapters 3–6 rewritten against Phase B, Chapter 7 Dataset written from scratch with N = 116 numbers; colour palette for all figures; numbered subsections. See §12.

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

> **Phase B results — Kaggle run on N = 118 (papermill 2026-10-09 23:45 → 2026-10-10 01:22 UTC, Tesla T4, 101 cells, 0 errors), pre-block-weighting.** The B2 run on N = 116 (outputs not committed; its B3-relevant numbers are in §10, "Phase B3") and the next **B3** run supersede them; every multi-block MERT-full regression number below is unweighted, and **every vibrato / portamento / timing-onset number below uses the Phase B descriptors that the 2026-11-05 audit found faulty** (§10, Phase B2).

**Guards.** All passed (§4.0 equivalence, NestedSelect parallel == serial on 4 CPUs, §4.4b lattice rebuild incl. the `_REG` matrices, §4.4c 4D, §4.4d label reproduction, §4.9 closed-form equivalence incl. B-INC, §5.5↔§5.4, §6.2b exact, §2.5 cache names).

**§2.2 CREPE.** Octave fixes 3 frames in 3 excerpts. Notes ≥ 800 ms: 259 (600 ms: 391, 1000 ms: 198); no qualifying note in 1 excerpt (`Shostakovich_SQ8_exrpt3_MEC`). 121 vibrato notes; rate undefined → declared median 5.811 Hz for 23 excerpts (all MEC with notes, plus SQ8 exrpt1 EXP/EXG and exrpt3 EXP/EXG — the author confirmed by ear that no vibrato was played there, so these are correct rejections). Rate 37 distinct values; half-extent 0–32.1 c; coverage 0–1. Portamento: 139 glides in 572 transitions (24.3 %), 50–312 ms (median 62), 80–1907 c. F0 range max 3100 c. 1626 notes ≥ 60 ms, 17 adjacent notes < 70 c apart.

**§2.3 / §2.3c Essentia, recording level.** DR gate median 6.2 % of frames (max 26 %), DR max 34.7 dB; 6 of the 9 gate variants have ρ < 0.9 vs the deployed DR (0.79–0.85). r(flux, LUFS) raw 0.874 → normalised 0.508. Forwarded 10 (short-term loudness SD dropped, r 0.97 with LRA). Flatness noise floor: 0/60 conclusive at FLAT_MIN 0.2 (9 at 0.1); 0 flags; LUFS by condition MEC −26.8, EXP −24.1, EXG −22.6; session map not provided.

**§2.4 / §3.3b timing.** DTW forwarded 10 (7 reference-free + 3 deviation; inside 3–10, no warning). Group means: tempo log2-SD MEC 0.261 > EXP 0.232 > EXG 0.209; IOI CV / SD and pulse clarity not monotone (0/3 non-decreasing). Author's explanation (2026-11-03): in EXG he played faster and more virtuosically in fast pieces and took more time in slow ones → local tempo / duration ratio are piece-dependent; tempo log2-SD and pulse clarity are checked visually in §2.6.

**Chance floors.** piece: uniform mean 0.2715, stratified 0.161, strict P95 0.3274; loo: 0.2720 / 0.183 / 0.3202.

**Classification (NestedSelect default).** piece: A 0.282, **B 0.361 (the only set above the strict P95)**, C 0.302, D 0.281; loo: A 0.517, B 0.476, C 0.626, D 0.540. NaiveBayes (Phase A default) piece: A 0.112, B 0.260, C 0.162, D 0.126. On X_C (piece) LA_HV and Neutral collapse (recall 0); per condition MEC 0.24 / EXP 0.29 / EXG 0.32; Krippendorff α HA_HV 0.31, LA_LV 0.31, others ≤ 0.03. Tier 4D (piece): Interp + SVM-Linear balanced + inner-CV C 0.417 [0.346, 0.475] (Δ +0.063 vs nested, q = 0.13); NB below nested on X_A / X_C / Interp (q ≤ 0.017).

**Regression (nested ridge, piece).** Ridge-nested · MERT-full R²v +0.016, R²a +0.446. Lattice: Essentia + DTW R²v 0.29, **R²a 0.67** (best); CREPE + Essentia + DTW R²a 0.60; every set containing MERT-full R²a 0.446–0.459 and R²v 0.014–0.019 — the 768 MERT columns dominate a single-α ridge (→ equal-share block weighting, next run). 4A-mert-pca: R²a PCA −0.25 vs full +0.45 (Δ +0.70, q = 0.0015), R²v −0.17 vs +0.02 (q = 0.039), F1 0.28 vs 0.33 (n.s.) — PCA-95 keeps the high-variance (piece-identity) directions and discards the emotion-relevant variance. Quadrant accuracy (fixed Ridge α = 10 on X_A) 0.25 piece / 0.62 loo.

**§4.4b unique contributions (piece).** F1: MERT −0.053, CREPE −0.045, Essentia −0.051, DTW +0.036 (none significant); full set F1 0.302 (below P95). **§4.4d:** NEUTRAL_RADIUS 1.0 beats 0.75 on both columns for X_A and X_C (rule triggers — pending decision); raw ground truth F1 0.34 vs normalised 0.30.

**§4.9 within-piece (piece-centred, LOPO).** Noise ceilings (split-half, centred): V 0.455 [0.088, 0.718], A 0.678 [0.520, 0.813]. Δvalence: Condition-only R² 0.047 (p 0.034), Interpretable 0.174 (p 0.006), MERT-full 0.175, MERT-PCA 0.134; **B-INC incremental ΔR² Interpretable +0.129 (q 0.037), MERT-full +0.128 (q 0.037), Interp + MERT-PCA +0.117 (q 0.037)**, MERT-PCA +0.077 (q 0.070). Δarousal: Condition-only 0.408, Interpretable 0.423, incremental +0.024 (p 0.044, q 0.070); the MERT sets add nothing. R²/ceiling (Interpretable) V 0.38, A 0.62.

**§5.1 BH.** 180 tests (30 features × 2 targets × 3 transitions): 10 raw p < .05 vs 9.0 expected, **0 q < .05** → the condition-delta correlations are **not evidence; do not cite them as findings**.

**§5.1b dose-response (Page, 20 pieces).** Techniques: spectral flux (normalised) z +5.22, **vibrato extent +5.14, vibrato coverage +4.90**, LUFS +4.74, DTW deviation +4.59 (built in at MEC), DR +4.11, rolloff −2.85, **tempo log2-SD −2.21 and local tempo −2.06 (both decrease)**; vibrato rate +1.90 n.s.; portamento n.s. Audience: arousal z +4.11, valence +1.26 n.s. → with the repaired descriptor, **vibrato extent/coverage do escalate**; arousal (not valence) follows the manipulation.

**§5.4 / §5.5 cost of explainability (piece; negative = interpretable wins).** Read-out V r 0.25 / R² 0.12, A r 0.77 / R² 0.62. Interpretable F1 0.353, R²v 0.120, R²a 0.620 vs MERT-PCA 0.282 / −0.171 / −0.252 and MERT-full 0.328 / 0.016 / 0.446. Headline ΔR valence −0.181, ΔR arousal −0.140, ΔR² valence −0.104, ΔR² arousal −0.174, ΔF1 −0.071; vs MERT-PCA ΔR² arousal −0.872.

**§5.6 sub-families (piece).** Removing Portamento raises F1 to 0.465 (loss −0.112, q 0.015) and removing Timbre to 0.424 (q 0.009) — consistent with portamento false alarms (under audit); removing Dynamics costs R²a 0.090 (q 0.11); Dynamics alone R²a 0.44; Vibrato alone F1 0.28, R²a 0.04.

**§6.1b probe / §6.2.** Flux 0.86, F0 mean 0.80, centroid 0.73, HPCP entropy 0.73, pulse clarity 0.71, note rate 0.65, vibrato coverage 0.48, F0 range 0.43; LUFS 0.08; vibrato rate −0.02. α hit the grid edge in **34** outer folds. §6.2 (piece): + all CREPE +0.057, + DTW +0.052, + Essentia −0.001, + 2 CREPE-unique −0.019. Master ablation table 138 rows / 9 tiers.

### Historical — Phase A run (2026-10-09; superseded by the Phase B run above)

**Guards (Phase A).** All passed: §4.0 engine equivalence, §4.4b lattice rebuild, §4.4d label reproduction, §6.2b, §4.9 closed-form permutation equivalence, §5.5↔§5.4 cross-check. 101 cells, 0 errors.

**Dummy floors fixed.** Uniform mean 0.272 (loo) / 0.2715 (piece); stratified 0.183 / 0.161; strict P95 0.320 (loo) / 0.327 (piece).

**MERT (A3) — false alarm.** The restore ran, but the embeddings are identical to the pre-fix run (the §6.1 correspondence table and every MERT-based F1 match to 4 decimals). The transformers "newly initialized" warning was spurious; pre-audit MERT numbers were valid. The assert stays.

**Classification, piece (PRIMARY).** The pre-declared default GaussianNB is **below chance on every set** (F1-macro A 0.112, B 0.287, C 0.168, D 0.113); on X_C three classes collapse (HA_HV, LA_HV, Neutral recall 0); quadrant accuracy 0.25; Krippendorff α ≤ 0 for all labels except LA_LV (0.41). LOOCV for comparison: C 0.686. **Tier 4D (piece):** SVM-Linear balanced + inner-CV C beats NB deployed on X_A (+0.143, q=0.023), X_C (+0.156, q=0.005), Interp (+0.151, q=0.023); Interp + SVM balanced = 0.385 [0.327, 0.434] — the only configuration above the strict chance P95; NB balanced priors: no effect. → Decision (Phase B): nested model selection replaces the single default classifier; NB stays reported as "pre-declared Phase A default, failed under piece".

**Regression, piece (nested ridge).** Arousal R² — X_C 0.50, CREPE+Essentia 0.52, Essentia 0.40; valence R² ≈ 0 (max 0.23, CREPE+DTW). MERT-PCA: R²v −0.17, R²a −0.25; MERT-full: R²a 0.45 (Δ vs PCA +0.70, q=0.0015) → PCA-95 discards the arousal-relevant variance and keeps piece identity (a tool-taxonomy finding).

**Cost of explainability, piece (negative = interpretable wins).** Headline ΔR valence −0.215, ΔR arousal −0.054, ΔF1 −0.127; vs MERT-PCA ΔR² arousal −0.756.

**§4.9 within-piece (piece-centred, LOPO).** Δarousal — Condition-only R² 0.41, Interpretable 0.47 (p=0.002), MERT-full 0.38, MERT-PCA 0.27. Δvalence — Condition-only 0.047 (p=0.034), Interpretable 0.20 (p=0.004), MERT-full 0.175, MERT-PCA 0.134. Within-piece SD: V 0.234, A 0.285 (scale 0–6). The "beats condition" difference was not yet tested (→ Phase B B-INC).

**§5.1 BH (Phase A).** 0 q < .05 (12 raw p < .05) over the Phase A family (29 interpretable features then; the current §5.1 runs 180 tests — see above).

**§6.1b grouped probe (Phase A).** F0 mean 0.80, spectral centroid 0.73, HPCP entropy 0.73, pulse clarity 0.59; loudness 0.08 (the α grid-edge count is 34 outer folds in the Phase B run — the earlier "32" is superseded).

**§2.3c recording level.** 0 LUFS outliers, 0 clipping, 21 noise-floor flags (15 excerpts inconclusive). EXG conclusive floor mean −66.2 dBFS vs MEC −71.4 / EXP −72.5; EXG LUFS +4.2 LU vs MEC. Facts from the author: 3 recording sessions with position markers; in EXG the performer deliberately played louder. Equality of gain across sessions is **not independently verified** (limitation); the 5th-percentile floor estimator is unreliable when an excerpt has no silence (→ Phase B B-NF). A session map is not available yet.

### Facts not changed by Phase B (Part 1 at N = 118 unless stated; N = 116 values in §11)

**Variance structure (audit).** 95% of valence and 91% of arousal variance across the 60 excerpts is **between-piece**; the condition explains **1% (V) / 4% (A)**. LOOCV (which keeps a piece's other two renditions in training) therefore mostly measures piece recognition; leave-one-piece-out and the within-piece analysis (§4.9) are the honest tests.

**Former "Headline 2" (escalation via dynamics & timbre, not vibrato) — withdrawn.** It came from the Phase A dose-response with the broken vibrato descriptor and the raw flux. In the Phase B run the repaired vibrato extent / coverage escalate strongly (z +5.14 / +4.90) next to normalised flux and loudness; only the vibrato *rate* stays n.s. (see §5.1b above).

**Normalization validation.** Raw grand mean V=2.97 / A=3.40 (arousal legitimately skews high); between-excerpt SD V=1.065 / A=0.956 (preserved); anchor cross-form |Δ| baseline **V=0.334 / A=0.289**; raw↔norm Pearson r = **0.996 (V) / 0.993 (A)**; **10/60 excerpts change class, 4/60 change quadrant** (the earlier "0/60 flips" claim was wrong). Per-rater min SD 0.51/0.54 → the σ-floor **never fires** (insurance, not active). **Conclusion (unchanged): normalization is a defensibility/robustness step — it confirms the signal is real, not a rater artifact; no material change in results.**

**Stimulus level.** The stimuli played to participants were **not loudness-normalized** (confirmed by the author): level differences between excerpts are part of the stimulus. §2.3c (Phase A) only checks that no recording was made at a clearly different gain/microphone distance.

**Cleaning (final, 2026-10-21 — supersedes the earlier "3 flagged, decision pending").** Screening v3 excludes **2 of 118** (`questionnaire_5_P012`: D1+D2, r = −0.20, MCD 18.4; `questionnaire_5_P020`: D1+D3, valence r = −0.57 / arousal r = +0.78, reversed valence scale). The old "3 flagged" came from the buggy v2 scripts. Kept: `questionnaire_1_P006` (extreme but consistent responder, r = 0.83, D4 only); sensitivity set with near-zero agreement and one flag: `questionnaire_3_P014`, `questionnaire_4_P010`, `questionnaire_5_P018`. Excerpt QC at N = 116: ICC(1,k) V 0.965 / A 0.953; median r_wg 0.60 / 0.59; arousal EXG < MEC in Bach_Adagio_S1 (−0.52), Mendelssohn_VC_Mvt2 (−0.18), La_Vita_E_Bella (−0.09). Full numbers in §11. **Exclusions applied in the notebook (§1.1) since 2026-11-03 — first run pending.**

---

## 5. Open issues

**PENDING DECISIONS (open, unchanged — for the author / supervisor):**
- `DTW_DEV_IN_INTERP=True` vs the thesis CoE definition (condition-encoding features should be out of *both* sides of the cost of explainability; `False` gives a strictly condition-independent CoE).
- `CREPE_UNIQUE2` hard-coded `[f0_range, vibrato_rate]` in §6.2 — read off the pre-Phase-B probe; in the Phase B probe F0 range is well represented (R² 0.43), vibrato rate is not (−0.02).
- The "all-positive" chance floor (≈ 0.322) cited in the thesis is not computed in the notebook.
- The §4.4d NEUTRAL_RADIUS rule triggers: under `piece`, r = 1.0 beats 0.75 on both columns for X_A and X_C (Phase B, N = 118).
- Fig 5 (§4.7) omits the MEC points (`cc` holds only EXP/EXG) / figure dedup.
- §4.2 / §4.7 use a fixed Ridge α = 10 while the rest of the pipeline uses nested ridge.

**Manual audit of 2026-11-05 — confirmed and FIXED in Phase B2 (§10):** portamento false alarms (crepe's Viterbi decoder turns every jump into a ≤ 220 c/frame ramp; Méditation MEC D6→D5 and A5→A4 were "glides" of 62.5 ms) · notes merged with slides and repeated notes merged (Méditation EXP "D6" = C♯6 + slide; Mozart VC4 17 / 16 / 19 notes vs 19 in the score) · real narrow vibrato rejected by the FFT presence / 5 c rules (Méditation EXP D5, B4). The Phase B worry "note splitting by wide vibrato" is covered by the B2 plateau tolerance + merge rule (checked via close splits and V2–V4).

**Phase B3 decision (2026-11-10):** the B2 re-articulation detector is **disabled** for every feature (`REART_ENABLED = False`) and documented as a negative result — its splits followed the vibrato amplitude modulation (§10, "Phase B3").

**Known limitations after B3 (documented, NOT fixed):** repeated notes without a silence count as one (Mozart_VC4 ≈ 17 / 16 / 20 vs 19 in the score; §2.6 V4 is INFO) · slow slides can create short "ramp plateaus" (a glide counted twice, e.g. B5→C6→C♯6 in Meditation_Thais_EXG) · the 80 ms glide minimum rejects the Méditation EXP C♯6→D6 (rise 75 ms) · residual MEC detections (V6: 5 MEC — the glides look like position shifts) · the tempo log2-SD / pulse clarity / log-attack time / `es.Key` items below.

**Audit findings still OPEN (documented, not changed in B2 or B3):**
- integrated LUFS Essentia − pyloudnorm = **+3.055 LU on all 6 audited excerpts** (Essentia duplicates the mono file to 2 channels — constant offset; decide whether to subtract 3.01 LU or report as is);
- `es.Key` gives **G minor for Mozart_VC4_EXP** (D major for the other five audited excerpts);
- Méditation `timing_tempo_logsd` **MEC 0.256 > EXP 0.158 > EXG 0.135** and pulse clarity lowest at MEC (✗ vs the metronomic MEC instruction — the tempogram reacts to articulation);
- log-attack time **not sharper in the spiccato Mozart EXP** (✗); LAT is computed on Essentia waveform onsets (`OnsetRate`), not on note onsets;
- DTW / duration ratio computed over the whole files incl. silence;
- DR gate sensitivity (6 of 9 variants ρ < 0.9);
- §2.3c flatness noise-floor test inconclusive for 60/60.

**Decisions taken (2026-11-03):** excerpt QC (ICC, r_wg, bimodality coefficient, Kruskal–Wallis, crossed additive model) stays **outside** the notebook (offline scripts) · per-MERT-layer caching is **ON HOLD** · the timing pattern is explained by the author (EXG faster / more virtuosic in fast pieces, more time in slow ones → local tempo and duration ratio are piece-dependent; tempo log2-SD and pulse clarity still checked by eye in §2.6) · block weighting `'equal'` deployed for the multi-block MERT-full regression heads (tier 4E descriptive).

**Other open items:**

1. **"8 emotions" is mechanically 5 classes.** `QUADRANT_EMOTIONS` maps each quadrant to a *set*, so pairs (Tenderness≡Peacefulness, Power≡Joyful, Sadness≡Nostalgia) are byte-identical columns. **Decision taken: report honestly as 5 classes** (4 quadrants + Neutral). Alternative (rebuild `Y` from raw multi-label tags) remains available. Good supervisor topic.
2. ~~**Neutral radius un-tuned.**~~ **Measured in §4.4d** — the 0.5/0.75/1.0 sweep reports macro-F1 **with and without** Neutral. In the Phase B run r = 1.0 wins on both columns (see PENDING DECISIONS).
3. **Figure dedup still pending** (partial cleanup only): a duplicate VA scatter still sits alongside the circumplex; check for any remaining per-class-F1 heatmap/radar and ablation-figure redundancy and prune (keep circumplex, P/R/F1 bars, PCA scree).
4. **DTW block size / DTW-dev in `df_interp`.** In the Phase B run the §2.4b gate forwarded 10 columns (7 ref-free + 3 dev) — inside 3–10, no warning. The `DTW_DEV_IN_INTERP` question is listed under PENDING DECISIONS.
5. ~~**Participant exclusions decided but not applied.**~~ **Applied in §1.1 (2026-11-03)**: `EXCLUDED_PARTICIPANTS` dropped before §1.1b, asserts 116 participants / 2,320 rows / 60 excerpts; first run pending. Expected Part 1 results are in §11; `Schindlers_List_EXP` moves HA_LV → LA_LV.
6. **Statistics hygiene:** the ablation study (§4.4b) carries bootstrap CIs, paired permutation tests and **BH-FDR within each tier × protocol** (now over F1, r and R²); **§5.1 now applies BH-FDR over all its tests** (Phase A); §5.1b (dose-response) still wants the same treatment or a pre-registered hypothesis set — not yet approved. Keep phrasing classifier-set comparisons as "across the classifiers tried," not population inference. **The lattice must stay descriptive** — selecting the winning subset on the same CV and then reporting its score is exactly the nested-selection error.
7. **Feature hygiene — repaired in Phase B (run on N = 118; vibrato extent / coverage now escalate, see §4; remaining doubts under audit in §2.6).** The Phase A descriptors were broken: `crepe_portamento_count` was a raw count; `crepe_vibrato_rate_hz` was bandpass-bounded to [4,8] Hz *and argmax-ed inside that same band* (it could never report "no vibrato"); vibrato/portamento ran on concatenated voiced frames. **Pre-Phase-B vibrato nulls** (the n.s. dose-response) are therefore evidence about the descriptor only. Phase B (B8/B9) analyses time-contiguous notes, searches [3,10] Hz with a presence test, splits extent from coverage and reports portamento as rates/shares — a vibrato null from the next run is a **musical result at this N**, with the remaining caveats. **The 2026-11-05 audit then found the Phase B detectors faulty (§10, Phase B2)**; B2 replaces them: plateau notes + re-articulation splits, a windowed 4–8 Hz sinusoid fit on notes ≥ 500 ms (R² ≥ 0.50, half-extent ≥ 3 c), and a Viterbi-aware glide rule (rise 80–500 ms, ≥ 50 ms longer than crepe's forced ramp). The Phase B vibrato escalation (extent z +5.14, coverage +4.90) must be re-read in the next run before the thesis keeps it. B2 detectability limits: a glide is only seen between two plateau notes; vibrato on notes < 500 ms or with half-extent < 3 c is invisible; rates are clamped to 4–8 Hz. **Phase B3:** the re-articulation splits turned out to follow the vibrato and are disabled (a repeated note without a silence is one note); a window whose best rate is on the grid edge is not vibrato; a plateau merge re-checks the 80 c range.
8. ~~**LOOCV leaks piece identity.**~~ **Addressed in Phase A:** leave-one-piece-out is now the PRIMARY protocol everywhere (§4.0), LOOCV is the optimistic secondary, and §4.9 removes the piece entirely (within-piece prediction). Quote the `piece` number for any claim about unseen material.
9. **`LA_HV` has n=5 from only 2 pieces** (La_Vita ×3, Meditation_Thais MEC+EXP): under `piece` it is learned from one piece at a time, and inner-CV tuning is skipped for it in many folds.
10. **Recording sessions.** The session map is **not provided yet** (`SESSION_MAP_CSV`, columns `excerpt_id, session`; §2.3c prints "session map not provided" and skips until it is uploaded). Gain equality across the 3 recording sessions is **not independently verified** (the performer reports playing louder in EXG on purpose).
11. **Phase C (deferred idea) — augmentation.** The supervisor's suggestion: augment the minority positive-valence classes (pitch shift ±1 semitone, tempo ±5%, never combined), with augmented copies used **only in training folds, and only when their source piece is itself in that training fold** (test = originals only, gate/PCA fitted on originals, scoring/permutations on the 60 originals). Caveats: it cannot add piece diversity (LA_HV comes from only 2 pieces); it perturbs studied variables (F0 mean, tempo, onset rate) while keeping labels; the copies' labels were never rated; phase-vocoder artefacts affect Essentia timbre features. **Decision:** evaluate class balancing (A5, tier `4D-balancing`) first; augmentation later as an ablation tier.
12. **Not approved yet:** SHAP fitted in-sample (§5.2) · §5.3 embedding-distance sign · §1.4 mixed model · BH in §5.1b.
13. **Known:** §5.5's headline CoE is a best-of over two black boxes (max(MERT-PCA, MERT-full)) — conservative; the two separate gaps are in `coe_gaps.csv`. (The README now points to `docs/thesis.md`; Fig 5 is under PENDING DECISIONS.)
14. **Parameter ablations still pending:** CREPE step_ms (re-extraction), the CREPE voicing threshold and the B2 descriptor constants as an ablation tier (cheap from `crepe_tracks_v2.npz` + `rms10_v3.npz`; §2.2 prints report-only sensitivities and the MEC calibration grids), Essentia frame sizes, the tempogram window / tempo prior; per-MERT-layer is ON HOLD.
15. **Runtime.** The Phase B run took ≈ 97 min on a 4-CPU T4 session (Phase A: 75.5 min). The B2 run added tier 4E and the §2.6 audit. **B3 adds** §4.4e (≈ 3–6 min: NestedSelect for the interpretable layer and the random control under both protocols, the rest memo hits; nested ridge for 7 blocks × 2 targets × 2 protocols; the validation curve and 10 000 label shuffles are seconds) and §5.6b (≈ 10–15 min, dominated by the pair tier: 29 NestedSelect + ridge fits under `piece` plus 28 × 5 × 2000 paired permutations; ≈ 3× that with `RUN_PAIR_ABLATION_LOO = True`; probes / correlations seconds). §2.2's report-only detector evaluation adds seconds.
16. **Thesis-side open items (2026-10-30):** recruitment channels for the participants (TODO in Ch. 7 §7.4) · mean duration per condition from `piece_durations.csv` (TODO in §7.2.3) · intended quadrant of SQ8_exrpt1 (Q3 but tagged Tension/Power) and SQ8_exrpt3 (Q4 but tagged Tenderness) in the dataset metadata — check before trusting the "Intended" column of Table 7.1 · movement/passage of each work in Table 7.1 · the 6 under-18 participants (shown as "Under 18" in Table 7.5 — ask the supervisor whether consent needs a mention) · author name, committee, NTUA logo, dedication, spelling of Λυμπεράτος/Κανταρέλης, year of `lyberatos2026witheflow` (arXiv 2506 = 2025).
17. **MERT generalisation / family overlap (B3):** answered by §4.4e (capacity overfitting vs piece-identity shortcut) and §5.6b (triangulated overlap) under pre-declared rules — read their verdicts before writing Ch. 9.

---

## 6. Technical report status

> The IEEEtran technical report below is the earlier draft. The **diploma thesis (NTUA template)** is now the main document — see §12.

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
- `df_long` — long-format per-rating (parse, §1.1), **screened** (`EXCLUDED_PARTICIPANTS` dropped in §1.1, `N_PARTICIPANTS_RAW` = 118); gains `valence_norm/arousal_norm` in §1.1b.
- **Block weighting (§0.3 / §4.0 / §4.4b):** `REG_BLOCK_WEIGHTING = 'equal'` · `block_weights(sizes)` · `BlockScaler(sizes)` · `block_sizes=` in `nested_ridge_predict` / `select_ridge_alpha` / `NestedRidgeOperator` / `_ridge_pipe` · `_abl_reg_kw(names, weighting=None)` · `ablate(reg_block_sizes=…, block_weighting=…)` · `abl_vs_ref(…, pcols=…)` · `abl_alpha_edge_report` · `df_abl_bw` (tier 4E); ablation rows carry `block_weighting`, `n_alpha_edge_val/aro`, `n_outer_folds`.
- **§2.6 audit (temporary):** `RUN_MANUAL_AUDIT`, `AUDIT_EXCERPTS`, everything else `_aud_*`; outputs only in `OUTPUT_DIR/manual_audit/`.
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
- **Phase B3 constants / names:** `REART_ENABLED=False` (HELPERS:descriptors; the re-articulation constants below are report-only), `segment_notes(_detail)(…, reart_enabled=None)`, `vibrato_window_reason`, the window `edge` / `reason` fields, `CREPE_DIAG` `n_edge_windows` / `n_edge_only`, `CREPE_NOTE_COUNTS`, `CREPE_NOTECOUNT_FRIEDMAN`, `CREPE_REART_REPORT` (§2.2); §4.4e `MG_GAP_MARGIN=0.10`, `MG_CURVE_RISE=0.05`, `MG_NN_FOLD=3`, `MG_N_PERM=10000`, `MG_RANDOM_SEED=0`, `MG_X_INTERP`, `MG_VERDICT`, `df_mg_gap` / `df_mg_curve` / `df_mg_nn`; §5.6 `df_abl_commonality`; §5.6b `RUN_PAIR_ABLATION_LOO=False`, `OV_I_TOL=0.02`, `OV_PROBE_MIN=0.30`, `OV_RHO_MIN=0.40`, `df_ov_pairs` / `df_ov_probe(_fam)` / `df_ov_corr(_fam)`; tier `3-subfamily-pairs`.
- **Phase B constants (§0.3 / §2.2 / §2.3 / §2.3c / §4.0):** `NC_N_SPLITS=200`, `SELECT_C_GRID=logspace(-3,2,6)` → 13 `SELECT_CONFIGS`, `MERT_REG_REPR='full'`, `SESSION_MAP_CSV`, `OLD_CACHE_NAMES`; CREPE descriptors (**Phase B2**): `OCT_WIN=10`, `OCT_DEV_RANGE=(1000,1400)`, `TREND_WIN=15`, `PLATEAU_TOL_CENTS=40`, `PLATEAU_MAX_RANGE_CENTS=80` (B3: also re-checked after a merge), `PLATEAU_MERGE_CENTS=50`, `PLATEAU_MERGE_GAP_MS=50`, `REART_DIP_DB=6` (sensitivity 3/6/9; detector disabled since B3), `REART_PRE_MS=150`, `REART_POST_MS=60`, `ENV_WIN_MS=25` (rectangular), `NOTE_MIN_MS_TIMING=60`, `VIB_MIN_NOTE_MS=500` (sensitivity 500/650/800), `VIB_WIN_MS=500`, `VIB_HOP_MS=250`, `VIB_RATE_BAND=(4,8)`, `VIB_RATE_STEP=0.05`, `VIB_FIT_R2_MIN=0.50`, `VIB_MIN_EXTENT=3.0`, `CREPE_VITERBI_MAX_CENTS_PER_FRAME=220`, `GLIDE_DUR_MS=(80,500)`, `GLIDE_MIN_EXCESS_MS=50`, `GLIDE_INTERVAL=(80,2000)`, `GLIDE_SIGN_MIN=0.8`, `GLIDE_RISE_SCALE=1.25`; legacy / report only: `BOUNDARY_LAG=5`, `BOUNDARY_CENTS=70` (also the close-split threshold), `VIB_BAND=(3,10)`, `VIB_NFFT=4096`, `VIB_PEAK_HALFWIDTH=0.5`, `VIB_PRESENCE_MIN=0.30`, `VIB_LEGACY_MIN_EXTENT=5`; Essentia `DR_ABS_GATE_DB=-60`, `DR_REL_GATE_DB=40`; §2.3c `FLAT_MIN=0.2` (0.1/0.2/0.3), `FLAT_BAND_HZ=(50,8000)`, `FLAT_RMS_PCTL=20`, `FLAT_MIN_FRAMES=20`, `SESSION_CONFINED_FRAC=0.80`; §4.0 `NMS_N_JOBS=os.cpu_count()`, `NMS_TIE_TOL=1e-12`.
- **Caches (Phase B `_v2` + Phase B2 `_v3` + Phase B3 `_v4`):** `emb_mert_v2.npz`, `crepe_tracks_v2.npz`, `rms10_v3.npz`, `feat_crepe_v4.csv`, `essentia_features_v2.csv/.npz`, `rms_frames_v2.npz`, `dtw_features_v4.csv`; §2.5 asserts no `CACHE_*` path carries a name in `OLD_CACHE_NAMES` (which now includes `feat_crepe_v2.csv`, `dtw_features_v2.csv`, `feat_crepe_v3.csv`, `dtw_features_v3.csv`) and that the v4 names are in use. §4.4e writes to `OUTPUT_DIR/mert_diagnostics/`.
- **Phase A constants:** `SEED=42`, `N_DUMMY_REPEATS=200`, `WP_N_PERM=500` (§0.3); `PRIMARY_PROTOCOL='piece'`, `PROTOCOLS=('piece','loo')`, `RIDGE_ALPHAS=logspace(-2,4,13)`, inner `GroupKFold(5)`, `DEFAULT_CLF_NAME='NestedSelect'` since Phase B (`NB_CLF_NAME='NaiveBayes'` = the Phase A default) (§4.0); `TUNE_GRID_C={'C': logspace(-3,2,6)}` (§4.1); probe `ALPHAS=[1,10,100,1000,10000]` (§6.1b); MERT cache `emb_mert_v2.npz`; `OUTPUT_DIR='/kaggle/working/pipeline_outputs'`.
- **Result containers:** `all_recs` (with `protocol`; NestedSelect rows carry `selected` / `chosen`), `DEFAULT_RECS[(set, protocol)]` (NestedSelect), `NB_RECS[(set, protocol)]`, `REG_RES[protocol]` (`reg_res` = primary; incl. `REG_MERTFULL`), `df_wp_ceiling` / `WP_CEILING`, ablation rows gain `selected` / `n_configs_selected` / `n_features_reg`, `PRED_QUADS`, `INTERP_CV_R[protocol][dim]`, `INTERP_CV_F1[protocol]`, `COE[protocol]`, `BLOCK_F1[protocol]`, `df_within_piece`, `df_delta_tests`, `df_reclevel`, `df_inner_skips`; ablation rows gain `r2_val/r2_aro/rmse_val/rmse_aro/beats_chance/classifier`.

---

## 8. Workflow

**Repo:** `makris2003/violin-emotion-thesis` (private). GitHub = source of truth; Kaggle = GPU execution. `CLAUDE.md` at repo root is auto-read by Claude Code. Loop: edit locally (VS Code) → commit/push → Kaggle pulls → run on GPU → commit results back. Claude Code **cannot run** the notebook (no audio/GPU) → surgical edits + static checks + local synthetic tests only (the §4.0 `# >>> HELPERS:cv_engine` block is self-contained for that purpose). The notebook is stored as single-line JSON, so review changes cell by cell (e.g. nbdime) rather than with a plain `git diff`. At ≈ 6 MB it exceeds the Read tool's notebook limit, so Claude Code cannot use NotebookEdit on it: cells are edited by splitting the JSON at the `{"cell_type"` boundaries, decoding / re-encoding the changed cell sources with sed/awk under `LC_ALL=C`, clearing the outputs of changed code cells, and verifying byte for byte that every untouched cell is unchanged. Kaggle "Pull from GitHub" can stale its OAuth token (unlink/relink to fix).

---

## 9. Quick answers

- **"Why is valence weak?"** Intrinsic to MER (needs harmony/context). Arousal is clean with the same features → it's a finding. To *explain* (not rescue) it, add 1–2 harmonic features (mode / tonal tension) so you can say "valence is governed by harmony, which the graded expressivity holds fixed."
- **"Why block weighting in the regression heads?"** With one ridge α on StandardScaler-ed columns, 768 MERT-full columns outweigh a ~10-column interpretable block: every MERT + X set scored like MERT alone (R²_aro 0.45–0.46, N = 118) while Essentia + DTW alone reached 0.67. Scaling block *b* by √(P/(B·p_b)) gives each tool an equal a-priori share of the penalty (group ridge), keeps the total variance P, and uses only column counts — so it cannot leak. Tier 4E reports weighted vs unweighted.
- **"Did normalization change results?"** Not materially — raw↔norm r = 0.996 (V) / 0.993 (A) at N = 118 (0.997 / 0.993 at N = 116); 10/60 excerpts change class at N = 118, 8/60 at N = 116 (mostly across the Neutral radius), 4/60 change quadrant at N = 118. That's the point: it confirms the findings aren't a rater artifact (defensibility, not correction).
- **"Why remove MFCC?"** Degenerate (F1≈0), indistinct from the Dummy floor, out-of-topic. Dummy floors are the official baseline.
- **"Is it a leaderboard (MERT vs CREPE)?"** No — that framing is de-emphasised. It's a categorisation + interpretable-attribution study; cost of explainability is the comparison, not "who wins."
- **"Why forward features up front unsupervised?"** To keep the CV leakage-free — a supervised selection would leak held-out folds. The gate never sees `Y`.
- **"Why leave-one-piece-out as the primary protocol?"** 95% (V) / 91% (A) of the excerpt variance is between pieces; LOOCV keeps a held-out excerpt's two sibling renditions in training, so it mostly measures piece recognition.
- **"Why nested model selection as the default (and not GaussianNB any more)?"** Phase A pre-declared GaussianNB so it could not be a post-hoc pick; under leave-one-piece-out it fell below chance on every feature set. Adopting the best fixed classifier from the 4D table afterwards would be a post-hoc pick on the same CV, so Phase B moves the choice *inside* every outer training fold (inner GroupKFold over the training pieces): the reported score is then an unbiased estimate of the whole select-then-fit procedure. NB stays visible as "pre-declared Phase A default — failed under piece".

---

## 10. Audit (Oct 2026)

A full audit of the 2026-08-16 run's outputs produced three phases of corrections.

- **Phase A — pipeline corrections (APPLIED and run on 2026-10-09; results in §4; the A3 MERT issue turned out to be a false alarm).** A-shared §4.0 CV engine · **A1** DummyClassifier floors were re-created with one seed in every fold, so `stratified`/`uniform` predicted the same constant everywhere (F1-macro = 0); now seeded per fold × label, 200 repeats, mean/SD/P95 · **A2** leave-one-piece-out PRIMARY everywhere, nested piece-grouped ridge instead of RidgeCV, §5.5 handles a negative CoE, new §4.9 within-piece prediction · **A3** MERT's positional-convolution weight-norm tensors were never loaded (checkpoint `weight_g/_v` vs model `parametrizations.weight.original0/1`) — the conv ran with random weights; now restored + asserted, cache `emb_mert_v2.npz`; **every MERT-based number from before is superseded** · **A4** GaussianNB as the single pre-declared default; every best-of selection removed (§4.5 headline, §4.6 best overall, §4.8, §7.1) · **A5** class balancing + per-label inner-CV C for SVM/LogReg, balanced RF/ET, XGBoost `scale_pos_weight`; tier `4D-balancing` · **A6** out-of-sample R² and RMSE next to every r, R² permutation p in the ablation BH family · **A7** MERT probe with GroupKFold(5) over pieces and nested α (grid to 1e4) · **A14** BH-FDR over all §5.1 tests (174 in Phase A with 29 interpretable features; 180 since Phase B), stars from q · **A-QC** §2.3c recording-level sanity check (report only).
- **Key fact behind A2:** 95% of valence variance and 91% of arousal variance across the 60 excerpts is between-piece; the condition explains 1% (V) / 4% (A).
- **Phase B — descriptor repairs + analysis upgrades (APPLIED 2026-10-10; RUN on N = 118 — results in §4).** Analysis upgrades: **B-NMS** nested model selection replaces the single default classifier · **B-INC** Freedman–Lane incremental test in §4.9 (ΔR² = R²(Condition + Features) − R²(Condition-only); reduced model = OLS on the centred condition one-hot; residuals permuted within piece; closed-form refits; BH over the 8 tests; columns `r2_cond_plus`, `d_r2_incremental`, `p_incremental`, `q_incremental`) · **B-NC** split-half noise ceiling (200 random rater halves, centred within piece, Spearman–Brown; uncentred for context; `r2_over_ceiling`) · **B-MERTFULL** regression heads on MERT-full (`ablate(X_reg=…)`, regression rebuild assert, §4.2 `Ridge-nested · MERT-full` row; `4A-mert-pca` unchanged). Descriptor repairs: **B-shared** CREPE tracks cached, `fix_octave_errors`, `segment_notes` (never across unvoiced gaps) · **B8** vibrato · **B9** portamento (+ `4A-crepe-portamento` removed) · **B10** F0 range P95 − P5 on the corrected track · **B11** L1-normalised flux · **B12** dynamic-range silence gate + 3 × 3 sweep · **B13** `timing_*` family (note onsets from CREPE, 151-frame tempogram × log-normal prior, `timing_tempo_logsd`), `dtw_global_tempo_ratio` → `dtw_duration_ratio` · **B-NF** spectral-flatness noise floor + FLAT_MIN sensitivity + optional session map · **B-cache** `_v2` names + assert.
- **Phase B2 — descriptor repairs after the manual audit (APPLIED 2026-11-05; run in full on N = 116)** — below.
- **Phase B3 — re-articulation detector disabled + MERT / family-overlap diagnostics (APPLIED 2026-11-10; NOT RUN)** — below.
- **Phase C (deferred idea):** augmentation (see §5 item 11).

### Phase B2 — descriptor repairs after the manual audit (2026-11-05)

**Evidence** — the §2.6 outputs of the 2026-11-03 N = 116 run (Méditation and Mozart VC4 × MEC / EXP / EXG), checked by eye and ear by the author:
- **(a) Portamento false alarms = crepe's Viterbi ramp.** Méditation MEC D6→D5 (6.57 s, −1177 c) and A5→A4 (9.36 s, −1198 c) were detected as glides (rise 62.5 ms, sign 1.0). crepe's Viterbi decoder (`crepe/core.py`, `to_viterbi_cents`: transition weight max(12 − |Δbin|, 0), 20-cent bins) moves at most 11 bins = **220 c per 10 ms frame**, so every large jump is decoded as a ramp (an octave ≈ 55 ms) — the 40 ms minimum glide duration could not separate it. Real glides in EXG last 175–312 ms (B5→C♯6 312.5, C♯6→D6 175, E5→A5 200).
- **(b) The 70 c boundary rule merges notes with their slides.** Méditation EXP "D6" 6.38–8.30 s (1920 ms, dev −12.5 c, FFT presence 0.08) is C♯6 + slide + D6; EXG C♯6 5.61–6.64 s contains slides.
- **(c) The FFT presence test rejects real vibrato.** Méditation EXP D5 8.40–10.00 s (presence 0.533, 5.58 Hz, ±4.15 c) and B4 12.73–16.12 s (presence 0.841, 5.63 Hz, ±4.40 c) were rejected only by the 5 c rule; at 0.5–2 s the Hann main lobe (±2/T Hz) is wider than the ±0.5 Hz presence window; the unclamped parabolic peak produced out-of-band rates (1.53 / 2.55 / 2.38 Hz). The author confirmed no vibrato in SQ8 exrpt1 (MEC/EXP/EXG) and exrpt3 (EXP/EXG).
- **(d) Repeated notes are merged.** Mozart VC4: 19 notes in the score, the pipeline found 17 (MEC) / 16 (EXP) / 19 (EXG); repeated D6s merged (MEC 1.39–3.09 s, EXP 1.38–3.43 s, EXG 1.70–3.04 s).
- **(e) Open, not changed:** LUFS offset +3.055 LU (all 6) · `es.Key` G minor for Mozart_VC4_EXP · Méditation tempo log2-SD MEC 0.256 > EXP 0.158 > EXG 0.135, pulse clarity lowest at MEC (✗) · Mozart LAT not sharper in spiccato EXP (✗) — see §5.

**New definitions** (`# >>> HELPERS:descriptors`, §2.2; numpy / scipy / bisect only):
- **B2 segmentation** — `segment_notes(cents, voiced, min_note_ms, step_ms=10, env_db=None, reart_dip_db=6)` (same return contract; `segment_notes_detail` also returns how each note starts — unvoiced gap / plateau / re-articulation split — and the split frames). Per voiced run: greedy plateaus of the 15-frame median trend (each frame within ±40 c of the plateau's running median, plateau range ≤ 80 c — a guard added so a slow ramp cannot pass as a plateau), candidates ≥ 60 ms; adjacent plateaus < 50 c and ≤ 50 ms apart merged; each note split at a local minimum t of the 10 ms envelope with max(env[t−150 ms : t]) − env[t] ≥ 6 dB and max(env[t : t+60 ms]) − env[t] ≥ 6 dB (candidates accepted deepest first, every part ≥ 60 ms); notes < min_note_ms dropped. Envelope `env_db_on_grid`: RMS over a rectangular 25 ms window centred on each CREPE frame time i·10 ms, on the same 16 kHz mono audio CREPE reads (zeros outside the signal, floor −120 dB), cached in `rms10_v3.npz`. The Phase B rule is kept as `_segment_notes_v2` (report only). A re-articulation split is a note onset for the timing family.
- **B9b portamento** — transitions between consecutive notes of one run; t_min = 0.8·|I| / 220 · 10 ms × 1.25 (the shortest 10–90 % rise a pure jump produces through the Viterbi decoder); excess = rise − t_min; glide iff rise ∈ 80–500 ms, excess ≥ 50 ms, |I| ∈ 80–2000 c, sign ≥ 0.80. Verdicts: across an unvoiced gap · interval out of range · no 90 % crossing · too short · viterbi-limited jump · too slow · not monotone · glide. §2.2 checks the installed crepe source (`"12 - abs"` in `to_viterbi_cents`; prints, never raises).
- **B8b vibrato** — notes ≥ 500 ms (sensitivity 500 / 650 / 800), 500 ms windows every 250 ms (last window shifted to the note end); degree-2 detrend; least-squares a·sin + b·cos over 4–8 Hz in 0.05 Hz steps; R²; rate = argmax with a parabolic refinement clamped to the grid; half-extent = √(a² + b²). Vibrato window iff R² ≥ 0.50 and half-extent ≥ 3 c. Coverage = union of vibrato-window time / analysed-note time; rate / extent = duration-weighted medians over vibrato windows; the declared substitutions and the ⚠️ are unchanged; column names unchanged. The FFT detector stays as the legacy `vibrato_note` (report only, §2.6).
- **Calibration logic (report only, label-free).** MEC is a negative control: by instruction it is played with no vibrato and no portamento, so every vibrato window / glide detected in a MEC excerpt is a false positive. §2.2 prints the false-positive rate of the 20 MEC excerpts' windows over R² ∈ {0.3, 0.4, 0.5, 0.6, 0.7} × half-extent ∈ {2, 3, 4, 5} c and of their transitions over `GLIDE_MIN_EXCESS_MS` ∈ {25, 50, 75, 100} ms (the other glide rules as deployed), ⚠️ if the deployed cell exceeds 5 %. Audio-derived detections only — never a rating, never `Y` — and nothing is selected automatically: the constants stay as declared.
- **Caches:** `feat_crepe_v2.csv` → `feat_crepe_v3.csv`, `dtw_features_v2.csv` → `dtw_features_v3.csv` (the onsets changed), new `rms10_v3.npz`; both v2 names in `OLD_CACHE_NAMES`, asserted by §2.5.

| Phase B | Phase B2 |
|---|---|
| boundary where the trend moves > 70 c over 5 frames | plateaus (±40 c, range ≤ 80 c) + merge < 50 c / ≤ 50 ms + re-articulation dips ≥ 6 dB |
| glide: 10–90 % rise × 1.25 of 40–400 ms, sign ≥ 0.8 | 80–500 ms, excess over the Viterbi ramp ≥ 50 ms, sign ≥ 0.8 |
| vibrato: notes ≥ 800 ms, FFT peak in [3, 10] Hz, presence ≥ 0.30, Hilbert half-extent ≥ 5 c | notes ≥ 500 ms, 500 ms windows / 250 ms hop, LS sinusoid 4–8 Hz, R² ≥ 0.50, half-extent ≥ 3 c |
| coverage = vibrato-note duration / analysed-note duration | vibrato-window time (union) / analysed-note time |
| rate / extent: duration-weighted medians over vibrato notes | over vibrato windows (rate clamped to 4–8 Hz) |
| `feat_crepe_v2.csv`, `dtw_features_v2.csv` | `feat_crepe_v3.csv`, `dtw_features_v3.csv`, + `rms10_v3.npz` |

### Phase B3 — re-articulation detector disabled + MERT / family-overlap diagnostics (2026-11-10; APPLIED, NOT RUN)

**Evidence — the full B2 run (N = 116, block weighting; numbers quoted by the author, outputs not committed):**
- **(a) Re-articulation splits follow vibrato, not bow changes.** The B2 detector split a plateau where the 10 ms RMS envelope dipped ≥ 6 dB. On the violin, vibrato also modulates the amplitude (the frequency sweeps through the body resonances); the 25 ms envelope resolves that modulation and the 6 dB rule took it for a bow re-attack. Splits per condition **MEC 24 / EXP 51 / EXG 68** for the same scores; Meditation_Thais 0 / 3 / 11 — the EXG F♯5 cut into 7 pieces of 140–370 ms (1–2 vibrato periods); Shostakovich_Sym5_Mvt3_EXG 11 splits. Note totals per condition: Phase B boundary rule 575 / 533 / 518 → B2 plateaus only 616 / 596 / 592 → B2 with splits 640 / 647 / 660. `REART_DIP_DB` sensitivity, Mozart_VC4 MEC / EXP / EXG: 26/23/28 (3 dB), 22/21/23 (6 dB), 19/17/22 (9 dB) vs 19 in the score — no threshold fixes all three. Consequences: §5.1b note-based timing turned significant (note rate z = +2.06, p = 0.040; IOI mean −2.53, p = 0.011; IOI CV +2.85, p = 0.004 — all n.s. in Phase B); §5.6 "Timing (ref-free)" alone F1 (piece) 0.417 → 0.542, its removal loss −0.012 → +0.110 (q = 0.037); long EXG vibrato notes fragmented below the 500 ms analysis minimum; the portamento share counted repeated-note pairs in its denominator.
- **(b) The merge broke the plateau range invariant.** Adjacent plateaus (< 50 c apart, ≤ 50 ms gap) were merged without re-checking the 80 c range: Meditation_Thais_EXG C♯6 (≈ 5.76–6.62 s) had a trend range of 128 c (V3 FAIL); merged notes also absorbed the transition frames between the plateaus.
- **(c) Vibrato rate pinned at the band edge.** At least one excerpt had a weighted-median vibrato rate of exactly 4.00 Hz, the lower edge of the 4–8 Hz grid: the best fit lay outside the band — a residual slow trend, not vibrato.
- **(d) MERT generalisation (§4.4b / §4.4c / §5.5 / §4.5b).** F1 LOOCV → piece: MERT 0.623 → 0.310 (−50 %, below the chance P95 0.323) vs CREPE 0.500 → 0.333, Essentia 0.480 → 0.356, DTW 0.570 → 0.542; adding MERT to an interpretable block lowered the piece F1 in 6 of 7 combinations and raised the LOOCV F1 in 7 of 7; unique F1 of MERT (piece) −0.122 (q = 0.034); NestedSelect chose C = 0.001 (almost constant) in 12/20 piece folds and C = 10 in 43/60 LOOCV folds. MERT-PCA25 ridge: negative out-of-sample R² under piece (valence −0.18, arousal −0.25); MERT-full ridge transfers arousal (R² 0.45) but not valence (0.01); the valence α of the MERT combos hit the upper grid edge. The MERT PCA is fit once on all 60 excerpts (label-free). No train score was computed anywhere → capacity overfitting vs a piece-identity shortcut (LOOCV keeps the other renditions of the same piece in training) could not be told apart.
- **(e) Family overlap (§5.6, piece).** Full interpretable layer (30 features, 8 families) F1 0.449, R²a 0.627; Σ family-alone R²a ≈ 1.55, Σ removal losses ≈ 0.23 → most of the arousal R² is shared. The §5.6 F1 figure drew "alone" bars from 0 although the chance mean is 0.272 (P95 0.323). "Small removal loss + high alone score" can be redundancy, noise (only 2 of 24 removal losses q < 0.05), dilution (Timing alone 0.542 > all 30 features 0.449) or a different NestedSelect choice per subset.

**Decision (author): disable the re-articulation detector for the features; keep it as a documented, report-only negative result.** Without it, repeated same-pitch notes with no silence between count as one note. That bias is set by the **score**, so it is ≈ the same in the three renditions of a piece and cancels in within-piece comparisons; the vibrato artefact grew with expressiveness and did not cancel. Only 4 of the 7 timing features use note onsets (`timing_note_rate`, `timing_ioi_mean_s`, `timing_ioi_sd_s`, `timing_ioi_cv`); tempo variability, local tempo and pulse clarity come from the tempogram and are unaffected. Vibrato and portamento never needed the detector.

**Changes (notebook):**
- **A1** `REART_ENABLED = False` in HELPERS:descriptors (with the evidence in a comment). `segment_notes` / `segment_notes_detail` gained an optional `reart_enabled=None` (None → `REART_ENABLED`; True forces the detector on — report only); every feature path (`note_onset_times`, `timing_note_features`, `vibrato_features`, `portamento_features`, `crepe_descriptors`, `extract_timing_reffree`) keeps `env_db` in its signature (length-checked) and ignores it. `_reart_splits`, `env_db_on_grid` and the `REART_*` constants are kept.
- **A2** `_merge_plateaus(…, max_range=PLATEAU_MAX_RANGE_CENTS)`: two plateaus merge only if the merged note's trend range, gap frames included, stays ≤ 80 c.
- **A3** `vibrato_window_fit` returns `edge` (R² argmax before refinement = the first or last grid point); `vibrato_window_reason` names the verdict ('band edge' / 'low R²' / 'small extent' / 'vibrato'); a band-edge window is never vibrato; windows record `edge` and `reason`; §2.2 prints the edge rejections (all excerpts and the 20 MEC, and how many were rejected *only* by the edge rule); the calibration grid applies the rule in every cell (asserted: deployed cell = MEC vibrato-window share of the descriptors).
- **A4** §2.2 banner "re-articulation detector DISABLED (REART_ENABLED = False) — see Phase B3"; report-only evidence table (detector on at 6 dB on the cached tracks + `env_db`: totals and splits per condition, per-excerpt counts, the 3 / 6 / 9 dB sensitivity with Mozart_VC4; asserted: detector-on notes = deployed notes + splits); label-free condition-independence check of the deployed note counts (per-condition totals, per-piece range, `scipy.stats.friedmanchisquare` on the 20 × 3 table and on the detector-on counts; ✅ p ≥ 0.05 / ⚠️, printed) → `CREPE_NOTECOUNT_FRIEDMAN`.
- **A5** caches `feat_crepe_v3.csv` → **`feat_crepe_v4.csv`**, `dtw_features_v3.csv` → **`dtw_features_v4.csv`** (with the old name the cached timing file would silently keep the B2 onsets); both v3 names in `OLD_CACHE_NAMES`, §2.5 asserts it; `crepe_tracks_v2.npz`, `rms10_v3.npz`, MERT / Essentia caches unchanged.
- **A6** §2.6 on the disabled path (asserted: no deployed split), band-edge windows shown with their reason; **V4 → INFO** ("known limitation — repeated notes without a silence are counted once", expected ≈ 17 / 16 / 20; criterion unchanged); the V2 / V3 "has vibrato" checks also require the matched note to cover ≥ 80 % of the annotated span; **new V7** = the §2.2 Friedman result.
- **§4.4e — MERT generalisation diagnostics** (new, right after §4.4d — the name §4.4d was taken by the target-side ablation). B1 train vs test per block (MERT-PCA25, MERT-full, CREPE, Essentia, DTW, the 30-feature layer, a 25-column N(0,1) capacity control) and protocol: classification — the per-fold NestedSelect choice is read from §4.0's memo (`return_info`; the §4.4b rows store only counts), refitted with §4.0's scaler + binary relevance; train F1 = mean over folds, test F1 pooled (asserted = NestedSelect and = the §4.4b / §4.4c rows); regression — nested ridge, train R² of the refitted fold model vs out-of-sample R² (asserted = the §4.4b / §4.4c predictions). B2 validation curve (`piece`, SVM-Linear balanced at each fixed C of `SELECT_C_GRID`) for MERT-PCA25 / interpretable / random. B3 nearest-neighbour identity: MERT-full (cosine, raw), MERT-PCA25, CREPE, Essentia, DTW, interpretable (z-scored over the 60 excerpts) — same-piece share vs 2/59, same-condition share vs 19/59, 10 000 label shuffles (keep 3 per piece / 20 per condition), silhouettes. **B4 pre-declared rules** — CAPACITY OVERFITTING: MERT-PCA25's piece train − test F1 gap ≥ the interpretable gap + 0.10 and ≥ the random-control gap + 0.10, AND every C at which MERT's train F1 has risen (≥ +0.05 over its value at the smallest C; ≥ 1 such C) keeps the test F1 ≤ the strict P95; PIECE-IDENTITY SHORTCUT: in a MERT space, same-piece share ≥ 3 × chance with p < 0.05 and above every interpretable space, while the same-condition share has p ≥ 0.05. Both can fire; neither is said explicitly.
- **§5.6 edits (C0)** chance mean + strict P95 lines on the F1 panel; `*` on removal losses with q < 0.05; commonality summary per protocol (F1 above the chance mean, R²a, R²v: Σ alone, Σ loss, full, shared ≈ full − Σ loss, with the non-additivity caveat) → `ablation_subfamily_commonality.csv`; the pre-declared timing reading printed.
- **§5.6b — family overlap diagnostics** (new, right after §5.6). C1 tier `3-subfamily-pairs` (28 pairs + the reference row; `piece`; `RUN_PAIR_ABLATION_LOO = False` — `ablate` already accepts `protocols`, so the engine is unchanged): I(A,B) = loss(A∪B) − loss(A) − loss(B) for F1 / R²a / R²v with loss(A), loss(B) from the §5.6 tier, interpreted only when the pair loss has q < 0.05 (I > 0.02 substitutes · |I| ≤ 0.02 each its own · I < −0.02 only together), 8 × 8 heatmaps, runtime printed, `ablation_master.csv` re-exported. C2 probes (nested ridge, `NestedRidgeOperator` asserted = `nested_ridge_predict`, `piece`): each feature of A from all non-A features and from each family B, RAW and WITHIN-PIECE (feature minus its piece mean — the held-out piece's own feature mean, never a rating). C3 Spearman 30 × 30 RAW and WITHIN-PIECE ordered by family + 8 × 8 mean |ρ| (raw = repertoire; within-piece = how techniques co-vary MEC → EXG). **C4 pre-declared triangulation** — A and B overlap only if ≥ 2 of: (1) I(A,B) > 0 with the pair loss q < 0.05; (2) within-piece probe R²(A←B) or R²(B←A) ≥ 0.30; (3) within-piece family mean |ρ| ≥ 0.40; the pairs passing and the one-test pairs are printed. C5 `overlap_pairs.csv`, `overlap_probe(_family).csv`, `overlap_corr(_family).csv` + 3 figures.

**A7 — acceptance criteria for the next run (written here, not asserted):** Meditation_Thais 19 / 19 / 20 notes; per-condition totals ≈ 616 / 596 / 592 (small changes from A2 allowed); Friedman n.s. (V7 PASS); §5.1b note rate / IOI dose-response back to n.s. **PRE-DECLARED reading of the timing result:** in §5.6 compare "Timing (ref-free)" alone F1 (piece) with 0.417 (Phase B) and 0.542 (B2). If it stays near 0.54, the timing signal is real; if it returns near 0.42, the B2 value was the vibrato artefact. Record which here, and quote only the B3 value in the thesis. *(Outcome: ___ — fill in after the run.)*

| Phase B2 | Phase B3 |
|---|---|
| plateaus + merge < 50 c / ≤ 50 ms + re-articulation dips ≥ 6 dB | plateaus + merge only if the merged range ≤ 80 c; **no re-articulation split** (`REART_ENABLED = False`; detector report-only) |
| vibrato window iff R² ≥ 0.50 and half-extent ≥ 3 c | … and the best rate is not on the 4 / 8 Hz grid edge (`edge`, `reason`) |
| `feat_crepe_v3.csv`, `dtw_features_v3.csv` | `feat_crepe_v4.csv`, `dtw_features_v4.csv` |
| §2.6 V1–V6 (V4 PASS / FAIL) | V1–V7 (V4 INFO, V2 / V3 ≥ 80 % span, V7 Friedman) |
| — | §4.4e MERT generalisation · §5.6b family overlap · §5.6 commonality / chance lines |

### What to check after the next run (N = 116 + Phase B3)

Interactive run up to §2.6:
- [ ] §2.2 banner "DISABLED"; Viterbi-source line and octave fixes unchanged.
- [ ] §2.2 evidence table: detector OFF ≈ 616 / 596 / 592 (A2 may move them a little), ON ≈ 640 / 647 / 660 with ≈ 24 / 51 / 68 splits; 3 / 6 / 9 dB Mozart_VC4 rows; asserts ✅.
- [ ] §2.2 condition-independence: totals, per-piece range, **Friedman n.s. ✅** (and the detector-on Friedman).
- [ ] §2.2 vibrato: band-edge rejections (all / MEC / edge-only); rate never exactly 4.00 / 8.00; calibration deployed cell ≤ 5 %.
- [ ] §2.2 close splits (A2 can add a few), portamento verdicts; §2.4 `dtw_features_v4.csv`, DTW forwarding 3–10; §2.5 asserts ✅.
- [ ] §2.6: asserts ✅; Meditation_Thais **19 / 19 / 20**; V3 C♯6 ≤ 80 c; V2 / V3 vibrato with ≥ 80 % span; V4 INFO ≈ 17 / 16 / 20; V6 (≈ 5 MEC — known); **V7 PASS**.
Full run:
- [ ] §5.1b note rate / IOI back to n.s.?; vibrato extent / coverage / rate z.
- [ ] §5.6 pre-declared timing reading (record above); commonality summary; figure lines / `*`.
- [ ] §4.4e: asserts ✅; gaps; validation curve; NN shares; **which rule fires**.
- [ ] §5.6b: runtime; I heatmaps; probes; correlations; **triangulation**; master table re-exported.
- [ ] All other guards ✅; total runtime (≈ the B2 run + 15–20 min).

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

### Phase B Kaggle run (N = 118, 2026-10-09/10) — checklist outcome

- [x] §2.2 octave corrections 3 frames / 3 excerpts; vibrato rate 37 distinct values (> 20); half-extent 0–32.1 c, coverage 0–1; substitutions: no note ≥ 800 ms 1/60 (no ⚠️), no vibrato note 23; notes 391 / 259 / 198 at 600 / 800 / 1000 ms; F0 range max 3100 c (old 5343 c); 17 close splits.
- [x] §2.2 portamento extents 80–1907 c; glides on 24.3 % of transitions (median 62 ms) → confirmed false alarms by the 2026-11-05 audit (Viterbi ramp) → fixed in Phase B2 (B9b).
- [x] §2.3 r(flux, LUFS) 0.874 raw → 0.508 normalised; DR max 34.7 dB, median 6.2 % gated (max 26 %); DR sensitivity ρ down to 0.79 → under audit.
- [x] §2.3c 0/60 conclusive at FLAT_MIN 0.2 (9 at 0.1), 0 flags, session map not provided → under audit.
- [x] §3.3b timing manipulation check: 0/3 variability features non-decreasing; tempo log2-SD falls MEC → EXG → author's explanation + under audit.
- [x] §4.0 4 CPUs, parallel == serial ✅.
- [x] NestedSelect beats the strict P95 under `piece` only on X_B (0.361); X_C 0.302 does not.
- [x] MERT-full in the lattice regression heads (`n_features_reg` 768–795), rebuild ✅ — and every MERT + X head scored like MERT alone (→ block weighting).
- [x] §4.9 B-INC: Δvalence incremental q 0.037 (Interpretable, MERT-full, Interp + MERT-PCA); Δarousal q 0.070; ceilings V 0.455, A 0.678.
- [x] §5.1b vibrato extent / coverage escalate (z +5.14 / +4.90); §5.6 vibrato alone weak; §6.1b vibrato coverage R² 0.48, rate −0.02 (`CREPE_UNIQUE2` → pending decision).
- [x] Runtime ≈ 97 min.

### B2 checklist (N = 116 + block weighting + Phase B2 + §2.6 audit) — run in full by the author; kept for reference (superseded by the B3 checklist above)

Interactive run up to §2.6 first (the §1 items already held in the 2026-11-03 run):
- [ ] §1.1 prints 2 excluded (`questionnaire_5_P012`, `questionnaire_5_P020`), 118 → 116 participants, 2360 → 2320 rows, per form 20 / 21 / 26 / 19 / 30; the three asserts pass.
- [ ] §1.1b maps ≈ V 2.94 + 1.57z, A 3.39 + 1.53z; §1.2b anchors V 0.342 → 0.338, A 0.276 → 0.253; 8/60 class changes; §1.2 classes HA_LV 20, LA_LV 17, HA_HV 11, Neutral 7, LA_HV 5 (`Schindlers_List_EXP` HA_LV → LA_LV); §1.4 prints the exclusions.
- [ ] §2.2 (B2): Viterbi-source line ("12 - abs" found); `rms10_v3.npz` computed (length assert ✅); segmentation old vs new counts (total / per condition / per excerpt), re-articulation splits, `REART_DIP_DB` 3 / 6 / 9 sensitivity incl. Mozart VC4 (score 19 each), close splits; vibrato analysed notes at 500 / 650 / 800 ms, windows / vibrato windows, substitution counts (⚠️ > 10 with no note ≥ 500 ms), rate inside 4–8 Hz; portamento verdict counts (expect many "viterbi-limited jump"), glide rise / excess / extent; **calibration grids: deployed false-positive rates ≤ 5 %? (⚠️ otherwise — discuss the constants before the full run; nothing is auto-tuned).**
- [ ] §2.4: timing recomputed with the B2 onsets (`dtw_features_v3.csv`); DTW forwarding count 3–10.
- [ ] §2.6: every recompute / detail-consistency assert passes (windows → coverage / rate / extent, window fit → R², transitions → portamento); declared substitutions listed; LUFS diff − 3 dB ≈ 0; listen to the players (onset clicks, beat clicks, glide tone); read the overview (vibrato windows, re-articulation splits, env_db) / zoom figures (fitted sinusoids), the per-piece ✓/✗ tables and the 20-piece exceptions; **AUTHOR VALIDATION V1–V6 all PASS** (+ the F♯5→A5 / A4→B4 old vs new verdicts); fill `manual_audit/audit_checklist.csv`.
Full run:
- [ ] §4.0 class prevalences 11 / 20 / 5 / 17 / 7 + analytic uniform F1; block-weighting guard ✅; printed weights (MERT-full ≈ 0.51, CREPE / Essentia ≈ 4.46, DTW ≈ 5.33).
- [ ] §4.4b `[w]` on the 7 MERT-full multi-block rows; MERT + X R²_aro no longer equal to MERT alone?; α grid-edge report.
- [ ] §4.4c tier 4E: Δ weighted − unweighted (r, R², q) per set and protocol; deployed == lattice ✅; α edge hits.
- [ ] §4.4d radius rule; all other guards; NestedSelect vs strict P95; §4.9 B-INC; §5.5 CoE; §5.6; §6.1b; total runtime.
- [ ] **§5.1b dose-response re-check for the vibrato family** (the abstract claims vibrato extent / coverage escalate MEC → EXP → EXG — Phase B z +5.14 / +4.90 with the faulty detector); §5.6 Vibrato / Portamento rows (Phase B: removing Portamento raised F1); §6.1b vibrato / portamento probe and `CREPE_UNIQUE2`.

### Phase A Kaggle run (2026-10-09) — checklist outcome

- [x] §2.1 restore message printed; the embeddings turned out identical to the pre-fix run (false alarm).
- [x] §2.3c: 0 LUFS outliers, 0 clipping, 21 noise-floor flags (15 inconclusive) — estimator replaced in Phase B (B-NF).
- [x] §4.0 equivalence assert passed; Dummy floors vary across folds (uniform 0.272 / stratified 0.183 under loo).
- [x] §4.5b: no default-classifier row beats the strict P95 under `piece` (NB below chance everywhere).
- [x] §4.4b lattice, §4.4d label-reproduction, §6.2b guards passed.
- [x] Tier 4D: balanced SVM + inner-CV C beats NB under `piece` on all three matrices.
- [x] §4.9: Interpretable beats Condition-only descriptively for both targets (untested → B-INC).
- [x] §5.5: CoE negative under `piece` (interpretable wins).
- [x] §6.1b: α hit the grid edge (count superseded: 34 outer folds in the Phase B run).
- [x] Runtime: 45.3 → 75.5 min (+30 min), dominated by §4.1 inner-CV tuning and the 4D tier.

---

## 11. Listening study — final facts (N = 116)

Everything below is written up in the thesis Chapter 7 (Dataset). Numbers at N = 116 were reproduced offline from the five CSVs with the notebook's own §1.1–§1.2b logic (the reproduction matches the notebook exactly at N = 118).

**Stimuli and recording.** 20 passages × 3 renditions, all played by the author: concerto, solo, orchestral/chamber parts played as a single line, and film themes. Pieces were chosen so that their EXP renditions would span the four quadrants (intended HA_HV 4, LA_HV 5, LA_LV 6, HA_LV 5 — from the author's metadata file, which is not part of the thesis). Composers (Table 7.1): Bach (BWV 1001 Adagio), Bruch (VC1), Morricone (Cinema Paradiso), R. Strauss (Don Juan), Rota (Godfather), Piovani (La vita è bella), Massenet (Méditation), Mendelssohn (VC E minor, II), Mozart (VC3 K. 216, VC4 K. 218), J. Williams (Schindler's List), R. Løvland (Song from a Secret Garden), Shostakovich (SQ8 ×3 passages, Sym. 5 I and III, VC1), **E. Tanzil (Serenade)**, Vitali (Chaconne, attr.). Home studio, commercial laptop, free DAW, **MOTU M4** interface, **AKG P120** (condenser) and **Shure SM57** (dynamic), 3 sessions. Exported mono WAV, 44.1 kHz, 16-bit PCM; excerpts ≈ 5–30 s. **Not loudness-normalized** (RMS −39.4 to −19.1 dBFS, peak ≤ −0.8 dBFS).

**Performer instructions.** MEC: détaché, just the notes, one moderate dynamic, no vibrato, metronomic. EXP: normal interpretation — the composer's dynamics in full (pp–ff), normal vibrato, a little rubato and portamento, expressive but not excessive; colour per piece (e.g. spiccato in Mozart, flautando in Méditation / Secret Garden, sul G in Godfather). EXG: EXP taken to excess — extreme dynamics (mostly louder), excessively wide vibrato, extreme rubato and portamento, often heavy bow, glissandi, added accents.

**Questionnaire.** 5 Google Forms; one link → Google Apps Script web app that redirects each visit to one of the 5 forms uniformly at random (no balancing, no log). Form sizes before screening 20/21/26/19/32, χ²(4) = 4.97, p = .29. Intro: purpose, ~15 min, anonymous, headphones + computer recommended. Four background items (training, listening frequency, age group, gender), then the 20 pieces **in the same alphabetical order in every form**. Per excerpt, three mandatory items (perceived framing): (1) "Which emotions do you think best describe this performance?" — 1–3 of Joyful activation, Power, Tension, Sadness, Tenderness, Peacefulness, Nostalgia, Neutral, **option order randomised per question** (all responses comply: 769 one tag, 987 two, 564 three at N = 116); (2) valence "How would you rate the mood of the piece?" 0 = Δυσάρεστη (unpleasant) … 6 = Ευχάριστη (pleasant); (3) arousal "What is the arousal level of the performance?" 0 = Απόλυτη Γαλήνη (complete calm) … 6 = Ένταση (intensity). Midpoint 3 unlabelled, treated as neutral. Limitations kept in the text: arousal anchors share words with the Peacefulness and Tension tags; the intro described the scales partly in felt terms. The form's typo in the arousal item is **not mentioned anywhere** (author's decision).

| Group | Pieces | F1 | F2 | F3 | F4 | F5 |
|---|---|---|---|---|---|---|
| G1 | Mozart_VC4, Schindlers_List, SQ8_exrpt2, Sym5_Mvt1 | MEC | MEC | EXG | EXP | EXP |
| G2 | Bach_Adagio_S1, Godfather, La_Vita_E_Bella, Mendelssohn_VC_Mvt2 | MEC | EXG | EXP | EXP | MEC |
| G3 | Bruch_VC1, Don_Juan, Mozart_VC3, SQ8_exrpt3 | EXG | EXP | EXP | MEC | MEC |
| G4 | Cinema_Paradiso, Secret_Garden, SQ8_exrpt1, Tanzil_Serenade | EXP | EXP | MEC | MEC | EXG |
| G5 | Meditation_Thais, Sym5_Mvt3, Shostakovich_VC1, Vitali_Chaconne | EXP | MEC | MEC | EXG | EXP |

**Participants (N = 116; 20/21/26/19/30 per form).** Collection 3 Apr – 7 Jun 2026. Training: none 26 (22.4%), amateur 12 (10.3%), conservatory 57 (49.1%), professional 21 (18.1%). Listening to classical/orchestral: never 7, rarely 22, monthly 25, weekly 27, daily 35. Age: under 18: 6 (none excluded), 18–24: 86, 25–34: 16, 35+: 8 (the thesis shows "Under 18" as its own row — the author asked to fold them into 18–24; that was not done because it would misstate their age). Gender: woman 57, man 56, other 3. Balanced across forms (training χ²(12) = 15.1, p = .24; frequency p = .59; age χ²(12) = 14.0, p = .30; gender p = .22). 2,320 ratings: 928 MEC, 928 EXP, 464 EXG; raters per excerpt MEC/EXP 41–50 (median 47), EXG 19–30 (median 21).

**Screening v3.** Flags at N = 118: D1 5, D2 5 (simulated cut-off 14.55; χ² cut-off 8.57 would flag 29), D3 2, D4 1, D5 0 → 2 excluded (see §4 Cleaning).

**Excerpt QC (N = 116).** ICC(1,k) V 0.965 / A 0.953; r_wg median 0.60 / 0.59 (min 0.13 / 0.28). Dip test unavailable → Sarle's bimodality coefficient: > 5/9 for 2 valence and 4 arousal excerpts, 5 of them ceiling-skewed; only `Secret_Garden_EXG` arousal looks split. Manipulation check: mean arousal MEC 3.18 < EXP 3.47 < EXG 3.68; EXG > MEC in 17/20 pieces; Kruskal–Wallis p < .05 in 8/20; strictly monotone in 9/20; inversions Bach (−0.52), Mendelssohn (−0.18), La Vita (−0.09).

**Normalization (N = 116).** σ-floor never engaged. Maps V = 2.94 + 1.57z, A = 3.39 + 1.53z (raw grand mean 2.940 / 3.430; between-excerpt SD 1.082 / 0.965). Anchor |Δ| (40 anchors) V 0.342 → 0.338, A 0.276 → 0.253. Within-excerpt SD V 1.27 → 1.16, A 1.26 → 1.18. Raw ↔ normalized Pearson r 0.997 / 0.993. 8/60 classes change raw → normalized. Crossed additive (rater + excerpt) model: Spearman 0.993 / 0.991, same class for 53/60. (At N = 118, for comparison: V = 2.95 + 1.58z, A = 3.39 + 1.52z; anchors V 0.334 → 0.320, A 0.289 → 0.267; 10/60 class changes.)

**Ground truth (N = 116).** HA_LV 20, LA_LV 17, HA_HV 11, Neutral 7, LA_HV 5 (N = 118: HA_LV 21, LA_LV 16). By condition: HA_HV 3/5/3, HA_LV 7/6/7, LA_LV 6/6/5, LA_HV 2/2/1, Neutral 2/1/4 (MEC/EXP/EXG). 15/20 pieces keep one class in all three renditions; 4 of the 7 Neutral excerpts are EXG. Only 10/20 EXP renditions land in their intended quadrant (5 differ in valence, 4 in arousal, 1 Neutral). Tag check: the modal tag class equals the VA class for 42/53 non-neutral excerpts; 8 of the 11 misses are HA_LV excerpts tagged mostly Sadness/Nostalgia; Neutral is never the modal tag. Tags: Tension 18.5%, Sadness 18.4%, Nostalgia 15.2%, Power 12.3%, Joyful activation 10.8%, Tenderness 10.1%, Neutral 7.3%, Peacefulness 7.3% (mean 1.9 tags per response).

**Supporting scripts (qc_pack, outside the repo).** `questionnaire_cleaning_v3.py` (`--csv-dir --out`), `demographics.py` (`--csv-dir --out --exclude`), `excerpt_durations.py` (`--audio-dir --out`; the author ran it on Kaggle), `work/gt.py` + `gt2–4.py` (offline reproduction of §1.1–§1.2b and the QC numbers above).

---

## 12. Diploma thesis (LaTeX, NTUA template) — writing status 2026-10-30

**Build.** `main.tex` (book class, `openany`, biblatex/biber, babel greek/english, TikZ/pgfplots). 73 pages now, cap 120 including front matter; target ≈ 113. Colour palette defined once in `main.tex` (`mecC` slate / `expC` blue / `exgC` vermilion for the conditions; `qHAHV` orange, `qHALV` red, `qLALV` blue, `qLAHV` green, `qNEU` grey for the classes; `stPerf/stTech/stSig/stList` for the performer → listener chain; macros `\cmec \cexp \cexg \swatch{}`) — **every new figure uses it**. Writing rule: a heading with substantial content is a numbered `\subsection` with a `subsec:` label (no `\subsection*` in the body chapters).

| Ch. | Title | Pages | Status |
|---|---|---|---|
| — | Front matter (title pages, copyright, Περίληψη, Abstract, Ευχαριστίες, ToC/LoF/LoT, abbreviations) | 16 | done; placeholders: author name, committee, logo, dedication |
| 1 | Εκτεταμένη Περίληψη (Greek) | 1 → 12 | skeleton; §1.1–1.2 writable now, rest after the results |
| 2 | Introduction | 4 | done (chain figure, contributions reflect Phase B) |
| 3 | Expressive Music Performance | 9 | done |
| 4 | Music Emotion: Models and Measurement | 6 | done |
| 5 | Computational Tools | 8 | done |
| 6 | ML and Interpretability Preliminaries | 9 | done (nested CV, chance floors, Freedman–Lane, noise ceiling, screening incl. bimodality coefficient) |
| 7 | Dataset: Design, Collection and Ground Truth | 9 | done, N = 116 numbers (§11); TODOs: recruitment, mean durations |
| 8 | Proposed Framework | 0 → 10 | can be written now from the notebook (method, not results) |
| 9 | Experiments and Results | 0 → 20 | after the N = 116 Phase B run |
| 10 | Conclusion | 0 → 3 | last |
| — | Bibliography | 8 | 109 entries |

Planning doc with the full validation, page budget, figure/table/equation plan and checklists: "Διπλωματική: validation και πλάνο συγγραφής" (Claude Docs, sections 1–13; §13 = open items for the next chat).