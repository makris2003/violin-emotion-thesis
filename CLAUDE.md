# Thesis Handoff — Violin Expressiveness → Emotion Recognition

**Purpose:** complete standalone context to continue work in a fresh conversation. Covers (1) scope, (2) current pipeline architecture, (3) progress, (4) findings, (5) open issues, (6) technical report status, (7) conventions, (8) workflow, (9) quick answers, (10) audit, **(11) listening study — final facts (N = 116)**, **(12) diploma thesis (LaTeX) — writing status**.

> **Latest (2026-11-20, later): Phase C3 — APPLIED, NOT RUN** ("Phase C3" in §10 of CLAUDE.md / the "Phase C3" section of docs/thesis.md). Small fixes after the C2 run (which stopped its primary null at its time budget with 600 draws per width, 561 at d = 52): the null **resumes from the Kaggle input** `violin-thesis-null-cache` (no redraws; settings hash unchanged), **per-width draw targets** (1000 at d = 7 / 17, 600 at the other widths), **one empirical-null rule** in every cell (exact width, else the next larger primary width, else d = 52 extrapolated — the 100-draw map is no longer a reference), a **post-hoc revised claim-10 rule** (the timing set without the tempogram must also keep its F1 above its null and lose < 0.10 of R² arousal) and unique Fig 8 labels. 118 cells. **No result number below has changed**; the C2 outputs (commit `5c94dc2`) are not transcribed yet. **Next:** the C3 run (§5.3 ≈ 839 new draws ≈ 1.7 h).
>
> **Earlier (2026-11-20): Phase C2 — APPLIED, RUN 2026-11-20 (outputs in `5c94dc2`, not transcribed)** ("Phase C2" in §10 of CLAUDE.md / the "Phase C2" section of docs/thesis.md). The notebook was restructured so its order is the story order — **12 parts**: setup · ground truth · features · tool validation · **manipulation check (Part 4)** · **evaluation yardstick (Part 5: CV engine, Dummy floors, the empirical null — drawn before any F1)** · prediction · MERT diagnostics · within-piece · which techniques · cost of explainability · synthesis (claims table). Content: predictive (27) / descriptive (30) interpretable layer (replaces `DTW_DEV_IN_INTERP`), BH in the Page tests, an **exact-width primary null** (1000 draws per width as applied in C2; Phase C3: 1000 at d = 7 / 17, 600 elsewhere) with BH in a pre-declared family, tempogram ⚑ + sensitivity rows, deployed nested ridge in §6.2 / §6.8 / Fig 5, rewritten §7.4 / §7.5 / §9.3, §11.1 claims table; §2.6 audit, the 8-classifier Wilcoxon and §6.2b removed. 118 cells. **No result number below has changed — nothing has run**; the numbers the C2 run will replace are flagged in "Current results" and listed in the C2 run checklist. Old section numbers in the historical text are annotated "(now §x.y)". **Next:** the C2 run (the primary null takes ≈ 15 h → two sessions, resumable).
>
> **Earlier (2026-11-17): Phase B3 and C1 have been RUN — results transcribed (§4, "Current results").**
> **Classification:** against the §4.5c (now §5.3 / §6.9) empirical null (P95 0.370 at 10 columns / 0.459 at 25 vs Dummy 0.323), only the sets with the reference-free timing block beat chance (DTW 0.398, CREPE + DTW 0.460, Essentia + DTW 0.373, piece).
> **Regression** is unaffected and carries the claims (interpretable R²a 0.648 / R²v 0.258 on unseen pieces; ΔR valence −0.370, arousal −0.156).
> **MERT:** the layer study shows the piece-identity shortcut at every depth (same-piece NN 93–100 %), no layer beats the null, and the suggestion W09-11 is not applied; recommended: **keep `MERT_LAYERS = [5, 6, 7]`** (author to confirm).
> **Timing:** the pre-declared reading confirms that the B2 timing F1 0.542 was the vibrato artefact (B3 0.398).
> **Overlap (§5.6b (now §10.3)):** a common cause, not pairwise duplication → per-technique attribution is a design limit.
> **Consolidated limitations / future work / remaining code obligations** are at the end of §5; **literature hooks** in §13. **Next:** Phase C2 (§5, "Remaining code obligations").

> **Earlier (2026-11-03):** the **Phase B run on N = 118 is done** (Kaggle, Tesla T4, papermill 2026-10-09 23:45 → 2026-10-10 01:22 UTC, ≈ 97 min, 101 cells, 0 errors) — §4 now records those results, labelled **N = 118 / pre-block-weighting**. Applied in the notebook since then (now **104 cells**), not yet run: the participant exclusions in §1.1 (N = 118 → 116), **equal-share block weighting** of the multi-block MERT-full regression heads (`REG_BLOCK_WEIGHTING`, tier `4E-block-weighting`) and the **temporary §2.6 (now removed in C2) manual descriptor audit**. **Next Kaggle run:** interactively up to §2.6 (now removed in C2) first (inspect the audit), then the full run. The diploma thesis (NTUA template, LaTeX) has Chapters 1–7 written, 73 of a 120-page cap (§12).

> **Earlier (2026-11-16): Phase C1 — APPLIED (since RUN, 2026-11-16)** (§10, "Phase C1"). The **B3 notebook was run on Kaggle on 2026-11-11** (papermill 08:29 → 10:36 UTC, ≈ 127 min, 108 cells, 0 errors; outputs committed in `58cb2e7`; its full results are not yet transcribed here). That run showed (§4.4e (now §7.1)) that **25 i.i.d. noise columns through the deployed NestedSelect score piece F1 0.354, above the Dummy strict P95 0.323** (0.468 at C = 0.001), and that MERT has a **piece-identity shortcut** (same-piece nearest neighbour MERT-full 100 %, MERT-PCA25 90 %, interpretable 57 %, chance 3.4 %). C1 adds **§4.5c (now §5.3 / §6.9), an empirical chance null** (NestedSelect on noise, d = 25 / 10, 100 draws, cached in `null_f1_draws.csv`) against which every reported F1 is read, and a **diagnostic MERT per-layer study** (§2.1c caches all 13 hidden states in `emb_mert_layers_v3.npz`; §4.4f (now §7.2): Stage A piece identity / cross-piece condition decoding / within-piece ordering, a pre-declared suggestion printed but **not applied**, Stage B prediction profile in tier `6-mert-layers`, nested layer-selection stability). **`MERT_LAYERS` stays `[5, 6, 7]` until the author decides.** `RUN_MANUAL_AUDIT = False`. Notebook 114 cells. **Next:** the full run.
>
> **Earlier (2026-11-10): Phase B3 — APPLIED (since run, 2026-11-11)** (§10, "Phase B3"). The B2 notebook was run in full on Kaggle by the author (N = 116; outputs not committed). That run showed that the B2 **re-articulation splits followed the vibrato amplitude modulation** (MEC 24 / EXP 51 / EXG 68 splits for the same scores) and inflated the note-based timing features; B3 **disables the detector for every feature** (`REART_ENABLED = False`, kept as a report-only negative result), re-checks the 80 c range after a plateau merge, rejects vibrato windows whose best rate is on the 4 / 8 Hz grid edge, renames the descriptor caches to `feat_crepe_v4.csv` / `dtw_features_v4.csv`, and adds **§4.4e (now §7.1)** (MERT generalisation: train vs test, validation curve, nearest-neighbour identity — capacity overfitting vs piece-identity shortcut) and **§5.6b (now §10.3)** (family overlap: pair-removal tier `3-subfamily-pairs`, cross-family probes, correlations, a pre-declared triangulation rule). Notebook 108 cells. **Next:** interactive run up to §2.6 (now removed in C2) (A7 acceptance criteria), then the full run.
>
> **Earlier (2026-11-05): Phase B2 — descriptor repairs after the manual audit — APPLIED (since run in full)** (§10, "Phase B2"). The committed notebook carries an **N = 116 run of 2026-11-03** (papermill 15:26 → 17:11 UTC, with block weighting and the §2.6 (now removed in C2) audit); its §2.6 (now removed in C2) outputs were audited by eye and ear on 2026-11-05 and showed four descriptor faults (Viterbi-ramp portamento false alarms, notes merged with slides, real narrow vibrato rejected, repeated notes merged). B2 replaces the note segmentation (plateaus + re-articulation splits from a 10 ms RMS envelope), the glide rule (Viterbi-aware) and the vibrato detector (windowed sinusoid fit); caches `feat_crepe_v3.csv`, `dtw_features_v3.csv`, `rms10_v3.npz`. The 2026-11-03 run's Part 2–7 numbers are not recorded here and are superseded. **Next:** interactive run up to §2.6 (now removed in C2) (calibration grids + AUTHOR VALIDATION V1–V6), then the full run.
>
> **Revision status (2026-11-03): Phase B run on N = 118; N = 116 + block weighting + §2.6 (now removed in C2) audit applied (since run on 2026-11-03, see above).** Phase A was applied and run on Kaggle (2026-10-09); its numbers (kept in §10 as history) are superseded by the Phase B run. Every Part 4/5/6 predictive number from runs before Phase A is pre-audit and superseded.
>
> **Earlier revision note:** This supersedes the older "validation-mode (MERT+CREPE only)" handoff. The pipeline has since gained **per-rater normalization** (new ground truth), an **active Essentia block** (dynamics/timbre/tonal), a **shared leakage-free forwarding gate** for every tool, a full **Part 5 attribution suite** and **Part 6 tool-characterization**, and the **MFCC baseline was removed**. The interpretable-feature narrative has moved **from vibrato to dynamics/timbre + pitch**. **DTW (timing/rubato) is implemented and wired end-to-end (§2.4/§2.4b)** and a **systematic ablation study** (§4.4b (now §6.3)/c/d + §5.6 (now §10.2), one shared engine, two CV protocols, FDR-corrected) was added; **both were executed in the Kaggle run of 2026-08-16**. Notebook: `notebook/thesis_pipeline.ipynb`, **95 cells**, Kaggle P100.

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

## 2. Pipeline Architecture (current: Phase C3, 118 cells — applied, NOT RUN)

Runs top-to-bottom; **the notebook order is the story order** (Phase C2 — old → new map in the "Phase C2" section).
`§` = the in-notebook section banners. 118 cells (67 code, 51 markdown).

| Part | § | Content |
|---|---|---|
| **0 Setup** | 0.1–0.4 | install / imports / **config** (`VA_MID=3.0`, `NEUTRAL_RADIUS=0.75`, `MERT_LAYERS=[5,6,7]`, tag maps; C2: `TEMPOGRAM_UNRELIABLE`, `NULL_N_DRAWS_PRIMARY` (C3: `{7: 1000, 17: 1000}`, `NULL_N_DRAWS_PRIMARY_DEFAULT = 600`), C3: `NULL_CACHE_INPUT_DIR`, `CREPE_UNIQUE_R2_MAX = 0.2`, the shared `bh_fdr`, the unsupervised forwarding gate) / audio manifest |
| **1 Audience ground truth** | 1.1 · 1.1b · 1.2 · 1.2b · 1.3 | parse + exclusions (116) · per-rater normalization · aggregate + 5-class labels · normalization validation · `Y` |
| **2 Feature extraction** | 2.1 / 2.1b / 2.1c | MERT (768, layers 5–7, cache `emb_mert_v2.npz`) / PCA-95 forwarding (MERT-PCA25) / all 13 hidden states (`emb_mert_layers_v3.npz`, read only by §7.2) |
| | 2.2 / 2.2b | CREPE (tracks cached, plateau notes, no re-articulation split, sinusoid-fit vibrato with the band-edge rule, Viterbi-aware portamento; `feat_crepe_v4.csv`; the report-only re-articulation table and legacy FFT helpers — pending decision) / forwarding |
| | 2.3 / 2.3b / 2.3c | Essentia (dynamics / timbre / tonal) / forwarding / recording-level check (report only) |
| | 2.4 / 2.4b | timing (`timing_*`, reference-free) + DTW deviation (`dtw_*`, 0/1 at MEC → descriptive only) (`dtw_features_v4.csv`) / forwarding (`DTW_DEV_IN_MODELS = False`) |
| | 2.5 | `X_A/B/C/D/F`, `X_MERT_FULL`, `*_REG` (MERT-full in the regression heads), cache-name asserts |
| **3 Tool validation** | 3.1–3.4 (+ 3.3b) | class floors · coverage · CREPE sanity · DTW timing sanity (⚑ tempogram pointer) · MERT scree |
| **4 Manipulation check** | 4.1 | interpretable layer: `INTERP_PRED_COLS` (27 — predictive) / `INTERP_DESC_COLS` (30 — + DTW-deviation, descriptive only); ⚑ flags; `X_INTERP` (27) |
| | 4.2 | performers: Page's trend test per technique (DESC), **one BH family with the two rating tests** → `dose_response.csv` |
| | 4.3a / 4.3b / 4.3c + conclusion | listeners: Friedman / Wilcoxon (`FRIEDMAN_RES`) / Page on valence & arousal + dose figure / tag lift + ground-truth class share per condition / printed conclusion (`MANIP_CHECK`) |
| **5 Evaluation yardstick** | 5.1 | the one CV engine (protocols, NestedSelect default, nested ridge + block weighting, metrics, Dummy floors computed, Freedman–Lane + noise-ceiling helpers, guards) |
| | 5.2 | Dummy floors table (`metrics_dummy_floors.csv`) |
| | 5.3 | empirical null of NestedSelect on noise: 100-draw map (d = 10 / 25, `null_f1_draws.csv`) + **exact-width primary null** (A–D, Interpretable (27), DTW, CREPE + DTW; C3: 1000 draws at d = 7 / 17, 600 elsewhere; `null_f1_draws_primary.csv`, resumable, copied from the Kaggle input `violin-thesis-null-cache`) · `emp_null_ref` / `emp_null_mark` — C3: the ONE reference rule (exact width, else the next larger primary width, else d = 52 extrapolated); the map is not a reference |
| **6 Prediction** | 6.1 | NestedSelect ★default (✓ vs the empirical null) + 8-classifier sensitivity table, both protocols (`ablation_results.csv`) |
| | 6.2 | VA regression: **deployed nested ridge on MERT-full 768 and Interpretable (27)**; fixed Ridge / SVR / MLP on MERT-PCA25 = sensitivity |
| | 6.3 | ablation engine + 15-subset lattice (rows carry the empirical-null p) + tier `1b-tempogram-sensitivity` |
| | 6.4 / 6.5 | configuration ablation (4A / 4B / 4D / 4E) / target side (radius, raw vs normalized) |
| | 6.6 (+ 6.6b, figures) | per-class P / R / F1 / support + chance-corrected agreement α · Dummy floors vs the default per set |
| | 6.7 / 6.8 | per-condition F1 / quadrant accuracy from the deployed nested ridge + majority & permutation null, Fig 5 (60 points), Fig 6 |
| | 6.9 | every F1 vs its empirical null (`emp_null_readout.csv`), **BH within the primary family** (`emp_null_primary.csv`) |
| **7 MERT diagnostics — what the embedding encodes** | 7.1 | train vs test, validation curve, NN identity (pre-declared rules); the null's draw-0 guard |
| | 7.2 (+ continued) | MERT layer study (DIAGNOSTIC — `MERT_LAYERS` stays [5, 6, 7]) + candidates vs the empirical null |
| | 7.3 / 7.3b | MERT-dim ↔ feature correlation / linear probe (`PROBE_R2`) |
| | 7.4 | embedding distance vs the signed ΔV / ΔA (BH over 6), Fig 8 (3 panels), 2 exploratory correlations |
| | 7.5 | what each tool adds on top of MERT (lattice rows read from `ABL_PRED`, BH within §7.5; CREPE-unique by the probe rule) |
| **8 Within-piece prediction** | 8.1 | piece-centred nested ridge + within-piece permutation, Freedman–Lane incremental test, split-half ceiling |
| **9 Which techniques** | 9.1 | condition deltas (DESC, ⚑, BH over all tests) + Fig TE1 (3 panels) + Fig 7 |
| | 9.2 | interpretable linear read-out (27, ⚑) → `INTERP_CV_R` / `INTERP_CV_F1` |
| | 9.3 | SHAP — nonlinear, in-sample cross-check + SHAP-vs-β agreement table |
| **10 Cost of explainability** | 10.1 / 10.2 / 10.3 | CoE (27 vs MERT-PCA25 / MERT-full 768) / sub-family ablation (7 families) / family overlap (21 pairs, test 1 needs I > 0.02) |
| **11 Synthesis** | 11.1 | claims table (`claims.csv`, ✔ / ◐ / ✘ by stated rules) + `final_summary.csv` + `all_predictions.csv`; output-file reference |

### 2.1 The shared forwarding gate (key mechanism)

Every tool is **extracted in full** (nothing lost on disk), then **forwarded** through one shared, **unsupervised** selector `unsup_forward_select(candidates, corr_max=0.90, name, eps, order)`: near-constant prune + correlation prune, in an a-priori preference order. It reads **only the feature matrix, never `Y`** — this is what makes a once-up-front selection safe to feed LOOCV (a supervised selection would leak held-out folds). `StandardScaler` stays **inside** every CV pipeline. MERT (§2.1b), CREPE (§2.2b), Essentia (§2.3b) and DTW (§2.4b) all use this same gate → `MERT_FORWARD`, `CREPE_FORWARD`, `ESSENTIA_FORWARD`, `DTW_FORWARD`.

### 2.2 What enters the models vs what is kept aside

`X_A=MERT-fwd`, `X_B=CREPE-fwd`, `X_C=MERT+CREPE+Essentia+DTW (all fwd)`, `X_D=MERT+Essentia`, `X_F=MERT+DTW` (the §6.2 (now §7.5) complementarity contrast). **`X_MERT_FULL`** (768-dim) serves the *representational* analyses (§3.4 structure, §5.3 (now §7.4) distances, §6.1 (now §7.3)/§6.1b (now §7.3b) correspondence), which must see all dimensions, and — since Phase B (B-MERTFULL, `MERT_REG_REPR='full'`) — the **regression heads**: `MERT_REG` and `X_C_REG/X_D_REG/X_F_REG` are the same matrices with the PCA-forwarded MERT block replaced by `X_MERT_FULL`. The classification heads keep `X_A`–`X_F`. Diagnostic-only columns (`CREPE_DIAGNOSTIC`, `ESSENTIA_DIAGNOSTIC`, `TIMING_DIAGNOSTIC`) never enter forwarding, `df_interp` or the gate sweeps. `STRICT_BLOCKS=('MERT','Essentia')` → a missing block raises rather than silently zero-filling (per the "hard imports, no fallbacks" policy).

### 2.3 Evaluation protocol (§4.0 (now §5.1), Phase A) — one engine for every cross-validated number

- **Protocols:** `PRIMARY_PROTOCOL='piece'` (leave-one-piece-out on `PIECE_GROUPS`) and `loo` (optimistic secondary), always both, piece first, with the optimism gap; `get_cv(protocol)` is the only splitter builder (`'piece5'` = GroupKFold(5) on pieces, for the §6.1b (now §7.3b) probe only).
- **Default classifier (Phase B, B-NMS): nested model selection** (`DEFAULT_CLF_NAME='NestedSelect'`, `cv_predict_default` = `cv_predict_nested_select`). Inside every outer training fold an inner GroupKFold(5) over the training pieces scores the 13 `SELECT_CONFIGS` (NB · SVM-Linear balanced · LogReg balanced, C ∈ logspace(−3, 2, 6)) by pooled inner out-of-fold macro-F1; ties → NB, then the smallest C; the winner is refitted on the outer training fold. Choosing the classifier on the CV that reports it would be selection bias; nesting makes the reported score an unbiased estimate of the select-then-fit procedure. A content-hash memo returns byte-identical calls; outer folds run in parallel (joblib, `threadpool_limits(1)` per fold) and §4.0 (now §5.1) asserts parallel == serial exactly. Every call prints how often each config was chosen. **GaussianNB** (`NB_CLF_NAME`, `make_nb_clf`) — the Phase A pre-declared default — failed under `piece` and stays as a reported row. §4.1 (now §6.1)'s 8 classifiers are a sensitivity table only; per-class / per-condition / α analyses use the default on `X_C`.
- **Engine:** `cv_predict_multilabel` (manual binary relevance, `SafeMultiOutputClassifier` semantics, optional per-label inner-CV tuning) — asserted identical to the old pipeline for GaussianNB; `nested_ridge_predict` (α by piece-grouped inner GroupKFold(5), replaces every `RidgeCV`); `NestedRidgeOperator` (closed-form twin, for permutation refits); `oos_r2` (baseline = training-fold mean), `rmse`, `bh_fdr`, `within_piece_permutation`.
- **Chance floors:** Dummy seeded per fold × label, 200 repeats → mean / SD / P95; "beats chance" = above the max P95 across strategies. **Phase C1:** the Dummy floor ignores the selection procedure and is too low for NestedSelect (B3: noise 0.354 > P95 0.323), so §4.5c (now §5.3 / §6.9) adds an **empirical null** — NestedSelect on noise of the nearest width (`EMP_NULL_F1`, `emp_null_p`); "beats the empirical null" = one-sided p < 0.05.

---

## 3. Progress since the previous handoff

- **Phase C3 (2026-11-20, applied, NOT RUN).** Small fixes after the C2 run: the null resumes from the Kaggle input `violin-thesis-null-cache`, per-width draw targets (1000 at d = 7 / 17, 600 elsewhere; settings hash unchanged), one empirical-null rule everywhere, the claim-10 rule revised post hoc, unique Fig 8 labels. 15 cells changed (code: §0.3, §5.3, §6.3, §6.9, §7.2 continued, §7.4, Fig 8, §10.2, §11.1; markdown: intro, §5.3, §6.3, §6.9, §10.2, output reference); `ast.parse` + use-before-define scan clean; all outputs cleared; the C2 notebook is kept as `notebook/thesis_pipeline_C2_backup.ipynb`. Details in §10, "Phase C3".
- **The C2 notebook was run on Kaggle (2026-11-20; outputs in `5c94dc2`).** §5.3 took 509.7 min and stopped the primary null at its 480-min budget (600 draws per width, 561 at d = 52). Not transcribed into §4.
- **Phase C2 (2026-11-20, applied; run 2026-11-20).** Restructured to the story order (12 parts, 118 cells) + the C2 obligations 1, 2, 3, 5a, 5b, 6, 7 (4 on hold). Every code cell re-checked with `ast.parse` and an AST use-before-define scan; all outputs cleared; C1 notebook kept as `notebook/thesis_pipeline_C1_backup.ipynb`. Details and the run checklist in §10, "Phase C2".
- **Phase C1 (2026-11-16, applied, not run).** Changed cells (outputs of changed code cells cleared): intro (md), §2.6 (now removed in C2) md + §2.6a (now removed in C2) code (`RUN_MANUAL_AUDIT = False`), §4.4c (now §6.4) md + code (the "per-layer MERT pending" text), §4.5 (now §6.6) md, §5.6 (now §10.2) md + code (empirical-null P95 line, `alone_emp_p`), §7.1 (now §11.1) code (empirical-null columns), output reference (md). New cells: §2.1c (md + code, after §2.1b), §4.4f (now §7.2) (md + code, after §4.4e (now §7.1)), §4.5c (now §5.3 / §6.9) (md + code, after the §4.5 (now §6.6) figures) → 114 cells. Details in §10, "Phase C1".
- **The B3 notebook was run on Kaggle on 2026-11-11** (≈ 127 min, 0 errors; outputs committed in `58cb2e7`). Not yet transcribed into §4; the numbers C1 builds on are in §10, "Phase C1".
- **Phase B3 (2026-11-10, applied; run 2026-11-11).** Changed cells (outputs of changed code cells cleared): Part 0 intro (md), §0.3 (`OLD_CACHE_NAMES` += the two v3 caches), §2.2 (md + code: `REART_ENABLED = False`, `reart_enabled` switch, merge range check, band-edge rule, `vibrato_window_reason`, banner, report-only re-articulation evidence table, note-count Friedman, `feat_crepe_v4.csv`), §2.4 (md + code: `dtw_features_v4.csv`, comments), §2.5 (cache asserts), §2.6 (now removed in C2) (md + both code cells: disabled path, band-edge display, V4 INFO, ≥ 80 % span rule, V7), §4.4c (now §6.4) (one print), §5.6 (now §10.2) (md + code: chance lines, q markers, commonality summary, `q_loss_r2_val`, timing reading), output reference (md). New cells: §4.4e (now §7.1) (md + code, after §4.4d (now §6.5)) and §5.6b (now §10.3) (md + code, after §5.6 (now §10.2)). Details in §10, "Phase B3".
- **The B2 notebook was run in full on Kaggle (N = 116)** by the author; its outputs are not committed (the repo notebook keeps the 2026-11-03 outputs of the unchanged cells). The numbers that motivated B3 are in §10, "Phase B3", evidence (a)–(e).
- **Phase B2 — descriptor repairs after the manual audit (2026-11-05, applied, since run).** Changed cells (outputs cleared): §0.3 (`OLD_CACHE_NAMES` += the two v2 descriptor caches), §2.2 (md + code: B2 segmentation, B8b vibrato, B9b portamento, Viterbi-source check, `rms10_v3.npz`, old vs new note counts, REART sensitivity, calibration grids), §2.4 (md + code: `env_db` threaded into the timing onsets, `dtw_features_v3.csv`), §2.5 (cache assert), §2.6 (now removed in C2) (md + both code cells: B2 tables / figures, detail asserts, AUTHOR VALIDATION V1–V6), §4.4c (now §6.4) (one "still pending" print), §5.6 (now §10.2) (md + the vibrato caveat print), output reference. Details in §10.
- **The N = 116 run (2026-11-03, papermill 15:26 → 17:11 UTC)** of the edits below is in the committed notebook; its §2.6 (now removed in C2) outputs are the audit evidence; its Part 2–7 numbers are not recorded and are superseded by B2.
- **Notebook edits for the next run (2026-11-03, since run).** (1) Participant exclusions wired into §1.1 (`EXCLUDED_PARTICIPANTS` in §0.3; asserts 116 participants / 2,320 rows / 60 excerpts; §1.4 (now §4.3a) prints the exclusions, the old `df_long_clean` branch is gone; §4.0 (now §5.1) prints the prevalences from `Y` instead of the hardcoded N = 118 ones). (2) **Equal-share block weighting** of the multi-block MERT-full regression heads (`block_weights`, `BlockScaler`, `REG_BLOCK_WEIGHTING = 'equal'`; §4.0 (now §5.1) guard; §4.4b (now §6.3)/§4.4d (now §6.5) heads; new tier `4E-block-weighting`). (3) **Temporary §2.6 (now removed in C2) manual descriptor audit** (6 excerpts). (4) Descriptive text made accurate (diploma thesis / NTUA, thesis framing and questions, torch pin = P100 / runs = T4, stale markdown in §2.5, §4.1 (now §6.1), §5.2 (now §9.3), §5.4 (now §9.2), §6.1 (now §7.3), §6.2 (now §7.5), output reference). The notebook is now 104 cells.
- **Phase B run on N = 118 (papermill 2026-10-09 23:45 → 2026-10-10 01:22 UTC, T4, ≈ 97 min, 101 cells, 0 errors).** Results in §4. It surfaced the MERT-dominance problem of the multi-block regression heads (→ block weighting) and several descriptor questions (→ §2.6 (now removed in C2) audit, §5).
- **Listening-study QC re-validated and finalised (2026-10-21).** The months-old cleaning scripts were checked against the five CSVs; the participant screening was rewritten (`questionnaire_cleaning_v3.py`, four bugs fixed — §11) → 2 exclusions. Excerpt QC re-run at N = 116. Demographics and durations scripts written. The normalization and ground truth were reproduced offline: at N = 118 the reproduction matches the notebook exactly, so the N = 116 numbers in §11 are what the notebook will produce once the exclusions are wired in.
- **Diploma thesis Chapters 1–7 written (2026-10-18 → 2026-10-30)** — front matter, Introduction, background Chapters 3–6 rewritten against Phase B, Chapter 7 Dataset written from scratch with N = 116 numbers; colour palette for all figures; numbered subsections. See §12.

- **Phase B of the Oct-2026 audit applied (2026-10-10) — results pending the next Kaggle run.** Nested model selection as the default classifier (B-NMS) · Freedman–Lane incremental test (B-INC) and split-half noise ceiling (B-NC) in §4.9 (now §8.1) · MERT-full regression heads (B-MERTFULL) · CREPE tracks cached + octave correction + note segmentation (B-shared) · repaired vibrato (B8), portamento (B9), F0 range (B10) · normalised flux (B11) · silence-gated dynamic range (B12) · `timing_*` family + 151-frame tempogram with log-normal prior (B13) · flatness noise floor + optional session map (B-NF) · `_v2` caches (B-cache). See §10.

- **Phase A of the Oct-2026 audit applied and run on Kaggle (2026-10-09; results in §4).** §4.0 (now §5.1) CV engine · re-seeded Dummy floors · leave-one-piece-out primary everywhere · nested ridge + R²/RMSE · §4.9 (now §8.1) within-piece prediction · MERT pos-conv weight restore (cache v2) · GaussianNB default + best-of selection removed · class balancing + inner-CV C (+ tier `4D-balancing`) · group-aware MERT probe · BH-FDR in §5.1 (now §9.1; its head → §4.1) · §2.3c recording-level check. See §10.

- **Per-rater normalization added (§1.1b/§1.2b) and made the new ground truth.** Per-dimension z with σ-floor 0.5, rescaled to preserve the raw grand mean and between-excerpt SD (so VA_MID and quadrant geometry are untouched — only rater scale-use bias is removed). **Validated via the ring:** the 40 anchors are each rated by two independent pools, so the cross-form discrepancy before/after is the honest test. **Functional with or without participant exclusions** (all constants derived at runtime). Raw retained in parallel.
- **Essentia re-enabled (§2.3/§2.3b)** — dynamics (integrated loudness, dynamic/loudness range), timbre (spectral flux/centroid/rolloff, dissonance, log-attack time), tonal (key strength, HPCP entropy). This closed the dynamics/timbre gap and populates the signal-processing tool category.
- **Shared leakage-free forwarding** unified across all tools (§2.1).
- **Part 5 attribution suite** built out: condition-delta, dose-response (Page), SHAP, embedding-distance, interpretable read-out, cost of explainability.
- **Part 6 tool characterization** (correspondence probe + complementarity).
- **MFCC baseline removed** — it was degenerate (F1≈0, indistinct from the Dummy floor) and carried out-of-topic leaderboard framing. The **DummyClassifier floors (§4.5b (now §5.2 / §6.6))** are the official chance baseline.
- **Cleanup pass (partial):** MFCC gone; some duplicate/out-of-topic figures still pending removal (see open issues).
- **Technical report drafted** — Abstract + Introduction + Methods (IEEEtran), following the Chowdhury/Widmer interpretable-MER backbone (see §6).
- **DTW (timing/rubato) implemented and integrated (§2.4/§2.4b).** Two families: **(1) reference-free** — local-tempo variability, IOI CV, onset rate, pulse clarity; computed per excerpt with no reference, so all 60 (MEC included) are graded and the family is condition-INdependent. **(2) within-piece DTW deviation** — chroma sequences of EXP/EXG aligned to the same piece's MEC with `librosa.sequence.dtw`; warp-path deviation, local-slope variability, tempo/path ratios. MEC is the alignment reference, so its family-(2) values are the **self-alignment identity (0 deviation / 1 ratio) by construction** — meaningful for attribution but it means family (2) partly encodes the condition. §2.4b splits them: family (1) alone feeds the predictive matrices (`DTW_MODEL_COLS`, `DTW_DEV_IN_MODELS=False`), both feed the attribution frame (`df_dtw_attr`, `DTW_DEV_IN_INTERP=True`). DTW is wired into §2.5 (X_C, new X_F), §3.3b, §5.1 (now §9.1; its head → §4.1), §5.1b (now §4.2 / §4.3b), §5.2 (now §9.3), §5.4 (now §9.2), §5.5 (now §10.1), §6.1 (now §7.3), §6.1b (now §7.3b), §6.2 (now §7.5). **Executed in the 2026-08-16 run** (13-col block; 11 forwarded = 8 ref-free + 3 dev; the models receive the 8 ref-free).
- **Systematic ablation study added (§4.4b (now §6.3)/c/d + §5.6 (now §10.2) + the §6.2b (now removed in C2) guard).** One engine `ablate()` scores every tier under one protocol (SVM-Linear → F1-macro, RidgeCV → V/A, scaler inside the CV pipeline — the same protocol as §5.5 (now §10.1)/§6.2 (now §7.5), so all numbers are comparable by construction), caches per-excerpt predictions, and consolidates into `ablation_master.csv`. It closes the asymmetry that §4.4 (now removed in C2; table in §6.1) compared only four hand-picked sets while §6.2 (now §7.5) could only ever *add* to MERT: the **full 15-subset lattice** now yields each tool's **unique** (full − full∖tool) and **marginal** (alone − chance floor) contribution, MERT's included. Every row is scored under **both** LOOCV and leave-one-piece-out, with bootstrap CIs, paired permutation p-values and BH-FDR. §5.6 (now §10.2) decomposes the single cost-of-explainability number by expressive family, giving Part 5 a third and *predictive* line of evidence alongside dose-response and SHAP. Three asserts keep it honest (the lattice must rebuild `X_C/X_D/X_F` exactly, the §4.4d (now §6.5) label re-derivation must reproduce `Y`, and §6.2 (now §7.5)'s additive numbers must equal the lattice rows). **Executed in the 2026-08-16 run** (112 scored rows, 9 tiers; all three asserts passed) — those numbers are pre-audit (SVM-Linear + RidgeCV-GCV protocol) and superseded by Phase A.

---

## 4. Current findings

### Current results — B3 run (2026-11-11) + C1 run (2026-11-16), N = 116

> ⚠️ **Phase C2 (applied 2026-11-20, NOT RUN) will replace** these numbers: everything that used the 30-feature interpretable layer in a model (now 27: Interpretable F1 ≈ 0.445, R²v ≈ 0.334, R²a ≈ 0.636 in the C1 run) — §6.2 (deployed rows), §7.1 / §7.2 interpretable references, §9.2 betas, §9.3 SHAP, §10.1 CoE, §10.2 families (7 instead of 8), §10.3 pairs (21 instead of 28; test 1 with I > 0.02); every empirical-null p / ✓ (exact-width primary null — Phase C3: 1000 draws at d = 7 / 17, 600 elsewhere, one reference rule — BH in the family — §6.9); the §4.2 stars (BH q instead of raw p); the new §6.3 tempogram rows, §6.8 (deployed nested ridge, chance references), §7.4 (signed, BH, 3 panels), §7.5 (lattice reads, CREPE-unique rule). Unchanged by construction: §4.2 / §9.1 values (descriptive 30), §8.1, the lattice, MERT rows, the Dummy floors, the 100-draw map. Section numbers below are pre-C2 with "(now §x.y)".

> ⚠️ **Phase C3 (applied 2026-11-20, NOT RUN) will also replace:** §6.9 / claim 5 (the DTW p / q — 1000 draws at d = 7); every ✓ / p that used the 100-draw map (the C1 empirical-null table below, the lattice, §7.1 and §7.2 rows); §10.2 `alone_emp_p` (the "against the empirical d = 10 null" line below); claim 10 (revised rule). Marked "⚠️ C3" inline.

> **B3 run (2026-11-11, full, 108 cells, 0 errors, ≈ 127 min) and C1 run (2026-11-16, by the author), N = 116.** Transcribed 2026-11-17 from the committed B3 outputs and the C1 numbers the author reported (`emp_null_readout.csv`, the §4.4f (now §7.2) summary and profile figure). The pipeline is seeded and deterministic, so the B3 numbers below that C1 did not touch are expected to be identical in the C1 run. **These supersede every earlier Part 2–7 number.**

**§2.2 segmentation (B3).** Deployed (re-articulation OFF) totals MEC 621 · EXP 601 · EXG 602; with the detector ON (report only) 645 / 652 / 669. Friedman χ²(2): deployed 0.535, p = 0.765 (**V7 PASS** — note counts do not depend on the condition); detector-on 0.861, p = 0.650. 1824 notes ≥ 60 ms; 46 adjacent notes < 70 c apart (2.5 %; 26 in B2 — the B3 merge range check leaves a few near-pitch neighbours unmerged).

**§2.2 vibrato (B3).** Analysed notes ≥ 500 ms: 477 (650 ms: 354, 800 ms: 263); no note ≥ 500 ms in 1 excerpt (SQ8_exrpt3_MEC). 1881 windows; 377 have their best rate on the 4 / 8 Hz grid edge (MEC 247 of 598), 17 of them rejected *only* by the edge rule (MEC 3). Rates 4.14–6.40 Hz. No vibrato window → declared median 5.770 Hz for 19 excerpts (the MEC excerpts + SQ8 exrpt1 EXP/EXG + exrpt3 EXP/EXG — correct rejections, confirmed by ear). MEC false positives at the deployed cell: **0.7 % of windows**.

**§2.2 portamento (B3).** **39 glides in 763 transitions (5.1 %)** (Phase B: 139, median 62 ms); rise 88–325 ms (median 100), excess 56–321 ms, extent 84–1202 c. Verdicts: across an unvoiced gap 1001 · too short 587 · interval out of range 94 · glide 39 · no 90 % crossing 21 · not monotone 16 · viterbi-limited jump 4 · too slow 2. MEC false positives: **2.0 % of transitions**.

**§2.6 (now removed in C2) AUTHOR VALIDATION (B3).** **19 PASS · 6 FAIL · 3 INFO** (B2: 17 PASS · 11 FAIL).
- **Notes:** Méditation 19 / 20 / 21 (expected 19 / 19 / 20 — accepted).
- **V3 PASS:** the EXG C♯6 trend range is 79 c (128 c in B2).
- **V2 / V3 vibrato with ≥ 80 % span: PASS.** EXG F♯5 8/9 windows over the full span; EXP D5 4/6 and B4 7/13 windows.
- **Méditation vibrato coverage:** EXP 0.38 (Phase B) → 0.70, EXG 0.58 → 0.73.
- **Méditation MEC portamento:** 0.170 glides/s → 0.
- **Remaining FAILs:**
  - V2 C♯6→D6 is not a glide: the interval is measured as 77 c (< 80 c), because the ±40 c plateau tolerance absorbs the start of the slide (C♯6 median +19 c).
  - V6: 4 MEC fail (Bach coverage 0.083; glides in Schindler / Mozart_VC3 / Secret_Garden — probably position shifts), plus the V6 aggregate.
- **V4 INFO:** Mozart_VC4 17 / 16 / 20 notes (known limitation).

20-piece direction checks:

| Check | Phase B | B3 |
|---|---|---|
| portamento rate / share / extent | 2 / 4 / 4 of 20 | **14 / 14 / 17** |
| vibrato extent | 17 / 20 | 15 / 20 |
| vibrato coverage | 17 / 20 | 14 / 20 |
| tempo log2-SD | — | 6 / 20 (✗, unchanged) |
| pulse clarity | — | 4 / 20 (✗, unchanged) |

The vibrato-extent rule "MEC ≤ 3 c" fails on any single MEC false-positive window, because the extent is conditional on detection; coverage is the presence measure.

**Pre-declared timing reading (B3 A7).** Timing (ref-free) alone, F1 (piece) = **0.398** (Phase B 0.417 · B2 0.542); removal loss +0.087 (q = 0.17). → **The B2 value was the vibrato artefact of the re-articulation splits.**

**§5.1b (now §4.2 / §4.3b) dose-response (Page test, 20 pieces; raw p — no BH in the notebook yet).**
- **Techniques:** vibrato extent z **+5.45** · flux (normalised) +5.22 · LUFS +4.74 · vibrato coverage **+4.66** · DTW deviation / slope SD +4.59 (built in at MEC) · DR +4.11 · rolloff −2.85 · portamento rate +2.69 (p .007) and share +2.61 (p .009) · IOI CV +2.37 and IOI mean −2.37 (p .018) · tempo log2-SD −2.21 (p .027) · local tempo −2.06 (−9.6 BPM, p .040) · note rate +2.06 (p .040) · vibrato rate 0.00.
- **Audience:** arousal **+3.95** (p .0001), valence +0.47 (n.s.).
- **A7 not met:** it expected note rate / IOI back to n.s. Since the note counts no longer depend on the condition (Friedman p .77), their rise is a duration effect: EXG is slightly faster overall.
- **BH over the 30 technique tests (computed by hand):** keeps the first 12 (down to IOI CV / IOI mean, p .018); drops tempo log2-SD, local tempo and note rate.
- **Tempogram contradiction:** the tempogram's local tempo says EXG is *slower*, while note rate says *faster* (see Limitations).

**§4.1 (now §6.1) / §4.4b (now §6.3) classification (NestedSelect).**

| Set | piece | loo |
|---|---|---|
| A MERT | 0.310 | 0.623 |
| B CREPE | 0.330 | 0.511 |
| C Combined | 0.367 | 0.682 |
| D MERT + Essentia | 0.324 | 0.587 |

Lattice (piece): Essentia 0.356, DTW 0.398, CREPE + DTW **0.460**, Essentia + DTW 0.373, CREPE + Essentia + DTW 0.445. MERT unique F1 (piece) −0.078 (q 0.23, n.s.; B2 −0.122). **Read every F1 against the C1 empirical null below.**

**§5.4 (now §9.2) / §5.5 (now §10.1) regression and cost of explainability (piece).**
- **Interpretable (30):** r_v 0.429 / **R²v 0.258** · r_a 0.783 / **R²a 0.648** · F1 0.444.
- **MERT-full:** R²a 0.446, R²v 0.012.
- **Headline (negative = interpretable wins):** ΔR valence **−0.370** · ΔR arousal **−0.156** · ΔR² valence −0.246 · ΔR² arousal −0.202 · ΔF1 −0.134. The interpretable layer wins on every column.
- **LOOCV, interpretable:** R²v 0.500, R²a 0.757, F1 0.538.

**§4.9 (now §8.1) within-piece (piece).**
- **Noise ceilings:** V 0.481 [0.165, 0.727], A 0.664 [0.509, 0.781].
- **Δvalence:** Condition-only R² 0.0005. B-INC incremental: Interpretable **+0.164 (q 0.027)**, MERT-full +0.166 (q 0.027), Interp + MERT-PCA +0.170 (q 0.027), MERT-PCA +0.097 (q 0.051).
- **Δarousal:** Condition-only **0.396** (0.60 of the ceiling). Interpretable incremental +0.063 (p 0.026, q 0.051). The MERT sets add nothing.

**§4.4e (now §7.1) MERT generalisation (piece).**

Train / test F1:

| Block | train | test | gap |
|---|---|---|---|
| MERT-PCA25 | 0.49 | 0.31 | 0.18 |
| MERT-full | 0.86 | 0.35 | 0.51 |
| interpretable (30) | 0.79 | 0.44 | 0.34 |
| random 25 | 0.34 | 0.35 | −0.02 |

Validation curve, test F1:

| | MERT | interpretable | noise |
|---|---|---|---|
| fixed C ≥ 0.01 | 0.07–0.28 | 0.43–0.47 | 0.14–0.17 |
| C = 0.001 | 0.361 | 0.448 | 0.468 |

Same-piece nearest neighbour (chance 3.4 %; all p < 0.0001):

| Space | same-piece NN | silhouette by piece |
|---|---|---|
| MERT-full | 100 % | 0.56 |
| MERT-PCA25 | 90 % | 0.26 |
| Essentia | 65 % | 0.11 |
| interpretable | 57 % | 0.07 |
| DTW | 52 % | 0.01 |
| CREPE | 28 % | −0.10 |

**Verdict: the PIECE-IDENTITY SHORTCUT rule fired; CAPACITY OVERFITTING did not.** MERT-full's same-condition share of 0 % carries no information: with 100 % same-piece neighbours it is forced.

**§5.6 (now §10.2) sub-families (piece; full 30-feature layer F1 0.444, R²a 0.648).**
- **Removal losses:** only Dynamics on R²a is significant (**0.120, q 0.028**). Timing (ref-free) F1 +0.087 (q 0.17).
- **Alone, F1:** Timing 0.398 · Timbre 0.393 · Dynamics 0.327 · Pitch 0.323 · Vibrato 0.316 · DTW-dev 0.314 · Tonal 0.295 · Portamento 0.273. Against the empirical d = 10 null (P95 0.370), only Timing and Timbre are above. ⚠️ C3: `alone_emp_p` now reads the primary null of the family's width or the next larger one.
- **Alone, R²a:** Dynamics 0.436 · Pitch 0.355 · Timing 0.319 · Timbre 0.153 · DTW-dev 0.122 · Vibrato 0.050 · Tonal −0.003 · Portamento −0.087.
- **Commonality (R²a):** full 0.648, Σ alone 1.34, Σ removal loss 0.30 → **shared ≈ 0.35**.

**§5.6b (now §10.3) family overlap (piece; pair tier 11.9 min).**
- **Significant interactions (pair loss q < 0.05):**
  - R²a: Pitch ↔ Timing (ref-free) I = +0.079 (q 0.025); Timing (ref-free) ↔ DTW-dev +0.108 (q 0.039). Both substitutes.
  - R²v: Pitch ↔ Timbre +0.136 (q 0.025); Portamento ↔ Timing (ref-free) +0.104 (q 0.042); Timbre ↔ Timing (ref-free) +0.022 (q 0.039).
  - Dynamics with Pitch / Timbre / Tonal / DTW-dev: each its own (I ≈ 0).
  - No significant F1 interaction.
- **Within-piece probe (family from all the others), R²:** DTW-dev 0.57 · Dynamics 0.43 · Vibrato 0.32 · Timbre 0.11 · Portamento 0.03 · Timing (ref-free) 0.03 · Tonal −0.02 · Pitch −0.07.
- **Triangulation:** overlap claimed only for **Dynamics ↔ Timing (DTW-dev)** (all 3 tests); 11 pairs pass one test.
- **Reading — two kinds of family:**
  - **"Performance" families** (dynamics, vibrato, DTW-dev, partly timbre) co-vary within a piece, because the performer escalates them together; the common cause is the condition.
  - **"Score" families** (pitch, tonal, reference-free timing) are ≈ constant within a piece.
  - The shared arousal variance is multi-way (a common cause), not pairwise duplication. → **A limit of the design: the effect cannot be attributed to a single technique.**

**§4.5c (now §5.3 / §6.9) empirical chance null (C1; piece, 100 draws).** ⚠️ C3: every p / ✓ in this table is replaced by the exact-width primary null (C2, completed in C3: 1000 draws at d = 7 / 17, 600 elsewhere) under the one reference rule.

| Width | Dummy strict P95 | Empirical P95 |
|---|---|---|
| d = 10 | 0.323 | **0.370** |
| d = 25 | 0.323 | **0.459** |

(Dummy uniform mean 0.272.) 36 / 52 reported piece F1 beat the Dummy P95, but only **4 / 52 beat the empirical null**. These are 3 distinct results, all containing the reference-free timing block:

| Set | F1 (piece) | p | Beats the empirical null |
|---|---|---|---|
| DTW (ref-free, 7) | 0.398 | 0.0099 | yes (lattice + §4.4e (now §7.1)) |
| CREPE + DTW (17) | 0.460 | 0.0099 | yes |
| Essentia + DTW (17) | 0.373 | 0.0495 | yes |
| A MERT | 0.310 | 0.83 | no |
| B CREPE | 0.330 | 0.25 | no |
| C Combined | 0.367 | 0.51 | no |
| D MERT + Essentia | 0.324 | 0.74 | no |
| CREPE + Essentia + DTW | 0.445 | 0.089 | no |
| interpretable (30) | 0.444 | 0.089 | no |
| Essentia | 0.356 | 0.12 | no |
| MERT-full | 0.345 | 0.64 | no |
| random control | 0.354 | 0.58 | no (the null is calibrated) |

LOOCV rows were not drawn (n/a). Caveats:
1. **Resolution:** 100 draws → the smallest attainable p is 1/101 = 0.0099. With BH over ≈ 20 distinct piece rows nothing survives.
2. **Width mapping:** the 17-feature sets were read against d = 10. Against d = 25 (P95 0.459), Essentia + DTW fails clearly and CREPE + DTW only just passes.
3. **Wide sets:** sets wider than 25 (Combined 52, interpretable 30, MERT-full 768) use the d = 25 null, an extrapolation that is likely lenient. They fail anyway.

**§4.4f (now §7.2) MERT layer study (C1, diagnostic).**
- **Stage A:**
  - **A1, same-piece NN:** 93–100 % for every candidate (lowest L10 93.3 %, silhouette 0.467; W05-07 100 %; interpretable 65 % in this metric).
  - **A2, condition decoding (piece-centred nearest-centroid):** 0.73–0.90 for all 25, all q < 0.05. Best L05 0.900 = W05-07 0.900 (interpretable 0.767).
  - **A3:** best L04 0.85 (W05-07 0.80); layers 0–3 are weak (0.45–0.65).
- **Stage B (piece):**
  - **F1:** 0.28–0.40; none beats W05-07, and **0 / 25 beat the empirical null** (W05-07 p 0.832).
  - **R²a:** peaks at L04 ≈ 0.535 (W03-05, W04-06 ≈ 0.51–0.53; paired q < 0.05 vs W05-07 0.445); 0.35–0.47 above layer 5; ≈ 0.08 at L00.
  - **R²v:** ≈ 0 at every depth (−0.11 … +0.07).
- **Nested selection (fixed estimators):** classes L00 in 5 / 20 folds (9 distinct choices) · R²a L11 9 / 20 (5 distinct) · R²v L11 16 / 20 (5 distinct) — **no stable best layer**.
- **The rule suggested W09-11 — not applied:** A1 drops only from ≈ 100 to ≈ 95 %, and Stage B is worse.
- **Conclusions:**
  1. The piece-identity shortcut is a property of the whole MERT stack, not of the layer choice.
  2. MERT *does* encode how a piece is played (A2 0.90 > interpretable 0.77), but only once the piece mean is removed: piece identity masks the information.
  3. No depth closes the gap to the interpretable layer. Best F1 ≤ 0.40 vs 0.44; best R²a 0.535 vs 0.648; R²v ≈ 0 vs 0.258. This strengthens the cost-of-explainability finding.
  4. **Decision (recommended; the author confirms): keep `MERT_LAYERS = [5, 6, 7]`.** Layers 3–6 for arousal (+0.09 R²a) may only be reported as a post-hoc sensitivity analysis: they were picked after seeing the ratings, and the nested selection did not choose them consistently.

**Overall conclusions (B3 + C1):**
1. **Classification:** classifying the 5 VA classes on unseen pieces is at the deployed pipeline's own chance level for every feature set except those that contain the reference-free timing block (tempo / rhythm / IOI). This is consistent with tempo as a primary emotion cue. Report classification against the empirical null.
2. **Regression and trend tests are unaffected** (noise R² ≈ 0):
   - arousal R² 0.648 / valence 0.258 on unseen pieces;
   - escalation of vibrato extent / coverage, flux, loudness and DR;
   - arousal follows the manipulation, valence does not;
   - within a piece, the features add to valence beyond the condition (q 0.027).
   The thesis's predictive claims rest on these.
3. **Cost of explainability:** the interpretable layer beats MERT in R and R² on unseen pieces. The F1 version compares two numbers at chance and should not be emphasised.
4. **MERT:** its failure on unseen pieces is a piece-identity shortcut, at every depth, not capacity overfitting.
5. **Audit:** the manual audit caught an artefact that would have become a headline (timing F1 0.542).

### Historical — Phase B run (N = 118, 2026-10-09/10; superseded by the B3 + C1 runs above)

> **Phase B results — Kaggle run on N = 118 (papermill 2026-10-09 23:45 → 2026-10-10 01:22 UTC, Tesla T4, 101 cells, 0 errors), pre-block-weighting.** The B2 run on N = 116 (outputs not committed; its B3-relevant numbers are in §10, "Phase B3") and the next **B3** run supersede them; every multi-block MERT-full regression number below is unweighted, and **every vibrato / portamento / timing-onset number below uses the Phase B descriptors that the 2026-11-05 audit found faulty** (§10, Phase B2).

**Guards.** All passed (§4.0 (now §5.1) equivalence, NestedSelect parallel == serial on 4 CPUs, §4.4b (now §6.3) lattice rebuild incl. the `_REG` matrices, §4.4c (now §6.4) 4D, §4.4d (now §6.5) label reproduction, §4.9 (now §8.1) closed-form equivalence incl. B-INC, §5.5 (now §10.1)↔§5.4 (now §9.2), §6.2b (now removed in C2) exact, §2.5 cache names).

**§2.2 CREPE.** Octave fixes 3 frames in 3 excerpts. Notes ≥ 800 ms: 259 (600 ms: 391, 1000 ms: 198); no qualifying note in 1 excerpt (`Shostakovich_SQ8_exrpt3_MEC`). 121 vibrato notes; rate undefined → declared median 5.811 Hz for 23 excerpts (all MEC with notes, plus SQ8 exrpt1 EXP/EXG and exrpt3 EXP/EXG — the author confirmed by ear that no vibrato was played there, so these are correct rejections). Rate 37 distinct values; half-extent 0–32.1 c; coverage 0–1. Portamento: 139 glides in 572 transitions (24.3 %), 50–312 ms (median 62), 80–1907 c. F0 range max 3100 c. 1626 notes ≥ 60 ms, 17 adjacent notes < 70 c apart.

**§2.3 / §2.3c Essentia, recording level.** DR gate median 6.2 % of frames (max 26 %), DR max 34.7 dB; 6 of the 9 gate variants have ρ < 0.9 vs the deployed DR (0.79–0.85). r(flux, LUFS) raw 0.874 → normalised 0.508. Forwarded 10 (short-term loudness SD dropped, r 0.97 with LRA). Flatness noise floor: 0/60 conclusive at FLAT_MIN 0.2 (9 at 0.1); 0 flags; LUFS by condition MEC −26.8, EXP −24.1, EXG −22.6; session map not provided.

**§2.4 / §3.3b timing.** DTW forwarded 10 (7 reference-free + 3 deviation; inside 3–10, no warning). Group means: tempo log2-SD MEC 0.261 > EXP 0.232 > EXG 0.209; IOI CV / SD and pulse clarity not monotone (0/3 non-decreasing). Author's explanation (2026-11-03): in EXG he played faster and more virtuosically in fast pieces and took more time in slow ones → local tempo / duration ratio are piece-dependent; tempo log2-SD and pulse clarity are checked visually in §2.6 (now removed in C2).

**Chance floors.** piece: uniform mean 0.2715, stratified 0.161, strict P95 0.3274; loo: 0.2720 / 0.183 / 0.3202.

**Classification (NestedSelect default).** piece: A 0.282, **B 0.361 (the only set above the strict P95)**, C 0.302, D 0.281; loo: A 0.517, B 0.476, C 0.626, D 0.540. NaiveBayes (Phase A default) piece: A 0.112, B 0.260, C 0.162, D 0.126. On X_C (piece) LA_HV and Neutral collapse (recall 0); per condition MEC 0.24 / EXP 0.29 / EXG 0.32; Krippendorff α HA_HV 0.31, LA_LV 0.31, others ≤ 0.03. Tier 4D (piece): Interp + SVM-Linear balanced + inner-CV C 0.417 [0.346, 0.475] (Δ +0.063 vs nested, q = 0.13); NB below nested on X_A / X_C / Interp (q ≤ 0.017).

**Regression (nested ridge, piece).** Ridge-nested · MERT-full R²v +0.016, R²a +0.446. Lattice: Essentia + DTW R²v 0.29, **R²a 0.67** (best); CREPE + Essentia + DTW R²a 0.60; every set containing MERT-full R²a 0.446–0.459 and R²v 0.014–0.019 — the 768 MERT columns dominate a single-α ridge (→ equal-share block weighting, next run). 4A-mert-pca: R²a PCA −0.25 vs full +0.45 (Δ +0.70, q = 0.0015), R²v −0.17 vs +0.02 (q = 0.039), F1 0.28 vs 0.33 (n.s.) — PCA-95 keeps the high-variance (piece-identity) directions and discards the emotion-relevant variance. Quadrant accuracy (fixed Ridge α = 10 on X_A) 0.25 piece / 0.62 loo.

**§4.4b (now §6.3) unique contributions (piece).** F1: MERT −0.053, CREPE −0.045, Essentia −0.051, DTW +0.036 (none significant); full set F1 0.302 (below P95). **§4.4d (now §6.5):** NEUTRAL_RADIUS 1.0 beats 0.75 on both columns for X_A and X_C (rule triggers — pending decision); raw ground truth F1 0.34 vs normalised 0.30.

**§4.9 (now §8.1) within-piece (piece-centred, LOPO).** Noise ceilings (split-half, centred): V 0.455 [0.088, 0.718], A 0.678 [0.520, 0.813]. Δvalence: Condition-only R² 0.047 (p 0.034), Interpretable 0.174 (p 0.006), MERT-full 0.175, MERT-PCA 0.134; **B-INC incremental ΔR² Interpretable +0.129 (q 0.037), MERT-full +0.128 (q 0.037), Interp + MERT-PCA +0.117 (q 0.037)**, MERT-PCA +0.077 (q 0.070). Δarousal: Condition-only 0.408, Interpretable 0.423, incremental +0.024 (p 0.044, q 0.070); the MERT sets add nothing. R²/ceiling (Interpretable) V 0.38, A 0.62.

**§5.1 (now §9.1; its head → §4.1) BH.** 180 tests (30 features × 2 targets × 3 transitions): 10 raw p < .05 vs 9.0 expected, **0 q < .05** → the condition-delta correlations are **not evidence; do not cite them as findings**.

**§5.1b (now §4.2 / §4.3b) dose-response (Page, 20 pieces).** Techniques: spectral flux (normalised) z +5.22, **vibrato extent +5.14, vibrato coverage +4.90**, LUFS +4.74, DTW deviation +4.59 (built in at MEC), DR +4.11, rolloff −2.85, **tempo log2-SD −2.21 and local tempo −2.06 (both decrease)**; vibrato rate +1.90 n.s.; portamento n.s. Audience: arousal z +4.11, valence +1.26 n.s. → with the repaired descriptor, **vibrato extent/coverage do escalate**; arousal (not valence) follows the manipulation.

**§5.4 (now §9.2) / §5.5 (now §10.1) cost of explainability (piece; negative = interpretable wins).** Read-out V r 0.25 / R² 0.12, A r 0.77 / R² 0.62. Interpretable F1 0.353, R²v 0.120, R²a 0.620 vs MERT-PCA 0.282 / −0.171 / −0.252 and MERT-full 0.328 / 0.016 / 0.446. Headline ΔR valence −0.181, ΔR arousal −0.140, ΔR² valence −0.104, ΔR² arousal −0.174, ΔF1 −0.071; vs MERT-PCA ΔR² arousal −0.872.

**§5.6 (now §10.2) sub-families (piece).** Removing Portamento raises F1 to 0.465 (loss −0.112, q 0.015) and removing Timbre to 0.424 (q 0.009) — consistent with portamento false alarms (under audit); removing Dynamics costs R²a 0.090 (q 0.11); Dynamics alone R²a 0.44; Vibrato alone F1 0.28, R²a 0.04.

**§6.1b (now §7.3b) probe / §6.2 (now §7.5).** Flux 0.86, F0 mean 0.80, centroid 0.73, HPCP entropy 0.73, pulse clarity 0.71, note rate 0.65, vibrato coverage 0.48, F0 range 0.43; LUFS 0.08; vibrato rate −0.02. α hit the grid edge in **34** outer folds. §6.2 (now §7.5) (piece): + all CREPE +0.057, + DTW +0.052, + Essentia −0.001, + 2 CREPE-unique −0.019. Master ablation table 138 rows / 9 tiers.

### Historical — Phase A run (2026-10-09; superseded by the Phase B run above)

**Guards (Phase A).** All passed: §4.0 (now §5.1) engine equivalence, §4.4b (now §6.3) lattice rebuild, §4.4d (now §6.5) label reproduction, §6.2b (now removed in C2), §4.9 (now §8.1) closed-form permutation equivalence, §5.5 (now §10.1)↔§5.4 (now §9.2) cross-check. 101 cells, 0 errors.

**Dummy floors fixed.** Uniform mean 0.272 (loo) / 0.2715 (piece); stratified 0.183 / 0.161; strict P95 0.320 (loo) / 0.327 (piece).

**MERT (A3) — false alarm.** The restore ran, but the embeddings are identical to the pre-fix run (the §6.1 (now §7.3) correspondence table and every MERT-based F1 match to 4 decimals). The transformers "newly initialized" warning was spurious; pre-audit MERT numbers were valid. The assert stays.

**Classification, piece (PRIMARY).** The pre-declared default GaussianNB is **below chance on every set** (F1-macro A 0.112, B 0.287, C 0.168, D 0.113); on X_C three classes collapse (HA_HV, LA_HV, Neutral recall 0); quadrant accuracy 0.25; Krippendorff α ≤ 0 for all labels except LA_LV (0.41). LOOCV for comparison: C 0.686. **Tier 4D (piece):** SVM-Linear balanced + inner-CV C beats NB deployed on X_A (+0.143, q=0.023), X_C (+0.156, q=0.005), Interp (+0.151, q=0.023); Interp + SVM balanced = 0.385 [0.327, 0.434] — the only configuration above the strict chance P95; NB balanced priors: no effect. → Decision (Phase B): nested model selection replaces the single default classifier; NB stays reported as "pre-declared Phase A default, failed under piece".

**Regression, piece (nested ridge).** Arousal R² — X_C 0.50, CREPE+Essentia 0.52, Essentia 0.40; valence R² ≈ 0 (max 0.23, CREPE+DTW). MERT-PCA: R²v −0.17, R²a −0.25; MERT-full: R²a 0.45 (Δ vs PCA +0.70, q=0.0015) → PCA-95 discards the arousal-relevant variance and keeps piece identity (a tool-taxonomy finding).

**Cost of explainability, piece (negative = interpretable wins).** Headline ΔR valence −0.215, ΔR arousal −0.054, ΔF1 −0.127; vs MERT-PCA ΔR² arousal −0.756.

**§4.9 (now §8.1) within-piece (piece-centred, LOPO).** Δarousal — Condition-only R² 0.41, Interpretable 0.47 (p=0.002), MERT-full 0.38, MERT-PCA 0.27. Δvalence — Condition-only 0.047 (p=0.034), Interpretable 0.20 (p=0.004), MERT-full 0.175, MERT-PCA 0.134. Within-piece SD: V 0.234, A 0.285 (scale 0–6). The "beats condition" difference was not yet tested (→ Phase B B-INC).

**§5.1 (now §9.1; its head → §4.1) BH (Phase A).** 0 q < .05 (12 raw p < .05) over the Phase A family (29 interpretable features then; the current §5.1 (now §9.1; its head → §4.1) runs 180 tests — see above).

**§6.1b (now §7.3b) grouped probe (Phase A).** F0 mean 0.80, spectral centroid 0.73, HPCP entropy 0.73, pulse clarity 0.59; loudness 0.08 (the α grid-edge count is 34 outer folds in the Phase B run — the earlier "32" is superseded).

**§2.3c recording level.** 0 LUFS outliers, 0 clipping, 21 noise-floor flags (15 excerpts inconclusive). EXG conclusive floor mean −66.2 dBFS vs MEC −71.4 / EXP −72.5; EXG LUFS +4.2 LU vs MEC. Facts from the author: 3 recording sessions with position markers; in EXG the performer deliberately played louder. Equality of gain across sessions is **not independently verified** (limitation); the 5th-percentile floor estimator is unreliable when an excerpt has no silence (→ Phase B B-NF). A session map is not available yet.

### Facts not changed by Phase B (Part 1 at N = 118 unless stated; N = 116 values in §11)

**Variance structure (audit).** 95% of valence and 91% of arousal variance across the 60 excerpts is **between-piece**; the condition explains **1% (V) / 4% (A)**. LOOCV (which keeps a piece's other two renditions in training) therefore mostly measures piece recognition; leave-one-piece-out and the within-piece analysis (§4.9 (now §8.1)) are the honest tests.

**Former "Headline 2" (escalation via dynamics & timbre, not vibrato) — withdrawn.** It came from the Phase A dose-response with the broken vibrato descriptor and the raw flux. In the Phase B run the repaired vibrato extent / coverage escalate strongly (z +5.14 / +4.90) next to normalised flux and loudness; only the vibrato *rate* stays n.s. (see §5.1b (now §4.2 / §4.3b) above).

**Normalization validation.** Raw grand mean V=2.97 / A=3.40 (arousal legitimately skews high); between-excerpt SD V=1.065 / A=0.956 (preserved); anchor cross-form |Δ| baseline **V=0.334 / A=0.289**; raw↔norm Pearson r = **0.996 (V) / 0.993 (A)**; **10/60 excerpts change class, 4/60 change quadrant** (the earlier "0/60 flips" claim was wrong). Per-rater min SD 0.51/0.54 → the σ-floor **never fires** (insurance, not active). **Conclusion (unchanged): normalization is a defensibility/robustness step — it confirms the signal is real, not a rater artifact; no material change in results.**

**Stimulus level.** The stimuli played to participants were **not loudness-normalized** (confirmed by the author): level differences between excerpts are part of the stimulus. §2.3c (Phase A) only checks that no recording was made at a clearly different gain/microphone distance.

**Cleaning (final, 2026-10-21 — supersedes the earlier "3 flagged, decision pending").** Screening v3 excludes **2 of 118** (`questionnaire_5_P012`: D1+D2, r = −0.20, MCD 18.4; `questionnaire_5_P020`: D1+D3, valence r = −0.57 / arousal r = +0.78, reversed valence scale). The old "3 flagged" came from the buggy v2 scripts. Kept: `questionnaire_1_P006` (extreme but consistent responder, r = 0.83, D4 only); sensitivity set with near-zero agreement and one flag: `questionnaire_3_P014`, `questionnaire_4_P010`, `questionnaire_5_P018`. Excerpt QC at N = 116: ICC(1,k) V 0.965 / A 0.953; median r_wg 0.60 / 0.59; arousal EXG < MEC in Bach_Adagio_S1 (−0.52), Mendelssohn_VC_Mvt2 (−0.18), La_Vita_E_Bella (−0.09). Full numbers in §11. **Exclusions applied in the notebook (§1.1) since 2026-11-03 — first run pending.**

---

## 5. Open issues

**PENDING DECISIONS (open, unchanged — for the author / supervisor):**
- **Closed by Phase C2 (applied 2026-11-20, not run):** `DTW_DEV_IN_INTERP` → the predictive / descriptive split (§4.1) · `CREPE_UNIQUE2` → the probe rule (`CREPE_UNIQUE_R2_MAX = 0.2`, §7.5) · the "all-positive" floor → superseded by the empirical null · `NEUTRAL_RADIUS` stays 0.75 · Fig 5 / the fixed α = 10 → §6.2 / §6.8 use the deployed nested ridge · `MERT_LAYERS` stays [5, 6, 7]. **Still open:** the report-only re-articulation code and the legacy FFT helpers (keep or remove).
- `DTW_DEV_IN_INTERP=True` vs the thesis CoE definition (condition-encoding features should be out of *both* sides of the cost of explainability; `False` gives a strictly condition-independent CoE).
- `CREPE_UNIQUE2` hard-coded `[f0_range, vibrato_rate]` in §6.2 (now §7.5) — read off the pre-Phase-B probe; in the Phase B probe F0 range is well represented (R² 0.43), vibrato rate is not (−0.02).
- The "all-positive" chance floor (≈ 0.322) cited in the thesis is not computed in the notebook. → **Superseded by the §4.5c (now §5.3 / §6.9) empirical null** (P95 0.370 / 0.459); drop it from the thesis.
- The §4.4d (now §6.5) NEUTRAL_RADIUS rule triggers: under `piece`, r = 1.0 beats 0.75 on both columns for X_A and X_C (Phase B, N = 118).
- Fig 5 (§4.7 (now §6.8)) omits the MEC points (`cc` holds only EXP/EXG) / figure dedup.
- §4.2 (now §6.2) / §4.7 (now §6.8) use a fixed Ridge α = 10 while the rest of the pipeline uses nested ridge.
- **NEW (C1): whether to change `MERT_LAYERS`** — decided by the author by hand after reading §4.4f (now §7.2) (Stage A suggestion, Stage B profile vs the empirical null, nested stability); nothing is applied automatically. **C1 run outcome:** the shortcut is in every layer, no layer beats the empirical null, nested selection is unstable, and the suggestion W09-11 is not worth adopting → **recommended: keep [5, 6, 7]** (record the author's decision).

**Addressed in C1 (run 2026-11-16 — outcome in §4):**
- **Chance floor too low for the deployed classifier** — the Dummy floors ignore NestedSelect's selection procedure; in the B3 run 25 i.i.d. noise columns scored piece F1 0.354 > the Dummy strict P95 0.323 (0.468 at C = 0.001). → §4.5c (now §5.3 / §6.9) empirical null; every reported classification F1 is read against it (`emp_null_readout.csv`, §5.6 (now §10.2), §7.1 (now §11.1)). **Outcome:** only 4 / 52 (3 distinct, all with the timing block) beat it; the final form (exact widths, 1000 draws, a primary family, BH) is a C2 obligation (applied in C2; Phase C3: 1000 draws at d = 7 / 17, 600 at the other widths).
- **Per-MERT-layer study** (the supervisors' request; ON HOLD since 2026-11-03) → §2.1c + §4.4f (now §7.2), diagnostic only. **Outcome:** the shortcut is in every layer; the information about the condition exists but is masked; no layer closes the gap to the interpretable layer (§4).

**Manual audit of 2026-11-05 — confirmed and FIXED in Phase B2 (§10):** portamento false alarms (crepe's Viterbi decoder turns every jump into a ≤ 220 c/frame ramp; Méditation MEC D6→D5 and A5→A4 were "glides" of 62.5 ms) · notes merged with slides and repeated notes merged (Méditation EXP "D6" = C♯6 + slide; Mozart VC4 17 / 16 / 19 notes vs 19 in the score) · real narrow vibrato rejected by the FFT presence / 5 c rules (Méditation EXP D5, B4). The Phase B worry "note splitting by wide vibrato" is covered by the B2 plateau tolerance + merge rule (checked via close splits and V2–V4).

**Phase B3 decision (2026-11-10):** the B2 re-articulation detector is **disabled** for every feature (`REART_ENABLED = False`) and documented as a negative result — its splits followed the vibrato amplitude modulation (§10, "Phase B3").

**Known limitations after B3 (documented, NOT fixed):** repeated notes without a silence count as one (Mozart_VC4 ≈ 17 / 16 / 20 vs 19 in the score; §2.6 (now removed in C2) V4 is INFO) · slow slides can create short "ramp plateaus" (a glide counted twice, e.g. B5→C6→C♯6 in Meditation_Thais_EXG) · the 80 ms glide minimum rejects the Méditation EXP C♯6→D6 (rise 75 ms) · residual MEC detections (V6: 5 MEC — the glides look like position shifts) · the tempo log2-SD / pulse clarity / log-attack time / `es.Key` items below.

**Audit findings still OPEN (documented, not changed in B2 or B3):**
- integrated LUFS Essentia − pyloudnorm = **+3.055 LU on all 6 audited excerpts** (Essentia duplicates the mono file to 2 channels — constant offset; decide whether to subtract 3.01 LU or report as is);
- `es.Key` gives **G minor for Mozart_VC4_EXP** (D major for the other five audited excerpts);
- Méditation `timing_tempo_logsd` **MEC 0.256 > EXP 0.158 > EXG 0.135** and pulse clarity lowest at MEC (✗ vs the metronomic MEC instruction — the tempogram reacts to articulation);
- log-attack time **not sharper in the spiccato Mozart EXP** (✗); LAT is computed on Essentia waveform onsets (`OnsetRate`), not on note onsets;
- DTW / duration ratio computed over the whole files incl. silence;
- DR gate sensitivity (6 of 9 variants ρ < 0.9);
- §2.3c flatness noise-floor test inconclusive for 60/60.

**Decisions taken (2026-11-03):** excerpt QC (ICC, r_wg, bimodality coefficient, Kruskal–Wallis, crossed additive model) stays **outside** the notebook (offline scripts) · per-MERT-layer caching is **ON HOLD** (*superseded 2026-11-16 by Phase C1: per-layer cache + diagnostic study; `MERT_LAYERS` unchanged*) · the timing pattern is explained by the author (EXG faster / more virtuosic in fast pieces, more time in slow ones → local tempo and duration ratio are piece-dependent; tempo log2-SD and pulse clarity still checked by eye in §2.6 (now removed in C2)) · block weighting `'equal'` deployed for the multi-block MERT-full regression heads (tier 4E descriptive).

**Other open items:**

1. **"8 emotions" is mechanically 5 classes.** `QUADRANT_EMOTIONS` maps each quadrant to a *set*, so pairs (Tenderness≡Peacefulness, Power≡Joyful, Sadness≡Nostalgia) are byte-identical columns. **Decision taken: report honestly as 5 classes** (4 quadrants + Neutral). Alternative (rebuild `Y` from raw multi-label tags) remains available. Good supervisor topic.
2. ~~**Neutral radius un-tuned.**~~ **Measured in §4.4d (now §6.5)** — the 0.5/0.75/1.0 sweep reports macro-F1 **with and without** Neutral. In the Phase B run r = 1.0 wins on both columns (see PENDING DECISIONS).
3. **Figure dedup still pending** (partial cleanup only): a duplicate VA scatter still sits alongside the circumplex; check for any remaining per-class-F1 heatmap/radar and ablation-figure redundancy and prune (keep circumplex, P/R/F1 bars, PCA scree).
4. **DTW block size / DTW-dev in `df_interp`.** In the Phase B run the §2.4b gate forwarded 10 columns (7 ref-free + 3 dev) — inside 3–10, no warning. The `DTW_DEV_IN_INTERP` question is listed under PENDING DECISIONS.
5. ~~**Participant exclusions decided but not applied.**~~ **Applied in §1.1 (2026-11-03)**: `EXCLUDED_PARTICIPANTS` dropped before §1.1b, asserts 116 participants / 2,320 rows / 60 excerpts; first run pending. Expected Part 1 results are in §11; `Schindlers_List_EXP` moves HA_LV → LA_LV.
6. **Statistics hygiene:** the ablation study (§4.4b (now §6.3)) carries bootstrap CIs, paired permutation tests and **BH-FDR within each tier × protocol** (now over F1, r and R²); **§5.1 (now §9.1; its head → §4.1) now applies BH-FDR over all its tests** (Phase A); §5.1b (now §4.2 / §4.3b) (dose-response) still wants the same treatment or a pre-registered hypothesis set — not yet approved. Keep phrasing classifier-set comparisons as "across the classifiers tried," not population inference. **The lattice must stay descriptive** — selecting the winning subset on the same CV and then reporting its score is exactly the nested-selection error.
7. **Feature hygiene — repaired in Phase B (run on N = 118; vibrato extent / coverage now escalate, see §4; remaining doubts under audit in §2.6 (now removed in C2)).** The Phase A descriptors were broken: `crepe_portamento_count` was a raw count; `crepe_vibrato_rate_hz` was bandpass-bounded to [4,8] Hz *and argmax-ed inside that same band* (it could never report "no vibrato"); vibrato/portamento ran on concatenated voiced frames. **Pre-Phase-B vibrato nulls** (the n.s. dose-response) are therefore evidence about the descriptor only. Phase B (B8/B9) analyses time-contiguous notes, searches [3,10] Hz with a presence test, splits extent from coverage and reports portamento as rates/shares — a vibrato null from the next run is a **musical result at this N**, with the remaining caveats. **The 2026-11-05 audit then found the Phase B detectors faulty (§10, Phase B2)**; B2 replaces them: plateau notes + re-articulation splits, a windowed 4–8 Hz sinusoid fit on notes ≥ 500 ms (R² ≥ 0.50, half-extent ≥ 3 c), and a Viterbi-aware glide rule (rise 80–500 ms, ≥ 50 ms longer than crepe's forced ramp). The Phase B vibrato escalation (extent z +5.14, coverage +4.90) must be re-read in the next run before the thesis keeps it. B2 detectability limits: a glide is only seen between two plateau notes; vibrato on notes < 500 ms or with half-extent < 3 c is invisible; rates are clamped to 4–8 Hz. **Phase B3:** the re-articulation splits turned out to follow the vibrato and are disabled (a repeated note without a silence is one note); a window whose best rate is on the grid edge is not vibrato; a plateau merge re-checks the 80 c range.
8. ~~**LOOCV leaks piece identity.**~~ **Addressed in Phase A:** leave-one-piece-out is now the PRIMARY protocol everywhere (§4.0 (now §5.1)), LOOCV is the optimistic secondary, and §4.9 (now §8.1) removes the piece entirely (within-piece prediction). Quote the `piece` number for any claim about unseen material.
9. **`LA_HV` has n=5 from only 2 pieces** (La_Vita ×3, Meditation_Thais MEC+EXP): under `piece` it is learned from one piece at a time, and inner-CV tuning is skipped for it in many folds.
10. **Recording sessions.** The session map is **not provided yet** (`SESSION_MAP_CSV`, columns `excerpt_id, session`; §2.3c prints "session map not provided" and skips until it is uploaded). Gain equality across the 3 recording sessions is **not independently verified** (the performer reports playing louder in EXG on purpose).
11. **Phase C (deferred idea) — augmentation.** The supervisor's suggestion: augment the minority positive-valence classes (pitch shift ±1 semitone, tempo ±5%, never combined), with augmented copies used **only in training folds, and only when their source piece is itself in that training fold** (test = originals only, gate/PCA fitted on originals, scoring/permutations on the 60 originals). Caveats: it cannot add piece diversity (LA_HV comes from only 2 pieces); it perturbs studied variables (F0 mean, tempo, onset rate) while keeping labels; the copies' labels were never rated; phase-vocoder artefacts affect Essentia timbre features. **Decision:** evaluate class balancing (A5, tier `4D-balancing`) first; augmentation later as an ablation tier.
12. **Not approved yet:** SHAP fitted in-sample (§5.2 (now §9.3)) · §5.3 (now §7.4) embedding-distance sign · §1.4 (now §4.3a) mixed model · BH in §5.1b (now §4.2 / §4.3b).
13. **Known:** §5.5 (now §10.1)'s headline CoE is a best-of over two black boxes (max(MERT-PCA, MERT-full)) — conservative; the two separate gaps are in `coe_gaps.csv`. (The README now points to `docs/thesis.md`; Fig 5 is under PENDING DECISIONS.)
14. **Parameter ablations still pending:** CREPE step_ms (re-extraction), the CREPE voicing threshold and the B2 descriptor constants as an ablation tier (cheap from `crepe_tracks_v2.npz` + `rms10_v3.npz`; §2.2 prints report-only sensitivities and the MEC calibration grids), Essentia frame sizes, the tempogram window / tempo prior; per-MERT-layer is now the C1 diagnostic §4.4f (now §7.2) (no longer pending).
15. **Runtime.** The Phase B run took ≈ 97 min on a 4-CPU T4 session (Phase A: 75.5 min). The B2 run added tier 4E and the §2.6 (now removed in C2) audit. **B3 adds** §4.4e (now §7.1) (≈ 3–6 min: NestedSelect for the interpretable layer and the random control under both protocols, the rest memo hits; nested ridge for 7 blocks × 2 targets × 2 protocols; the validation curve and 10 000 label shuffles are seconds) and §5.6b (now §10.3) (≈ 10–15 min, dominated by the pair tier: 29 NestedSelect + ridge fits under `piece` plus 28 × 5 × 2000 paired permutations; ≈ 3× that with `RUN_PAIR_ABLATION_LOO = True`; probes / correlations seconds). §2.2's report-only detector evaluation adds seconds. **The B3 run took ≈ 127 min. C1 adds** §2.1c ≈ 0.5–1 min (one MERT forward pass per excerpt; B3 extracted `emb_mert` in 26 s), §4.4f (now §7.2) ≈ 40–55 min (Stage A 1–2 min, Stage B 35–45 min with both protocols / 10–15 min `piece` only, nested 3–10 min) and §4.5c (now §5.3 / §6.9) ≈ 50–70 min (2 × 100 NestedSelect `piece` runs; `RUN_NULL_LOO` ≈ +3 h) → ≈ 3.5–4 h in total; every C1 result is cached for reruns.
16. **Thesis-side open items (2026-10-30):** recruitment channels for the participants (TODO in Ch. 7 §7.4) · mean duration per condition from `piece_durations.csv` (TODO in §7.2.3) · intended quadrant of SQ8_exrpt1 (Q3 but tagged Tension/Power) and SQ8_exrpt3 (Q4 but tagged Tenderness) in the dataset metadata — check before trusting the "Intended" column of Table 7.1 · movement/passage of each work in Table 7.1 · the 6 under-18 participants (shown as "Under 18" in Table 7.5 — ask the supervisor whether consent needs a mention) · author name, committee, NTUA logo, dedication, spelling of Λυμπεράτος/Κανταρέλης, year of `lyberatos2026witheflow` (arXiv 2506 = 2025).
17. **MERT generalisation / family overlap (B3):** answered by §4.4e (now §7.1) (capacity overfitting vs piece-identity shortcut) and §5.6b (now §10.3) (triangulated overlap) under pre-declared rules — read their verdicts before writing Ch. 9. B3 run: the **piece-identity shortcut fired** (MERT-full, MERT-PCA25), capacity overfitting did not; the C1 layer study (§4.4f (now §7.2)) asks whether another depth carries less piece identity and more condition information.

### Limitations, future work and remaining code obligations (2026-11-17)

**Limitations — consolidated (2026-11-17; for Ch. 9 / Ch. 10; supersedes the scattered "known limitations" lists above, which stay as history).**

*Design and data*
- **Scale of the corpus.** One performer, one violin, home recordings in 3 sessions; 20 pieces × 3 renditions (N = 60). Between-piece variance dominates (95 % V / 91 % A), and every representation learns the piece identity to some degree (same-piece nearest neighbour 28–100 %).
- **Joint manipulation of the techniques.** The techniques were manipulated *together* (MEC → EXP → EXG), so they co-vary within a piece (common cause). Per-technique attribution is not identifiable from this design (§5.6b (now §10.3)).
- **Classes.** 5 classes (the 8 tags collapse); LA_HV n = 5 from 2 pieces, Neutral n = 7. The labels are effectively single-label (support sums to 60) but are modelled with binary relevance; under piece, LA_HV and Neutral have recall 0.
- **Chance level.** Classification beats the pipeline's empirical null only with the reference-free timing block. The Dummy floor understates chance for a nested-selection pipeline.
- **Empirical null.** 100 draws (p ≥ 0.0099); two widths (10 / 25), mapped by nearest width and extrapolated beyond 25.
- **Recording level.** The stimuli were not loudness-normalised; gain equality across the sessions is not verified; the session map is missing.
- **Questionnaire.** Arousal anchors share words with tags; partly felt framing; fixed piece order; EXG has about half the raters of MEC / EXP.

*Descriptors*
- **Repeated notes** without a silence count as one (Mozart_VC4 17 / 16 / 20 vs 19 in the score). There is no re-articulation detection: the energy envelope confuses bow re-attacks with the vibrato amplitude modulation (B3 negative result).
- **Slow slides** can create "ramp plateaus" (a glide counted twice; extent underestimated). The ±40 c plateau tolerance absorbs slide ends (the EXP C♯6→D6 is measured as 77 c < the 80 c glide minimum). The glide floor is 80 c / 80 ms.
- **Vibrato.**
  - Invisible on notes < 500 ms or < ±3 c; rates clamped to 4–8 Hz.
  - The extent is conditional on detection (a single MEC false window gives extent ≥ 3 c).
  - Declared median substitutions for 19 excerpts.
- **CREPE.** 10 ms steps; the Viterbi slew limit is 220 c per frame, so every jump is rendered as a ramp of ≥ ≈ 55 ms per octave (handled by the glide excess rule).
- **Tempogram features** (local tempo, tempo log2-SD, pulse clarity) are unreliable: they fail the audit (6 / 20, 4 / 20), and the local-tempo direction contradicts the note rate.
- **Essentia and DTW details.**
  - LAT on Essentia waveform onsets (✗ for spiccato).
  - `es.Key` unstable (Mozart_VC4_EXP read as G minor).
  - LUFS +3.01 LU dual-mono offset.
  - DTW / duration ratio over whole files including silence.
  - DR gate sensitivity; flatness noise-floor test inconclusive.
- **DTW-dev features** are 0 / 1 at MEC by construction (they encode the condition) and are in the interpretable layer (`DTW_DEV_IN_INTERP = True`).

*Statistics*
- **§5.1b (now §4.2 / §4.3b)** is not BH-corrected in the notebook. By hand, tempo log2-SD, local tempo and note rate fail BH.
- **MERT PCA** is fit transductively on all 60 excerpts (label-free).
- **§5.5 (now §10.1) headline CoE** is a best-of over two black boxes (conservative).
- **LOPO label-composition artefact:** noise falls *below* chance at C ≥ 0.01 (0.14–0.17), so "below chance" alone is not evidence of anti-learning. At C = 0.001 a balanced linear SVC is a class-mean-difference (prototype) rule whose threshold follows the fold's label composition, which inflates the variance of the score (noise up to 0.468).

**Future work (2026-11-17).**
1. **Factorial recordings.** One technique at a time (vibrato only, dynamics only, …) to break the common cause and attribute effects per technique.
2. **A larger corpus.** More performers, instruments and pieces: more piece diversity, a weaker piece-identity shortcut, coverage of LA_HV.
3. **Piece-centred MERT within pieces.** Subtract each piece's 3-rendition mean (label-free) and test MERT inside §4.9 (now §8.1). A2 shows the information is there (0.90). This does not help prediction on unseen pieces, because it also removes the between-piece emotion.
4. **MERT layers.** Follow up only as a pre-registered test on new data (the arousal sensitivity of layers 3–6).
5. **Re-articulation detection with a second cue** (bow-noise burst / spectral flux at the attack, a CREPE confidence dip), validated on more than one excerpt.
6. **Tempo from note onsets** instead of the tempogram.
7. **Vibrato dynamics:** onset delay and the evolution within a note (depth / rate trajectories).
8. **Modelling:** multiclass / ordinal / continuous-first; augmentation (the Phase C idea) only after class balancing.
9. **Valence:** harmonic / mode features to explain it.
10. **Parameter ablations:** CREPE step / voicing, the B2 constants, Essentia frame sizes, the tempogram window / prior.

**Remaining code obligations — Phase C2 candidates (2026-11-17, in priority order).**
> **Status 2026-11-20 (Phase C2 applied, NOT RUN):** 1 ✅ (exact-width primary null, §5.3 / §6.9) · 2 ✅ (BH in §4.2) · 3 ✅ ("keep flagged" + §6.3 sensitivity rows) · 4 ⏸ ON HOLD · 5a ✅ (§7.1 text) · 5b ✅ (§10.3 test 1) · 6 ✅ (`MERT_LAYERS` stays [5, 6, 7]) · 7 ✅ (§2.6 removed; re-articulation / FFT helpers left in place — pending) · 8 ✅ except the helpers · 9 open (the C2 run) · 10 open (optional). The list below is the 2026-11-17 text.

1. **Final form of the empirical null.**
   - A pre-declared primary family: headline A–D, interpretable (30), DTW, CREPE + DTW, and the multiclass model of item 4.
   - An exact-width null per primary set, with `NULL_N_DRAWS = 1000` for that family (cached), and BH within the family. *(Phase C3: 1000 draws at d = 7 / 17, 600 at the other widths.)*
   - The 100-draw d = 10 / 25 map stays for the descriptive lattice.
2. **BH in §5.1b (now §4.2 / §4.3b)** (q column; stars from q).
3. **Tempogram features:** either remove `timing_local_tempo_bpm`, `timing_tempo_logsd` and `timing_pulse_clarity` from the models, or keep them flagged "unreliable" and out of the attribution claims. Optionally add a note-onset tempo.
4. **Multiclass classifier:** exactly one of 5 classes per excerpt (multinomial LogReg / SVC with balancing inside NestedSelect; new tier), with a confusion matrix, read against the empirical null.
5. **Small fixes:**
   - The §4.4e (now §7.1) text "near-constant model" → "class-mean-difference (prototype) classifier whose threshold depends on the fold's label composition".
   - §5.6b (now §10.3) test 1 should require I > `OV_I_TOL` (0.02), to match the reading band. It counted Dynamics ↔ DTW-dev with I = +0.001; the overlap still holds through tests 2 and 3.
6. **Record the `MERT_LAYERS` decision** (recommended: keep [5, 6, 7]) and close the C1 pending decision.
7. **Remove the §2.6 (now removed in C2) manual audit** before the final run, and decide whether the report-only re-articulation and legacy FFT helpers stay.
8. **Pending decisions still open:** `DTW_DEV_IN_INTERP`, `CREPE_UNIQUE2`, the "all-positive" floor (superseded by the empirical null — remove it from the thesis), `NEUTRAL_RADIUS`, Fig 5 / dedup, the fixed α = 10 in §4.2 (now §6.2) / §4.7 (now §6.8).
9. **Final full run**, then transcription into the docs and Ch. 9.
10. **Optional:** piece-centred MERT in §4.9 (now §8.1) (future-work item 3).

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

- **Phase C3 constants / names (2026-11-20):** §0.3 `NULL_N_DRAWS_PRIMARY = {7: 1000, 17: 1000}`, `NULL_N_DRAWS_PRIMARY_DEFAULT = 600`, `NULL_CACHE_INPUT_DIR` · §5.3 `null_primary_target`, `_NULL_PRIMARY_HASH_N_DRAWS = 1000` (the frozen hash field), `NULL_PRIMARY_BUDGET_MIN = 240`, `emp_null_ref` / `emp_null_mark` / `emp_null_dim` / `emp_null_p` on the one rule (kinds exact / borrowed / extrapolated / not drawn), `EMP_NULL_PRIMARY[…]['target']` · §6.9 `null_target` · §7.4 `SHIFT_PIECE_LABEL` · §10.2 `alone_emp_null`.
- **Phase C2 constants / names (2026-11-20):** §0.3 `TEMPOGRAM_UNRELIABLE`, `NULL_N_DRAWS_PRIMARY = 1000` (C3: a per-width dict), `CREPE_UNIQUE_R2_MAX = 0.2`, `bh_fdr` (moved from §5.1) · §4.1 `INTERP_PRED_COLS` (27), `INTERP_DESC_COLS` (30), `INTERP_FLAG_REASON`, `flag_label`, `top_feature_note`, `pred_short`, `X_INTERP` (now 27 columns) · §4.2 `df_dose` (+ `q`, `stars_q`, `flag`), `DOSE_FAMILY_N` · §4.3a `FRIEDMAN_RES` · §4.3b `DOSE_EMO` · Part 4 `MANIP_CHECK` · §4.3c `CLASS_SHARE_BY_COND` · §5.2 `df_dummy_floors` · §5.3 `PRIMARY_NULL_FAMILY`, `PRIMARY_NULL_ALIASES`, `PRIMARY_NULL_WIDTHS`, `NULL_PRIMARY_CACHE`, `NULL_PRIMARY_CHUNK = 50`, `NULL_PRIMARY_BUDGET_MIN = 480`, `df_null_primary`, `EMP_NULL_PRIMARY`, `NULL_PRIMARY_COMPLETE`, `emp_null_ref`, `emp_null_mark` · §6.2 `REG_INTERP`, `REG_DEPLOYED`, `REG_ROLE` · §6.3 tier `1b-tempogram-sensitivity`, `TG_DROP`, `df_abl_tempogram(_deltas)`; ablate rows `emp_null_p/_p95/_d/_kind`, `beats_emp_null` · §6.6 `kripp_alpha`, `df_base` (+ empirical-null columns) · §6.8 `PRED_QUADS` (MERT-full, deployed), `PRED_QUADS_INTERP`, `df_quad_acc`, `QUAD_N_PERM = 1000` · §6.9 `emp_readout_rows`, `df_emp_null_primary` · §7.3b `PROBE_R2` · §7.4 `df_sh_tests`, `df_sh_explore`, `SHIFT_TRANS` · §7.5 `CREPE_UNIQUE`, `CREPE_SUBST_COUNTS`, tier `7.5-mert-plus`, `df_block` (new layout) · §9.2 `READOUT_ORDER_CLS / _REG` · §9.3 `SHAP_COLS`, `SHAP_LABELS`, `df_shap_vs_beta` · §11.1 `df_claims`. Removed: `DTW_DEV_IN_INTERP`, `RUN_MANUAL_AUDIT`, `AUDIT_EXCERPTS`, `CREPE_UNIQUE2`, `PIVOT_F1` / `PIVOT_HL`, `default_rec` is now print-only.
- **Scale 0–6**, `VA_MID=3.0`, `NEUTRAL_RADIUS=0.75`, `MERT_LAYERS=[5,6,7]`.
- `df_long` — long-format per-rating (parse, §1.1), **screened** (`EXCLUDED_PARTICIPANTS` dropped in §1.1, `N_PARTICIPANTS_RAW` = 118); gains `valence_norm/arousal_norm` in §1.1b.
- **Block weighting (§0.3 / §4.0 (now §5.1) / §4.4b (now §6.3)):** `REG_BLOCK_WEIGHTING = 'equal'` · `block_weights(sizes)` · `BlockScaler(sizes)` · `block_sizes=` in `nested_ridge_predict` / `select_ridge_alpha` / `NestedRidgeOperator` / `_ridge_pipe` · `_abl_reg_kw(names, weighting=None)` · `ablate(reg_block_sizes=…, block_weighting=…)` · `abl_vs_ref(…, pcols=…)` · `abl_alpha_edge_report` · `df_abl_bw` (tier 4E); ablation rows carry `block_weighting`, `n_alpha_edge_val/aro`, `n_outer_folds`.
- **§2.6 (now removed in C2) audit (temporary):** `RUN_MANUAL_AUDIT`, `AUDIT_EXCERPTS`, everything else `_aud_*`; outputs only in `OUTPUT_DIR/manual_audit/`.
- `df_agg` / `df_all` — per-excerpt aggregated. **`valence_mean/arousal_mean` are now the NORMALIZED means** (ground truth); `valence_mean_raw/arousal_mean_raw` kept in parallel.
- `emotion_labels` — VA-quadrant-derived (from normalized means); **this becomes `Y`**. `top_tags` — raw tags, descriptive only (§1.5 (now §4.3c)).
- `Y` (60×N), `mlb.classes_` = class names; `df_expr` — modelled-excerpt frame.
- **Forwarding:** `unsup_forward_select(...)`; `MERT_FORWARD` / `CREPE_FORWARD` / `ESSENTIA_FORWARD`; forwarded frames `emb_mert_fwd`, `df_crepe_fwd`, `df_ess_fwd`. Full blocks on disk: `df_mert`, `df_crepe`, `df_essentia`.
- **Assembly:** `FEAT_A/B/E/DTW/C/D/F`, `X_A/X_B/X_C/X_D/X_F`, `X_MERT_FULL`; `STRICT_BLOCKS=('MERT','Essentia')` (DTW non-strict but a missing row warns loudly); `bfv(...)` builder.
- **Interpretable frame & labels** used by Part 5: `df_interp` + `INTERP_FEAT_LABELS` (CREPE + forwarded Essentia + forwarded DTW, built in §5.1 (now §9.1; its head → §4.1); §5.1b (now §4.2 / §4.3b) / §5.2 (now §9.3) / §5.4 (now §9.2) / §5.5 (now §10.1) all derive their technique set from `INTERP_FEAT_LABELS ∩ df_interp.columns`).
- Part 5 outputs: `df_dose` (§5.1b (now §4.2 / §4.3b)) · SHAP importances (§5.2 (now §9.3)) · read-out coefficients (§5.4 (now §9.2)) · CoE table (§5.5 (now §10.1)).
- Excerpt IDs `PieceName_ConditionCode`; participant IDs `questionnaire_{n}_P{idx:03d}` (alias `S{n}_P{idx:03d}`).
- `OUTPUT_DIR` — all figures/CSVs. Skip stubs at §2.4: madmom, MusiCNN (DTW is active).
- **Timing/DTW constants:** `DTW_SR=22050`, `DTW_HOP=512`, **`DTW_TG_WIN=151`** (≈ 3.5 s), log-normal tempo prior centred on 100 BPM within 40–200 BPM, interior tempogram frames only (≥ 20), chroma-CQT alignment; `DTW_REFFREE` (= the `timing_*` family) / `DTW_DEV` (the `dtw_*` family) name the two families; `TIMING_DIAGNOSTIC=['timing_octave_jump_frac']`.
- **Phase B3 constants / names:** `REART_ENABLED=False` (HELPERS:descriptors; the re-articulation constants below are report-only), `segment_notes(_detail)(…, reart_enabled=None)`, `vibrato_window_reason`, the window `edge` / `reason` fields, `CREPE_DIAG` `n_edge_windows` / `n_edge_only`, `CREPE_NOTE_COUNTS`, `CREPE_NOTECOUNT_FRIEDMAN`, `CREPE_REART_REPORT` (§2.2); §4.4e (now §7.1) `MG_GAP_MARGIN=0.10`, `MG_CURVE_RISE=0.05`, `MG_NN_FOLD=3`, `MG_N_PERM=10000`, `MG_RANDOM_SEED=0`, `MG_X_INTERP`, `MG_VERDICT`, `df_mg_gap` / `df_mg_curve` / `df_mg_nn`; §5.6 (now §10.2) `df_abl_commonality`; §5.6b (now §10.3) `RUN_PAIR_ABLATION_LOO=False`, `OV_I_TOL=0.02`, `OV_PROBE_MIN=0.30`, `OV_RHO_MIN=0.40`, `df_ov_pairs` / `df_ov_probe(_fam)` / `df_ov_corr(_fam)`; tier `3-subfamily-pairs`.
- **Phase B constants (§0.3 / §2.2 / §2.3 / §2.3c / §4.0 (now §5.1)):** `NC_N_SPLITS=200`, `SELECT_C_GRID=logspace(-3,2,6)` → 13 `SELECT_CONFIGS`, `MERT_REG_REPR='full'`, `SESSION_MAP_CSV`, `OLD_CACHE_NAMES`; CREPE descriptors (**Phase B2**): `OCT_WIN=10`, `OCT_DEV_RANGE=(1000,1400)`, `TREND_WIN=15`, `PLATEAU_TOL_CENTS=40`, `PLATEAU_MAX_RANGE_CENTS=80` (B3: also re-checked after a merge), `PLATEAU_MERGE_CENTS=50`, `PLATEAU_MERGE_GAP_MS=50`, `REART_DIP_DB=6` (sensitivity 3/6/9; detector disabled since B3), `REART_PRE_MS=150`, `REART_POST_MS=60`, `ENV_WIN_MS=25` (rectangular), `NOTE_MIN_MS_TIMING=60`, `VIB_MIN_NOTE_MS=500` (sensitivity 500/650/800), `VIB_WIN_MS=500`, `VIB_HOP_MS=250`, `VIB_RATE_BAND=(4,8)`, `VIB_RATE_STEP=0.05`, `VIB_FIT_R2_MIN=0.50`, `VIB_MIN_EXTENT=3.0`, `CREPE_VITERBI_MAX_CENTS_PER_FRAME=220`, `GLIDE_DUR_MS=(80,500)`, `GLIDE_MIN_EXCESS_MS=50`, `GLIDE_INTERVAL=(80,2000)`, `GLIDE_SIGN_MIN=0.8`, `GLIDE_RISE_SCALE=1.25`; legacy / report only: `BOUNDARY_LAG=5`, `BOUNDARY_CENTS=70` (also the close-split threshold), `VIB_BAND=(3,10)`, `VIB_NFFT=4096`, `VIB_PEAK_HALFWIDTH=0.5`, `VIB_PRESENCE_MIN=0.30`, `VIB_LEGACY_MIN_EXTENT=5`; Essentia `DR_ABS_GATE_DB=-60`, `DR_REL_GATE_DB=40`; §2.3c `FLAT_MIN=0.2` (0.1/0.2/0.3), `FLAT_BAND_HZ=(50,8000)`, `FLAT_RMS_PCTL=20`, `FLAT_MIN_FRAMES=20`, `SESSION_CONFINED_FRAC=0.80`; §4.0 (now §5.1) `NMS_N_JOBS=os.cpu_count()`, `NMS_TIE_TOL=1e-12`.
- **Caches (Phase B `_v2` + Phase B2 `_v3` + Phase B3 `_v4` + Phase C1):** `emb_mert_v2.npz`, **`emb_mert_layers_v3.npz`** (C1, §2.1c — 13 × 768 per excerpt; a new name, nothing retired), `crepe_tracks_v2.npz`, `rms10_v3.npz`, `feat_crepe_v4.csv`, `essentia_features_v2.csv/.npz`, `rms_frames_v2.npz`, `dtw_features_v4.csv`, **`null_f1_draws.csv`** (C1, §4.5c (now §5.3 / §6.9)) and **`mert_layers/mert_layer_nested.csv` + `mert_layer_nested_pred.csv`** (C1, §4.4f (now §7.2)) — the C1 caches are reused only when their settings hash / shape checks match; §2.5 asserts no `CACHE_*` path carries a name in `OLD_CACHE_NAMES` (which now includes `feat_crepe_v2.csv`, `dtw_features_v2.csv`, `feat_crepe_v3.csv`, `dtw_features_v3.csv`) and that the v4 names are in use; the C1 cells assert their own names are not retired. §4.4e (now §7.1) writes to `OUTPUT_DIR/mert_diagnostics/`, §4.4f (now §7.2) to `OUTPUT_DIR/mert_layers/`.
- **Phase C1 constants / names:** `RUN_MANUAL_AUDIT=False` (§2.6 (now removed in C2)) · §2.1c `MERT_N_HIDDEN=13`, `extract_mert_layers`, `emb_mert_layers`, `CACHE_MERT_LAYERS` · §4.4f (now §7.2) `RUN_MERT_LAYER_STAGE_B=True`, `RUN_MERT_LAYER_NESTED=True`, `MERT_LAYER_PROTOCOLS=('piece','loo')`, `ML_N_PERM_COND=1000`, `ML_SUGGEST_TOL=0.05`, `ML_TIER='6-mert-layers'`, `ML_DEPLOYED='W05-07'`, `ML_CANDIDATES`, `ML_SET_NAME`, `ML_NESTED_CLF='SVM-Linear C=1'`, `ML_SUGGESTED`, `ML_NESTED_SCORES`, `df_mert_layer_a/_b/_deltas/_nested/_nested_freq`, `df_mert_layer_study`, `ml_plot_profile` · §4.5c (now §5.3 / §6.9) `NULL_N_DRAWS=100`, `NULL_DIMS=(25,10)`, `NULL_SEED=0`, `NULL_PROTOCOLS=('piece',)`, `RUN_NULL_LOO=False`, `NULL_CACHE`, `EMP_NULL_F1`, `emp_null_dim`, `emp_null_p`, `df_null_draws`, `df_emp_null_readout`, `df_mert_layer_study_null` · §5.6 (now §10.2) `alone_emp_p`; tier `6-mert-layers`.
- **Phase A constants:** `SEED=42`, `N_DUMMY_REPEATS=200`, `WP_N_PERM=500` (§0.3); `PRIMARY_PROTOCOL='piece'`, `PROTOCOLS=('piece','loo')`, `RIDGE_ALPHAS=logspace(-2,4,13)`, inner `GroupKFold(5)`, `DEFAULT_CLF_NAME='NestedSelect'` since Phase B (`NB_CLF_NAME='NaiveBayes'` = the Phase A default) (§4.0 (now §5.1)); `TUNE_GRID_C={'C': logspace(-3,2,6)}` (§4.1 (now §6.1)); probe `ALPHAS=[1,10,100,1000,10000]` (§6.1b (now §7.3b)); MERT cache `emb_mert_v2.npz`; `OUTPUT_DIR='/kaggle/working/pipeline_outputs'`.
- **Result containers:** `all_recs` (with `protocol`; NestedSelect rows carry `selected` / `chosen`), `DEFAULT_RECS[(set, protocol)]` (NestedSelect), `NB_RECS[(set, protocol)]`, `REG_RES[protocol]` (`reg_res` = primary; incl. `REG_MERTFULL`), `df_wp_ceiling` / `WP_CEILING`, ablation rows gain `selected` / `n_configs_selected` / `n_features_reg`, `PRED_QUADS`, `INTERP_CV_R[protocol][dim]`, `INTERP_CV_F1[protocol]`, `COE[protocol]`, `BLOCK_F1[protocol]`, `df_within_piece`, `df_delta_tests`, `df_reclevel`, `df_inner_skips`; ablation rows gain `r2_val/r2_aro/rmse_val/rmse_aro/beats_chance/classifier`.

---

## 8. Workflow

**Repo:** `makris2003/violin-emotion-thesis` (private). GitHub = source of truth; Kaggle = GPU execution. `CLAUDE.md` at repo root is auto-read by Claude Code. Loop: edit locally (VS Code) → commit/push → Kaggle pulls → run on GPU → commit results back. Claude Code **cannot run** the notebook (no audio/GPU) → surgical edits + static checks + local synthetic tests only (the §4.0 (now §5.1) `# >>> HELPERS:cv_engine` block is self-contained for that purpose). The notebook is stored as single-line JSON, so review changes cell by cell (e.g. nbdime) rather than with a plain `git diff`. (Phase C2: with every output cleared the file is ≈ 0.8 MB — still above that limit.) At ≈ 7 MB without the §2.6 (now removed in C2) audio outputs (≈ 35 MB with them) it exceeds the Read tool's notebook limit, so Claude Code cannot use NotebookEdit on it: cells are edited by splitting the JSON at the `{"cell_type"` boundaries, decoding / re-encoding the changed cell sources with sed/awk under `LC_ALL=C`, clearing the outputs of changed code cells, and verifying byte for byte that every untouched cell is unchanged. Kaggle "Pull from GitHub" can stale its OAuth token (unlink/relink to fix).

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

- **"Does any feature set beat chance in classification?"** Against the *empirical* null of the deployed pipeline (§4.5c (now §5.3 / §6.9): NestedSelect on noise, P95 0.370 at 10 columns / 0.459 at 25), only the sets that contain the reference-free timing block do (DTW 0.398, CREPE + DTW 0.460, Essentia + DTW 0.373, all piece). The Dummy floor (P95 0.323) is too low for a nested-selection pipeline: 25 noise columns score 0.354. Regression is unaffected (noise R² ≈ 0), and the thesis claims rest on it.
- **"Why does MERT fail on unseen pieces?"** It recognises the piece. The nearest neighbour of every excerpt in MERT-full is another rendition of the same piece (100 %), at every depth (93–100 %, §4.4f (now §7.2)). Capacity overfitting did not fire (§4.4e (now §7.1)). MERT does carry condition information once the piece mean is removed (A2 0.90 > interpretable 0.77), but the piece identity masks it.
- **"Which technique drives arousal?"** Not identifiable with this design. The performer escalated the techniques together, so within a piece dynamics, vibrato and the DTW deviation predict each other (§5.6b (now §10.3)), and ≈ 0.35 of the 0.65 arousal R² is shared across families. Only Dynamics has a significant unique loss (q 0.028). The remedy is a factorial, one-technique-at-a-time recording.

---

## 10. Audit (Oct 2026)

A full audit of the 2026-08-16 run's outputs produced three phases of corrections.

- **Phase A — pipeline corrections (APPLIED and run on 2026-10-09; results in §4; the A3 MERT issue turned out to be a false alarm).** A-shared §4.0 (now §5.1) CV engine · **A1** DummyClassifier floors were re-created with one seed in every fold, so `stratified`/`uniform` predicted the same constant everywhere (F1-macro = 0); now seeded per fold × label, 200 repeats, mean/SD/P95 · **A2** leave-one-piece-out PRIMARY everywhere, nested piece-grouped ridge instead of RidgeCV, §5.5 (now §10.1) handles a negative CoE, new §4.9 (now §8.1) within-piece prediction · **A3** MERT's positional-convolution weight-norm tensors were never loaded (checkpoint `weight_g/_v` vs model `parametrizations.weight.original0/1`) — the conv ran with random weights; now restored + asserted, cache `emb_mert_v2.npz`; **every MERT-based number from before is superseded** · **A4** GaussianNB as the single pre-declared default; every best-of selection removed (§4.5 (now §6.6) headline, §4.6 (now §6.7) best overall, §4.8 (now §6.6), §7.1 (now §11.1)) · **A5** class balancing + per-label inner-CV C for SVM/LogReg, balanced RF/ET, XGBoost `scale_pos_weight`; tier `4D-balancing` · **A6** out-of-sample R² and RMSE next to every r, R² permutation p in the ablation BH family · **A7** MERT probe with GroupKFold(5) over pieces and nested α (grid to 1e4) · **A14** BH-FDR over all §5.1 (now §9.1; its head → §4.1) tests (174 in Phase A with 29 interpretable features; 180 since Phase B), stars from q · **A-QC** §2.3c recording-level sanity check (report only).
- **Key fact behind A2:** 95% of valence variance and 91% of arousal variance across the 60 excerpts is between-piece; the condition explains 1% (V) / 4% (A).
- **Phase B — descriptor repairs + analysis upgrades (APPLIED 2026-10-10; RUN on N = 118 — results in §4).** Analysis upgrades: **B-NMS** nested model selection replaces the single default classifier · **B-INC** Freedman–Lane incremental test in §4.9 (now §8.1) (ΔR² = R²(Condition + Features) − R²(Condition-only); reduced model = OLS on the centred condition one-hot; residuals permuted within piece; closed-form refits; BH over the 8 tests; columns `r2_cond_plus`, `d_r2_incremental`, `p_incremental`, `q_incremental`) · **B-NC** split-half noise ceiling (200 random rater halves, centred within piece, Spearman–Brown; uncentred for context; `r2_over_ceiling`) · **B-MERTFULL** regression heads on MERT-full (`ablate(X_reg=…)`, regression rebuild assert, §4.2 (now §6.2) `Ridge-nested · MERT-full` row; `4A-mert-pca` unchanged). Descriptor repairs: **B-shared** CREPE tracks cached, `fix_octave_errors`, `segment_notes` (never across unvoiced gaps) · **B8** vibrato · **B9** portamento (+ `4A-crepe-portamento` removed) · **B10** F0 range P95 − P5 on the corrected track · **B11** L1-normalised flux · **B12** dynamic-range silence gate + 3 × 3 sweep · **B13** `timing_*` family (note onsets from CREPE, 151-frame tempogram × log-normal prior, `timing_tempo_logsd`), `dtw_global_tempo_ratio` → `dtw_duration_ratio` · **B-NF** spectral-flatness noise floor + FLAT_MIN sensitivity + optional session map · **B-cache** `_v2` names + assert.
- **Phase B2 — descriptor repairs after the manual audit (APPLIED 2026-11-05; run in full on N = 116)** — below.
- **Phase B3 — re-articulation detector disabled + MERT / family-overlap diagnostics (APPLIED 2026-11-10; RUN 2026-11-11)** — below.
- **Phase C1 — empirical chance null + DIAGNOSTIC MERT layer study + `RUN_MANUAL_AUDIT = False` (APPLIED 2026-11-16; RUN 2026-11-16 — results in §4)** — below.
- **Phase C2 — story-order restructuring + the C2 obligations (APPLIED 2026-11-20; RUN 2026-11-20 — outputs in `5c94dc2`, not transcribed)** — below.
- **Phase C3 — null resumed from a Kaggle input, per-width draw targets, one null rule, claim-10 rule, Fig 8 labels (APPLIED 2026-11-20; NOT RUN)** — below.
- **Phase C (deferred idea):** augmentation (see §5 item 11).

### Phase C3 — applied 2026-11-20, NOT RUN

**What it is.** Small fixes after the C2 run (Kaggle, 2026-11-20; outputs committed in `5c94dc2`, not yet
transcribed into "Current results"). The C2 primary null stopped at its 480-min budget with **600 draws per width
(561 at d = 52)**. Edited only (no cell executed, nothing committed); the C2 notebook (with its outputs) is kept as
`notebook/thesis_pipeline_C2_backup.ipynb` (untracked); all outputs of `thesis_pipeline.ipynb` cleared (also the
papermill / widget metadata). Still **118 cells**; numbering and structure of Phase C2 unchanged. Every code cell
passes `ast.parse`; the AST use-before-define scan finds nothing. **No reported result number in these docs has
been changed.**

**Changes.**
- **A · Resume the null from a Kaggle input (no redraws).** §0.3 `NULL_CACHE_INPUT_DIR =
  '/kaggle/input/datasets/george031218732003/violin-thesis-null-cache'` (Kaggle dataset
  `george031218732003/violin-thesis-null-cache`, holding the C2 `null_f1_draws.csv` and `null_f1_draws_primary.csv`).
  At the start of §5.3, before any cache is read, each file is **copied** to `OUTPUT_DIR` if the input exists and
  the `OUTPUT_DIR` copy does not (inputs are read-only; the cell appends). The cell then prints the rows per
  (protocol, d) and how many match the current settings hash, with a loud warning if 0 match (cache invalidated).
- **B · Per-width draw targets.** `NULL_N_DRAWS_PRIMARY = {7: 1000, 17: 1000}`, `NULL_N_DRAWS_PRIMARY_DEFAULT = 600`
  (§0.3), read through `null_primary_target(d)` (§5.3). Reason: the DTW set (d = 7) passed BH with p = 4/601 = 0.0067
  against a BH limit of 0.05/7 = 0.0071 — one draw decided it; d = 17 is CREPE + DTW (primary) and Essentia + DTW.
  d = 52 is completed from 561 to 600. `NULL_PRIMARY_BUDGET_MIN = 240` (safety cap). The final n_draws / target per
  width are printed in §5.3 and next to every primary p (§6.9 `null_n_draws`, `null_target`; claim 5 lists them).
  **Settings hash unchanged:** the C2 hash contained `n_draws = int(NULL_N_DRAWS_PRIMARY)` = 1000; that field is now
  the frozen literal `_NULL_PRIMARY_HASH_N_DRAWS = 1000`, so the per-width targets are not in the hash and every C2
  row keeps its hash. Hash fields (primary): n_draws (frozen 1000), seed, protocols, n, y_sha1, groups_sha1, configs,
  inner_splits, tie_tol, generator; (map, unchanged): n_draws 100, dims, seed, protocols, n, y_sha1, groups_sha1,
  configs, inner_splits, tie_tol, generator. The primary hash is now built before the cache check (moved up, same
  value).
- **C · One null rule everywhere.** In C2, §6.9 used the exact-width null while §7.1, §7.2, §10.2, §10.3 and claim 10
  used the 100-draw map (DTW: p 0.0067 vs 0.0099; Essentia + DTW, d = 17, got ✓ against the map d = 10, P95 0.370,
  although the exact d = 17 null exists, P95 0.433). Now **one helper**, `emp_null_ref` (with `emp_null_mark`,
  `emp_null_dim`, `emp_null_p` as wrappers), by width only:
  1. a primary null exists at exactly d → **"exact-width null d = …"**;
  2. else the **smallest primary width ≥ d** → **"null borrowed from d = … (conservative, next larger width)"** (the
     null P95 rises with d);
  3. else (d > 52, e.g. MERT-full 768) → **"null borrowed from d = 52 (extrapolated)"**.

  Primary widths: 7, 10, 17, 25, 27, 35, 52. `emp_null_kind` ∈ exact / borrowed / extrapolated / not drawn. The
  100-draw map is still drawn and cached (the primary d = 10 / 25 reuse its draws 0–99; the §7.1 draw-0 guard reads
  it) but is **not a reference anywhere**. Cells: §5.3 (rule + explanation), §6.3 (✓ via `ablate`, explanation, the
  lattice figure lines now the exact-width d = 10 / 25 P95), §6.4 / §10.3 / §7.2 first cell (✓ via `ablate` — no
  code change needed), §6.9 (read-out via `emp_readout_rows`, explanation, `null_extrapolated`), §7.1 (via
  `emp_readout_rows`), §7.2 continued (profile-figure P95 from the rule), §10.2 (`alone_emp_p` + new label column
  `alone_emp_null`, figure lines), §11.1 (claims 5 / 10).
- **D · Claim 10 rule (§11.1) — revised post hoc.** C2 checked only that the timing F1 survives removing the
  tempogram; the C2 §6.3 showed that the timing **arousal** signal depends on it (R²a 0.32 → 0.09 for "DTW ref-free −
  tempogram (4d)", p .016, q .08). New rule: **✔** Timing (ref-free) has the highest family-alone F1 (piece) AND beats
  its null (rule C, `alone_emp_p` < .05) AND the 4-feature timing set without the tempogram keeps its F1 above its
  null (empirical p < .05, rule C — d = 4 → borrowed from d = 7) AND loses < 0.10 of R² arousal (ΔR²a > −0.10) ·
  **◐** highest family-alone F1 but another condition fails · **✘** not the highest. The value string shows the ΔF1
  and the ΔR²a (p, q) of the tempogram removal. The code comment states it as a post-hoc revision after the C2 run.
- **E · Fig 8 labels (§7.4).** `SHIFT_PIECE_LABEL` (built once in §7.4): the first 10 characters, extended only where
  two piece names would collide (the five Shostakovich pieces → full names, `Shostakovich_V` for the VC1); asserted
  unique; used by Fig 8 (the only place in §7.4 that prints piece names).

**C2 outputs the C3 run will replace** (C2 values from the committed `5c94dc2` outputs, not transcribed elsewhere):
- §6.9 / claim 5: the DTW p 0.0067 (600 draws) / q 0.0466 → 1000 draws at d = 7; CREPE + DTW p 0.0283 / q 0.099
  (600 → 1000 draws); C: Combined at 561 → 600 draws. Claim 5 in C2: ◐ (1 of 7 primary sets q < .05: DTW).
- Every ✓ / p that used the map: in §6.9 Essentia (map d = 10 → exact d = 10), Essentia + DTW (✓ vs the map d = 10,
  p 0.0495 → exact d = 17), CREPE + Essentia (map d = 25 → borrowed d = 25), the MERT + X sets of width 32 / 42 / 45
  (map d = 25 → borrowed d = 35 / 52), the two tempogram rows (map d = 10 / 25 → borrowed d = 7 / 25); the §7.1 block
  rows, the §7.2 layer-candidate p / q, every ✓ printed by `ablate` in §6.3 / §6.4 / §7.2 / §10.2 / §10.3.
- §10.2 `alone_emp_p` (map d = 10 / 25 → the primary null of the family's width or the next larger one).
- Claim 10 (C2: ✔ under the old rule — "Timing (ref-free) F1 0.398, map-null p 0.0099 · − tempogram ΔF1 −0.018
  (q 0.916)").

#### C3 run checklist

Runtime: §5.3 ≈ 839 new draws (d = 7: 400, d = 17: 400, d = 52: 39; d = 10 / 25 / 27 / 35: 0) × ≈ 7.3 s ≈ **102 min
(≈ 1.7 h)**, inside the 240-min cap; everything else ≈ the C2 run.
- [ ] Add the Kaggle dataset `violin-thesis-null-cache` as an input; §5.3 prints "copied … → OUTPUT_DIR" for both
      files and the cache check shows **all rows matching** (map 200 / 200, primary 4 161 / 4 161), no ⚠️.
- [ ] §5.3: "to draw" = 839 (d=7: 400, d=17: 400, d=52: 39); final draws d = 7 / 17: 1000, all others 600; no
      INCOMPLETE warning; runtime ≈ 1.7 h.
- [ ] §6.9: DTW p / q with 1000 draws (BH limit for rank 1 = 0.0071); `null_target` column; Essentia + DTW now
      "exact-width null d = 17".
- [ ] §6.3 / §7.1 / §7.2 / §10.2: labels "exact-width" / "borrowed … (conservative, next larger width)" /
      "extrapolated"; no "map" label anywhere.
- [ ] §7.4: Fig 8 labels unique.
- [ ] §11.1: claim 5 lists the draws per width; claim 10 shows ΔF1 and ΔR²a; read the verdict against the revised
      rule (the C2 ΔR²a ≈ −0.23 would give ◐).

### Phase C2 — applied 2026-11-20, NOT RUN

**What it is.** A restructuring of `notebook/thesis_pipeline.ipynb` so that **the notebook order is the story
order** (every part answers the question the previous one opens — the presentation follows the same order), plus
the C2 code obligations. Edited only (no cell executed, nothing committed); the C1 notebook is kept as
`notebook/thesis_pipeline_C1_backup.ipynb` (untracked). All outputs cleared (also the papermill / widget metadata).
**118 cells** (67 code, 51 markdown; C1: 114). Every code cell passes `ast.parse`; an AST scan finds no name used
before the cell that defines it. **No reported result number in these docs has been changed** — nothing has run;
the numbers the C2 run will replace are listed in the checklist below.

**Old → new section map.** Historical text elsewhere in this document keeps the old numbers, annotated
"(now §x.y)".

| New part · section | Content | From (C1 §) |
|---|---|---|
| **Part 0 · Setup** · §0.1–0.4 | unchanged (§0.3 gains the C2 constants and `bh_fdr`) | §0.1–0.4 |
| **Part 1 · Audience ground truth** · §1.1, 1.1b, 1.2, 1.2b, 1.3 | parse, normalization, aggregate, validation, labels | same (§1.4 / §1.5 → Part 4) |
| **Part 2 · Feature extraction** · §2.1–2.5 | unchanged | same; **§2.6 manual audit deleted** |
| **Part 3 · Tool validation** · §3.1–3.4 | unchanged (+ one ⚑ tempogram line) | §3.1–3.4 |
| **Part 4 · Manipulation check** · §4.1 | interpretable matrices (`INTERP_PRED_COLS` / `INTERP_DESC_COLS`) | head of §5.1 |
| §4.2 | performers escalated — Page's test on the techniques (BH) | §5.1b technique part |
| §4.3 (a · b · c) + conclusion | Friedman / Wilcoxon · Page on the ratings · lift + class share per condition · printed conclusion | §1.4 · §5.1b rating part · §1.5 |
| **Part 5 · Evaluation yardstick** · §5.1 | protocol and CV engine | §4.0 |
| §5.2 | Dummy floors (table only) | §4.5b (first half) |
| §5.3 | empirical null: the 100-draw map + the exact-width primary null | §4.5c (first half) + new |
| **Part 6 · Prediction** · §6.1 | classification sensitivity table (✓ = empirical null, NestedSelect rows only) | §4.1 |
| §6.2 | valence / arousal regression (deployed nested ridge first) | §4.2 |
| §6.3 | block lattice (+ tempogram sensitivity tier) | §4.4b |
| §6.4 | configuration ablation | §4.4c |
| §6.5 | target-side ablation | §4.4d |
| §6.6 (+ §6.6b) | per-class metrics with α merged in · floors vs the default · figures | §4.5 + §4.8 + §4.5b (second half) |
| §6.7 | per-condition F1 | §4.6 |
| §6.8 | quadrant accuracy (deployed nested ridge) + Fig 5 + Fig 6 | §4.7 |
| §6.9 | every F1 read against the empirical null, BH in the primary family | §4.5c (second half) |
| **Part 7 · MERT diagnostics — what the embedding encodes** · §7.1 | train vs test, validation curve, NN identity | §4.4e |
| §7.2 (+ continued) | MERT layer study (+ the candidates vs the null) | §4.4f (+ §4.5c layer part) |
| §7.3 / §7.3b | MERT ↔ feature correspondence / linear probe (`PROBE_R2`) | §6.1 / §6.1b |
| §7.4 | embedding shift vs signed rating shift | §5.3 |
| §7.5 | what each tool adds on top of MERT (rewritten) | §6.2 |
| **Part 8 · Within-piece prediction** · §8.1 | unchanged method | §4.9 |
| **Part 9 · Which techniques** · §9.1 | condition deltas + Fig TE1 (3 panels) | §5.1 (minus its head) |
| §9.2 | interpretable linear read-out | §5.4 |
| §9.3 | SHAP (in-sample cross-check) + agreement table | §5.2 |
| **Part 10 · Cost of explainability** · §10.1 | cost of explainability | §5.5 |
| §10.2 | sub-family ablation | §5.6 |
| §10.3 | family overlap diagnostics | §5.6b |
| **Part 11 · Synthesis** · §11.1 | claims table + save outputs | §7.1 (rewritten) |

Deleted: **§2.6** (manual audit, its two code cells and markdown, `RUN_MANUAL_AUDIT`, `AUDIT_EXCERPTS`, `_aud_*`),
**§4.4** "Ablation table + Wilcoxon" (n = 8 classifiers are not independent units; minimum p 0.0078 — the A–D ×
classifier table and `ablation_results.csv` now come from §6.1), **§6.2b** (§7.5 reads the lattice rows directly,
so there is nothing left to cross-check), the separate **§4.8** cell (merged into §6.6).

**Changes (by prompt section; C2 obligation numbers in brackets).**
- **A · Interpretable layer** — §4.1 builds `df_interp` once and two lists: `INTERP_PRED_COLS` (27 = CREPE 10 +
  Essentia-forward 10 + DTW reference-free 7; asserted = the lattice set "CREPE + Essentia + DTW", columns in §4.1 and
  values in §6.3) and `INTERP_DESC_COLS` (30 = + the 3 `DTW_FORWARD_DEV`). `DTW_DEV_IN_INTERP` is removed (it lived in
  §2.4b, not §0.3; §2.4b keeps `DTW_DEV_IN_MODELS = False`, `df_dtw_attr` = both families); comments in §0.3 / §2.4b.
  PRED is used by §6.2, §6.4 (4D "Interp", already identical), §6.8, §7.1 (`MG_X_INTERP`), §7.2 (reference), §8.1,
  §9.2, §9.3 (`X_INTERP` is now the 27-column matrix), §10.1–§10.3; DESC only by §4.2 and §9.1 (⚑ rows). §10.2 drops
  the "Timing (DTW-dev)" family with one printed line; §10.3's assert compares the 27-column matrices.
- **B · Part 4** — §4.2 Page test on `INTERP_DESC_COLS`, **one BH family = the technique tests + the two rating tests**
  (`q`, stars from q, ⚑ for DTW-deviation and tempogram) **[2]**; §4.3a = former §1.4 (+ `FRIEDMAN_RES`), §4.3b = the
  rating rows + the dose figure, §4.3c = former §1.5 + the ground-truth class share per condition
  (`class_share_by_condition.csv`); a markdown conclusion + a cell that prints it from the variables (`MANIP_CHECK`).
- **C · Part 5** — §5.2 floors only (`metrics_dummy_floors.csv`); §5.3 = the 100-draw map **before** any F1 + the
  **exact-width primary null [1]**: pre-declared family A–D, Interpretable (27), DTW, CREPE + DTW;
  `NULL_N_DRAWS_PRIMARY = 1000` (§0.3; Phase C3: per width — 1000 at d = 7 / 17, 600 elsewhere); same seeding (map draws 0–99 reused); cache `null_f1_draws_primary.csv`
  (per-row settings hash, **resumable**, checkpoint every 50 draws, round-robin over widths,
  `NULL_PRIMARY_BUDGET_MIN = 480`, C3: 240); `emp_null_ref` / `emp_null_mark` (C2 labels — C3: one rule, no "map" reference — "exact-width", "map", "null borrowed from
  d = …"). Every printed classification ✓ from §5.3 on = above the empirical-null P95; in §6.1 only the NestedSelect
  rows carry it (the fixed classifiers carry none, a printed line says why); ablate rows gain `emp_null_p`,
  `emp_null_p95`, `emp_null_d`, `emp_null_kind`, `beats_emp_null` (`beats_chance` kept). §6.9 reads the map and the
  primary null, **BH within the primary family** (`emp_null_primary.csv`, `q_primary`); §7.1 / §7.2 append their rows to
  `emp_null_readout.csv`.
- **D · Tempogram [3, "keep flagged"]** — `TEMPOGRAM_UNRELIABLE` (§0.3); kept in the models; ⚑ in §4.2, §9.1, §9.2, §9.3;
  never quoted alone as a top feature (`top_feature_note`); §6.3 tier `1b-tempogram-sensitivity`: "DTW ref-free −
  tempogram" (4) vs DTW and "CREPE + Essentia + DTW − tempogram" (24) vs its parent, both protocols, paired p, BH in
  the tier (`ablation_tempogram_sensitivity.csv`).
- **E · Part 7** — §7.1 "near-constant model" → "class-mean-difference (prototype) classifier whose threshold depends
  on the fold's label composition" **[5a]**; the draw-0 guard moved here. §7.2: 27-column reference, decision line
  **`MERT_LAYERS` stays [5, 6, 7], the study is diagnostic [6]**; the candidates-vs-null part (former §4.5c) follows it.
  §7.3b exposes `PROBE_R2`. §7.4: signed ΔV / ΔA only, BH over the 6 tests, **Fig 8 with 3 panels** (r, p, q in each
  title), two labelled exploratory correlations of the MEC → EXG distance (MEC arousal; ΔLUFS), a pointer to §7.2 A3
  (`analysis_d_shift_tests.csv`). §7.5 rewritten: MERT / MERT + CREPE / Essentia / DTW read from `ABL_PRED` (no refit),
  ΔF1 / ΔR²v / ΔR²a with the lattice's paired permutation p and BH within §7.5; **CREPE-unique by rule** (probe R² <
  `CREPE_UNIQUE_R2_MAX = 0.2`, §0.3; list printed, asserted non-empty, median-substitution counts printed) — the only
  new fit, "MERT + CREPE-unique" (tier `7.5-mert-plus`); the ±0.005 ADDS / HURTS verdict is gone.
- **F · Part 6** — §6.2 headline = deployed nested ridge on MERT-full 768 + an "Interpretable (27)" nested-ridge row;
  fixed Ridge α = 10 / SVR / MLP on MERT-PCA25 labelled "sensitivity (not deployed)" (`role` column). §6.6 = one
  per-class table with "chance-corrected agreement α" (+ the not-inter-rater-reliability sentence). §6.8 + Fig 5 from
  the deployed nested ridge (MERT-full and Interpretable), MEC in the colour map (all 60 points), majority-quadrant
  share + a 1000× permutation null (`quadrant_accuracy.csv`).
- **G · Parts 9 / 10** — §9.1 Fig TE1 with the **MEC → EXG** panel, ⚑ rows, BH over all §9.1 tests, subtitle with the
  raw p < .05 count, the count expected by chance and the q < .05 count (`patch_TE1_Fig8.py` was not in the repo —
  implemented from the description). §9.2: 27 columns, ⚑. §9.3: labelled "nonlinear, in-sample cross-check (RF fitted
  on all 60, no CV score)", stale comment fixed, agreement table SHAP top vs β top per class
  (`shap_vs_readout_agreement.csv`). §10.3 test 1 requires I > `OV_I_TOL` (0.02) **[5b]**.
- **H · Labels** — "MERT-PCA25" / "MERT-full 768" next to numbers where the cell builds the label (§6.1 md, §6.2, §7.5,
  §8.1, §10.1, figure titles); the lattice set names keep "MERT" (renaming them would break the `ABL_PRED` keys the
  guards and §7.2 rely on) and §6.3 prints a legend line instead; a note on why regression uses MERT-full (§5.1 md,
  §6.2 md).
- **I · §11.1 claims table** — `df_claims` (claim, section, value from variables, comparator, verdict ✔ / ◐ / ✘, rule)
  → `claims.csv`, 10 rows in the requested order; `final_summary.csv` now carries the exact-width p and the primary BH
  q; `all_predictions.csv` holds the deployed nested-ridge predictions (MERT-full and Interpretable) and quadrants.
- **J · Housekeeping [7]** — §2.6 removed. Report-only re-articulation code and the legacy FFT helpers untouched
  (**pending decision**). No multiclass classifier (**[4] on hold**). `bh_fdr` moved from the §5.1 HELPERS block to §0.3
  (Part 4 runs before the engine; a stand-alone test of the HELPERS block must now define it).

**Closed decisions.** `DTW_DEV_IN_INTERP` → replaced by the predictive / descriptive split (§4.1) · `CREPE_UNIQUE2` →
the probe rule (`CREPE_UNIQUE_R2_MAX = 0.2`) · `NEUTRAL_RADIUS` stays 0.75 · `MERT_LAYERS` stays [5, 6, 7] (study
diagnostic) · Fig 5 (MEC missing) and the fixed α = 10 in the former §4.2 / §4.7 → resolved by §6.2 / §6.8 (deployed
nested ridge; fixed models kept as labelled sensitivity rows). C2 obligations **1, 2, 3, 5a, 5b, 6, 7 done** (applied,
not run); **4 (multiclass) on hold**; 8 (pending decisions) closed except the re-articulation / FFT helpers; 9 (final
run) and 10 (optional piece-centred MERT) open.

#### C2 run checklist (what to look at; which C1 numbers change)

Runtime (C2, as applied then; the C3 targets are in "C3 run checklist"): the primary null is the new cost — 7 widths × 1000 draws, minus the 200 map draws reused ≈ 6 800
NestedSelect runs × ≈ 8 s (C1: 26.6 min per 200 draws) ≈ **15 h**, more than one Kaggle session. With
`NULL_PRIMARY_BUDGET_MIN = 480` the first session stops the primary null after 8 h (≈ 3 600 draws, ≈ 500 per width)
and continues the notebook with p computed on the draws available (printed as INCOMPLETE); keep
`null_f1_draws_primary.csv` in `OUTPUT_DIR` and re-run to resume. Everything else ≈ the C1 run (≈ 3 h): §6.1 ≈ 29 min,
§7.2 ≈ 32 min, §6.4 ≈ 25 min, §10.2 ≈ 18 min (fewer families), §6.3 ≈ 16 min (+ the tempogram tier, ≈ 2–4 min),
§10.3 ≈ 10 min (21 pairs instead of 28), §7.5 one extra fit, §6.8 1000-permutation null (seconds).

- [ ] §0.3 / §2.4b: no `DTW_DEV_IN_INTERP`; §4.1 prints 27 predictive + 3 descriptive features and the ⚑ list; the
      lattice-set assert passes.
- [ ] §4.2 / §4.3b: q column; how many techniques keep q < .05 (C1 hand-BH: 12 of 30 in the old family; the new family
      adds the two ratings); arousal q < .05, valence n.s.; the Part 4 conclusion prints ✅.
- [ ] §5.3: map loaded from `null_f1_draws.csv` (settings unchanged → no refit); primary widths printed (expected
      7, 10, 17, 25, 27, 35, 52); completeness of the primary null; runtime.
- [ ] §6.1: ✓ / · only on NestedSelect rows; `ablation_results.csv` written.
- [ ] §6.2: the deployed rows — MERT-full R²a (C1 0.446) and **Interpretable (27): R²v ≈ 0.334, R²a ≈ 0.636** (the C1
      27-column numbers; C1 "Interpretable (30)" R²v 0.258 / R²a 0.648 is superseded).
- [ ] §6.3: X_INTERP == lattice assert; tempogram tier Δ / q; lattice rows now show `emp_null_kind` / `emp_null_p`.
- [ ] §6.6: α column; §6.8: quadrant accuracy MERT-full vs Interpretable vs majority / permutation null; Fig 5 has 60 points.
- [ ] §6.9: primary family p and **BH q** (C1, 100-draw map: DTW 0.0099, CREPE + DTW 0.0099, the rest ≥ 0.089 — all
      will change with exact widths and 1000 draws — Phase C3: 1000 at d = 7 / 17, 600 elsewhere).
- [ ] §7.1: draw-0 guard passes; interpretable reference now 27 columns (C1 train / test F1 0.79 / 0.44 with 30).
- [ ] §7.2: interpretable reference (A1 65 %, A2 0.767 with 30) changes; W05-07 rows unchanged (HARD CHECK).
- [ ] §7.4: signed r / q for 3 transitions; exploratory r with MEC arousal and ΔLUFS.
- [ ] §7.5: the CREPE-unique list (C1 probe R² < 0.2: vibrato rate −0.02 / … — read it) and its Δ / q.
- [ ] §8.1: unchanged inputs (the interpretable set was already the 27 predictive columns) → numbers expected identical.
- [ ] §9.1: unchanged values (descriptive 30) with ⚑; TE1 has 3 panels. §9.2 / §9.3: 27 columns → betas / SHAP change;
      agreement count.
- [ ] §10.1 CoE: **changes** (interpretable side 27 instead of 30; C1 ΔR v −0.370 / a −0.156, ΔR² v −0.246 / a −0.202,
      ΔF1 −0.134 — interpretable F1 expected ≈ 0.445).
- [ ] §10.2: 7 families, the printed DTW-dev line; commonality; timing reading. §10.3: 21 pairs, test 1 with I > 0.02.
- [ ] §11.1: `claims.csv` — read every verdict against its rule.

### Phase B2 — descriptor repairs after the manual audit (2026-11-05)

**Evidence** — the §2.6 (now removed in C2) outputs of the 2026-11-03 N = 116 run (Méditation and Mozart VC4 × MEC / EXP / EXG), checked by eye and ear by the author:
- **(a) Portamento false alarms = crepe's Viterbi ramp.** Méditation MEC D6→D5 (6.57 s, −1177 c) and A5→A4 (9.36 s, −1198 c) were detected as glides (rise 62.5 ms, sign 1.0). crepe's Viterbi decoder (`crepe/core.py`, `to_viterbi_cents`: transition weight max(12 − |Δbin|, 0), 20-cent bins) moves at most 11 bins = **220 c per 10 ms frame**, so every large jump is decoded as a ramp (an octave ≈ 55 ms) — the 40 ms minimum glide duration could not separate it. Real glides in EXG last 175–312 ms (B5→C♯6 312.5, C♯6→D6 175, E5→A5 200).
- **(b) The 70 c boundary rule merges notes with their slides.** Méditation EXP "D6" 6.38–8.30 s (1920 ms, dev −12.5 c, FFT presence 0.08) is C♯6 + slide + D6; EXG C♯6 5.61–6.64 s contains slides.
- **(c) The FFT presence test rejects real vibrato.** Méditation EXP D5 8.40–10.00 s (presence 0.533, 5.58 Hz, ±4.15 c) and B4 12.73–16.12 s (presence 0.841, 5.63 Hz, ±4.40 c) were rejected only by the 5 c rule; at 0.5–2 s the Hann main lobe (±2/T Hz) is wider than the ±0.5 Hz presence window; the unclamped parabolic peak produced out-of-band rates (1.53 / 2.55 / 2.38 Hz). The author confirmed no vibrato in SQ8 exrpt1 (MEC/EXP/EXG) and exrpt3 (EXP/EXG).
- **(d) Repeated notes are merged.** Mozart VC4: 19 notes in the score, the pipeline found 17 (MEC) / 16 (EXP) / 19 (EXG); repeated D6s merged (MEC 1.39–3.09 s, EXP 1.38–3.43 s, EXG 1.70–3.04 s).
- **(e) Open, not changed:** LUFS offset +3.055 LU (all 6) · `es.Key` G minor for Mozart_VC4_EXP · Méditation tempo log2-SD MEC 0.256 > EXP 0.158 > EXG 0.135, pulse clarity lowest at MEC (✗) · Mozart LAT not sharper in spiccato EXP (✗) — see §5.

**New definitions** (`# >>> HELPERS:descriptors`, §2.2; numpy / scipy / bisect only):
- **B2 segmentation** — `segment_notes(cents, voiced, min_note_ms, step_ms=10, env_db=None, reart_dip_db=6)` (same return contract; `segment_notes_detail` also returns how each note starts — unvoiced gap / plateau / re-articulation split — and the split frames). Per voiced run: greedy plateaus of the 15-frame median trend (each frame within ±40 c of the plateau's running median, plateau range ≤ 80 c — a guard added so a slow ramp cannot pass as a plateau), candidates ≥ 60 ms; adjacent plateaus < 50 c and ≤ 50 ms apart merged; each note split at a local minimum t of the 10 ms envelope with max(env[t−150 ms : t]) − env[t] ≥ 6 dB and max(env[t : t+60 ms]) − env[t] ≥ 6 dB (candidates accepted deepest first, every part ≥ 60 ms); notes < min_note_ms dropped. Envelope `env_db_on_grid`: RMS over a rectangular 25 ms window centred on each CREPE frame time i·10 ms, on the same 16 kHz mono audio CREPE reads (zeros outside the signal, floor −120 dB), cached in `rms10_v3.npz`. The Phase B rule is kept as `_segment_notes_v2` (report only). A re-articulation split is a note onset for the timing family.
- **B9b portamento** — transitions between consecutive notes of one run; t_min = 0.8·|I| / 220 · 10 ms × 1.25 (the shortest 10–90 % rise a pure jump produces through the Viterbi decoder); excess = rise − t_min; glide iff rise ∈ 80–500 ms, excess ≥ 50 ms, |I| ∈ 80–2000 c, sign ≥ 0.80. Verdicts: across an unvoiced gap · interval out of range · no 90 % crossing · too short · viterbi-limited jump · too slow · not monotone · glide. §2.2 checks the installed crepe source (`"12 - abs"` in `to_viterbi_cents`; prints, never raises).
- **B8b vibrato** — notes ≥ 500 ms (sensitivity 500 / 650 / 800), 500 ms windows every 250 ms (last window shifted to the note end); degree-2 detrend; least-squares a·sin + b·cos over 4–8 Hz in 0.05 Hz steps; R²; rate = argmax with a parabolic refinement clamped to the grid; half-extent = √(a² + b²). Vibrato window iff R² ≥ 0.50 and half-extent ≥ 3 c. Coverage = union of vibrato-window time / analysed-note time; rate / extent = duration-weighted medians over vibrato windows; the declared substitutions and the ⚠️ are unchanged; column names unchanged. The FFT detector stays as the legacy `vibrato_note` (report only, §2.6 (now removed in C2)).
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

### Phase B3 — re-articulation detector disabled + MERT / family-overlap diagnostics (2026-11-10; APPLIED; RUN 2026-11-11)

**Evidence — the full B2 run (N = 116, block weighting; numbers quoted by the author, outputs not committed):**
- **(a) Re-articulation splits follow vibrato, not bow changes.** The B2 detector split a plateau where the 10 ms RMS envelope dipped ≥ 6 dB. On the violin, vibrato also modulates the amplitude (the frequency sweeps through the body resonances); the 25 ms envelope resolves that modulation and the 6 dB rule took it for a bow re-attack. Splits per condition **MEC 24 / EXP 51 / EXG 68** for the same scores; Meditation_Thais 0 / 3 / 11 — the EXG F♯5 cut into 7 pieces of 140–370 ms (1–2 vibrato periods); Shostakovich_Sym5_Mvt3_EXG 11 splits. Note totals per condition: Phase B boundary rule 575 / 533 / 518 → B2 plateaus only 616 / 596 / 592 → B2 with splits 640 / 647 / 660. `REART_DIP_DB` sensitivity, Mozart_VC4 MEC / EXP / EXG: 26/23/28 (3 dB), 22/21/23 (6 dB), 19/17/22 (9 dB) vs 19 in the score — no threshold fixes all three. Consequences: §5.1b (now §4.2 / §4.3b) note-based timing turned significant (note rate z = +2.06, p = 0.040; IOI mean −2.53, p = 0.011; IOI CV +2.85, p = 0.004 — all n.s. in Phase B); §5.6 (now §10.2) "Timing (ref-free)" alone F1 (piece) 0.417 → 0.542, its removal loss −0.012 → +0.110 (q = 0.037); long EXG vibrato notes fragmented below the 500 ms analysis minimum; the portamento share counted repeated-note pairs in its denominator.
- **(b) The merge broke the plateau range invariant.** Adjacent plateaus (< 50 c apart, ≤ 50 ms gap) were merged without re-checking the 80 c range: Meditation_Thais_EXG C♯6 (≈ 5.76–6.62 s) had a trend range of 128 c (V3 FAIL); merged notes also absorbed the transition frames between the plateaus.
- **(c) Vibrato rate pinned at the band edge.** At least one excerpt had a weighted-median vibrato rate of exactly 4.00 Hz, the lower edge of the 4–8 Hz grid: the best fit lay outside the band — a residual slow trend, not vibrato.
- **(d) MERT generalisation (§4.4b (now §6.3) / §4.4c (now §6.4) / §5.5 (now §10.1) / §4.5b (now §5.2 / §6.6)).** F1 LOOCV → piece: MERT 0.623 → 0.310 (−50 %, below the chance P95 0.323) vs CREPE 0.500 → 0.333, Essentia 0.480 → 0.356, DTW 0.570 → 0.542; adding MERT to an interpretable block lowered the piece F1 in 6 of 7 combinations and raised the LOOCV F1 in 7 of 7; unique F1 of MERT (piece) −0.122 (q = 0.034); NestedSelect chose C = 0.001 (almost constant) in 12/20 piece folds and C = 10 in 43/60 LOOCV folds. MERT-PCA25 ridge: negative out-of-sample R² under piece (valence −0.18, arousal −0.25); MERT-full ridge transfers arousal (R² 0.45) but not valence (0.01); the valence α of the MERT combos hit the upper grid edge. The MERT PCA is fit once on all 60 excerpts (label-free). No train score was computed anywhere → capacity overfitting vs a piece-identity shortcut (LOOCV keeps the other renditions of the same piece in training) could not be told apart.
- **(e) Family overlap (§5.6 (now §10.2), piece).** Full interpretable layer (30 features, 8 families) F1 0.449, R²a 0.627; Σ family-alone R²a ≈ 1.55, Σ removal losses ≈ 0.23 → most of the arousal R² is shared. The §5.6 (now §10.2) F1 figure drew "alone" bars from 0 although the chance mean is 0.272 (P95 0.323). "Small removal loss + high alone score" can be redundancy, noise (only 2 of 24 removal losses q < 0.05), dilution (Timing alone 0.542 > all 30 features 0.449) or a different NestedSelect choice per subset.

**Decision (author): disable the re-articulation detector for the features; keep it as a documented, report-only negative result.** Without it, repeated same-pitch notes with no silence between count as one note. That bias is set by the **score**, so it is ≈ the same in the three renditions of a piece and cancels in within-piece comparisons; the vibrato artefact grew with expressiveness and did not cancel. Only 4 of the 7 timing features use note onsets (`timing_note_rate`, `timing_ioi_mean_s`, `timing_ioi_sd_s`, `timing_ioi_cv`); tempo variability, local tempo and pulse clarity come from the tempogram and are unaffected. Vibrato and portamento never needed the detector.

**Changes (notebook):**
- **A1** `REART_ENABLED = False` in HELPERS:descriptors (with the evidence in a comment). `segment_notes` / `segment_notes_detail` gained an optional `reart_enabled=None` (None → `REART_ENABLED`; True forces the detector on — report only); every feature path (`note_onset_times`, `timing_note_features`, `vibrato_features`, `portamento_features`, `crepe_descriptors`, `extract_timing_reffree`) keeps `env_db` in its signature (length-checked) and ignores it. `_reart_splits`, `env_db_on_grid` and the `REART_*` constants are kept.
- **A2** `_merge_plateaus(…, max_range=PLATEAU_MAX_RANGE_CENTS)`: two plateaus merge only if the merged note's trend range, gap frames included, stays ≤ 80 c.
- **A3** `vibrato_window_fit` returns `edge` (R² argmax before refinement = the first or last grid point); `vibrato_window_reason` names the verdict ('band edge' / 'low R²' / 'small extent' / 'vibrato'); a band-edge window is never vibrato; windows record `edge` and `reason`; §2.2 prints the edge rejections (all excerpts and the 20 MEC, and how many were rejected *only* by the edge rule); the calibration grid applies the rule in every cell (asserted: deployed cell = MEC vibrato-window share of the descriptors).
- **A4** §2.2 banner "re-articulation detector DISABLED (REART_ENABLED = False) — see Phase B3"; report-only evidence table (detector on at 6 dB on the cached tracks + `env_db`: totals and splits per condition, per-excerpt counts, the 3 / 6 / 9 dB sensitivity with Mozart_VC4; asserted: detector-on notes = deployed notes + splits); label-free condition-independence check of the deployed note counts (per-condition totals, per-piece range, `scipy.stats.friedmanchisquare` on the 20 × 3 table and on the detector-on counts; ✅ p ≥ 0.05 / ⚠️, printed) → `CREPE_NOTECOUNT_FRIEDMAN`.
- **A5** caches `feat_crepe_v3.csv` → **`feat_crepe_v4.csv`**, `dtw_features_v3.csv` → **`dtw_features_v4.csv`** (with the old name the cached timing file would silently keep the B2 onsets); both v3 names in `OLD_CACHE_NAMES`, §2.5 asserts it; `crepe_tracks_v2.npz`, `rms10_v3.npz`, MERT / Essentia caches unchanged.
- **A6** §2.6 (now removed in C2) on the disabled path (asserted: no deployed split), band-edge windows shown with their reason; **V4 → INFO** ("known limitation — repeated notes without a silence are counted once", expected ≈ 17 / 16 / 20; criterion unchanged); the V2 / V3 "has vibrato" checks also require the matched note to cover ≥ 80 % of the annotated span; **new V7** = the §2.2 Friedman result.
- **§4.4e (now §7.1) — MERT generalisation diagnostics** (new, right after §4.4d (now §6.5) — the name §4.4d (now §6.5) was taken by the target-side ablation). B1 train vs test per block (MERT-PCA25, MERT-full, CREPE, Essentia, DTW, the 30-feature layer, a 25-column N(0,1) capacity control) and protocol: classification — the per-fold NestedSelect choice is read from §4.0 (now §5.1)'s memo (`return_info`; the §4.4b (now §6.3) rows store only counts), refitted with §4.0 (now §5.1)'s scaler + binary relevance; train F1 = mean over folds, test F1 pooled (asserted = NestedSelect and = the §4.4b (now §6.3) / §4.4c (now §6.4) rows); regression — nested ridge, train R² of the refitted fold model vs out-of-sample R² (asserted = the §4.4b (now §6.3) / §4.4c (now §6.4) predictions). B2 validation curve (`piece`, SVM-Linear balanced at each fixed C of `SELECT_C_GRID`) for MERT-PCA25 / interpretable / random. B3 nearest-neighbour identity: MERT-full (cosine, raw), MERT-PCA25, CREPE, Essentia, DTW, interpretable (z-scored over the 60 excerpts) — same-piece share vs 2/59, same-condition share vs 19/59, 10 000 label shuffles (keep 3 per piece / 20 per condition), silhouettes. **B4 pre-declared rules** — CAPACITY OVERFITTING: MERT-PCA25's piece train − test F1 gap ≥ the interpretable gap + 0.10 and ≥ the random-control gap + 0.10, AND every C at which MERT's train F1 has risen (≥ +0.05 over its value at the smallest C; ≥ 1 such C) keeps the test F1 ≤ the strict P95; PIECE-IDENTITY SHORTCUT: in a MERT space, same-piece share ≥ 3 × chance with p < 0.05 and above every interpretable space, while the same-condition share has p ≥ 0.05. Both can fire; neither is said explicitly.
- **§5.6 (now §10.2) edits (C0)** chance mean + strict P95 lines on the F1 panel; `*` on removal losses with q < 0.05; commonality summary per protocol (F1 above the chance mean, R²a, R²v: Σ alone, Σ loss, full, shared ≈ full − Σ loss, with the non-additivity caveat) → `ablation_subfamily_commonality.csv`; the pre-declared timing reading printed.
- **§5.6b (now §10.3) — family overlap diagnostics** (new, right after §5.6 (now §10.2)). C1 tier `3-subfamily-pairs` (28 pairs + the reference row; `piece`; `RUN_PAIR_ABLATION_LOO = False` — `ablate` already accepts `protocols`, so the engine is unchanged): I(A,B) = loss(A∪B) − loss(A) − loss(B) for F1 / R²a / R²v with loss(A), loss(B) from the §5.6 (now §10.2) tier, interpreted only when the pair loss has q < 0.05 (I > 0.02 substitutes · |I| ≤ 0.02 each its own · I < −0.02 only together), 8 × 8 heatmaps, runtime printed, `ablation_master.csv` re-exported. C2 probes (nested ridge, `NestedRidgeOperator` asserted = `nested_ridge_predict`, `piece`): each feature of A from all non-A features and from each family B, RAW and WITHIN-PIECE (feature minus its piece mean — the held-out piece's own feature mean, never a rating). C3 Spearman 30 × 30 RAW and WITHIN-PIECE ordered by family + 8 × 8 mean |ρ| (raw = repertoire; within-piece = how techniques co-vary MEC → EXG). **C4 pre-declared triangulation** — A and B overlap only if ≥ 2 of: (1) I(A,B) > 0 with the pair loss q < 0.05; (2) within-piece probe R²(A←B) or R²(B←A) ≥ 0.30; (3) within-piece family mean |ρ| ≥ 0.40; the pairs passing and the one-test pairs are printed. C5 `overlap_pairs.csv`, `overlap_probe(_family).csv`, `overlap_corr(_family).csv` + 3 figures.

**A7 — acceptance criteria for the next run (written here, not asserted):** Meditation_Thais 19 / 19 / 20 notes; per-condition totals ≈ 616 / 596 / 592 (small changes from A2 allowed); Friedman n.s. (V7 PASS); §5.1b (now §4.2 / §4.3b) note rate / IOI dose-response back to n.s. **PRE-DECLARED reading of the timing result:** in §5.6 (now §10.2) compare "Timing (ref-free)" alone F1 (piece) with 0.417 (Phase B) and 0.542 (B2). If it stays near 0.54, the timing signal is real; if it returns near 0.42, the B2 value was the vibrato artefact. Record which here, and quote only the B3 value in the thesis. *(Outcome as printed by the 2026-11-11 run — committed outputs, for the author to confirm: Timing (ref-free) alone F1 (piece) = **0.398** → closer to 0.417, i.e. the B2 value was the vibrato artefact of the re-articulation splits; deployed note-count Friedman χ²(2) = 0.535, p = 0.765 → V7 PASS. The other A7 items are not transcribed yet.)*

| Phase B2 | Phase B3 |
|---|---|
| plateaus + merge < 50 c / ≤ 50 ms + re-articulation dips ≥ 6 dB | plateaus + merge only if the merged range ≤ 80 c; **no re-articulation split** (`REART_ENABLED = False`; detector report-only) |
| vibrato window iff R² ≥ 0.50 and half-extent ≥ 3 c | … and the best rate is not on the 4 / 8 Hz grid edge (`edge`, `reason`) |
| `feat_crepe_v3.csv`, `dtw_features_v3.csv` | `feat_crepe_v4.csv`, `dtw_features_v4.csv` |
| §2.6 (now removed in C2) V1–V6 (V4 PASS / FAIL) | V1–V7 (V4 INFO, V2 / V3 ≥ 80 % span, V7 Friedman) |
| — | §4.4e (now §7.1) MERT generalisation · §5.6b (now §10.3) family overlap · §5.6 (now §10.2) commonality / chance lines |

### Phase C1 — empirical chance null + DIAGNOSTIC MERT layer study (2026-11-16; APPLIED; RUN 2026-11-16 — results in §4)

**Why the empirical null — evidence from the B3 run (2026-11-11, `piece`).** §4.4e (now §7.1)'s capacity control (25 i.i.d. N(0, 1) columns, seed 0) through the deployed NestedSelect scored **piece F1 0.354**, above the Dummy strict P95 **0.323** (uniform mean 0.272); on the validation curve the noise reached **0.468 at C = 0.001**, and NestedSelect picked C = 0.001 in 14 of the 20 piece folds for the noise. Likely mechanism: with a tiny C the balanced linear SVC is ≈ a class-mean-difference rule whose decision threshold follows the training-label composition of each leave-one-piece-out fold, and the labels cluster within pieces — so a classifier that learns nothing from the features still inherits the fold-wise label balance. The Dummy floor ignores the selection procedure and is therefore not an adequate null for the deployed classifier. Regression is unaffected (noise R² ≈ 0 under both protocols). The same run: same-piece nearest neighbour MERT-full 100 %, MERT-PCA25 90 %, interpretable 57 % (chance 3.4 %) — the pre-declared **piece-identity shortcut** rule fired, capacity overfitting did not; MERT F1 LOOCV 0.623 → piece 0.310.

**§4.5c (now §5.3 / §6.9) — how the null is built and read.** A null distribution of the **deployed path exactly as §4.4b (now §6.3) runs it**: `NULL_N_DRAWS = 100` draws per d ∈ `NULL_DIMS = (25, 10)` (25 = MERT-PCA25 / interpretable layer; 10 = CREPE / Essentia; DTW's 7 is closest to 10), each an i.i.d. standard-normal 60 × d matrix (seed `NULL_SEED` + draw, `NULL_SEED = 0`) through `cv_predict_nested_select` with the same `Y`, §4.0 (now §5.1) folds, 13 `SELECT_CONFIGS`, inner GroupKFold(5) and tie rule; score = the pooled test F1-macro computed as the headline; the NestedSelect config counts (and the folds at the smallest C) are recorded. `NULL_PROTOCOLS = ('piece',)`; `RUN_NULL_LOO = False` (≈ 3× the cost). Cached in `null_f1_draws.csv` and reused only if the settings hash (draws, dims, seed, protocols, `Y`, configs, inner splits, tie rule, piece groups) matches; the cell prints which. Draw 0 at d = 25 is asserted to be the §4.4e (now §7.1) random control (matrix and F1). New names only (`EMP_NULL_F1[(protocol, d)] = {mean, p95, p99, draws}`, `emp_null_dim`, `emp_null_p`; the `CHANCE_FLOOR_*` names are untouched), printed next to the Dummy floors with the share of draws above the Dummy strict P95. **Reading:** every reported classification F1 — the §4.1 (now §6.1) headline A–D, every §4.4b (now §6.3) lattice row (`MERT + DTW` is the §2.5 `X_F` matrix; §4.1 (now §6.1) has no X_F row), the §4.4e (now §7.1) blocks and the §4.4f (now §7.2) layer candidates — is compared with the null of the nearest width (d ≤ 17 → 10, d ≥ 18 → 25; wider blocks — MERT-full, `X_C`, `X_D`, the 30-feature layer — use the 25-column null as an extrapolation, flagged): empirical one-sided p = (1 + #draws ≥ observed) / (1 + 100), "beats the empirical null" = p < 0.05 (the smallest attainable p is 1/101); a protocol that was not drawn gets "n/a". → `emp_null_readout.csv`; §5.6 (now §10.2) draws the empirical P95 (d = 25) on its F1 panel and prints `alone_emp_p`; §7.1 (now §11.1) adds "Empirical-null p" / "Beats empirical null".

**MERT layer study — DIAGNOSTIC; `MERT_LAYERS` stays `[5, 6, 7]` until the author decides by hand.**
- **§2.1c (new, after §2.1b).** `extract_mert_layers(audio_path)`: one forward pass exactly as `extract_mert` (same loading, processor, device, pos-conv-restored model) → (13, 768) float32 time-means of every hidden state (0 = transformer input, 1–12 = layers; `len(out.hidden_states) == 13` asserted); cache `emb_mert_layers_v3.npz` (loaded if present). Consistency: the mean of the `MERT_LAYERS` rows equals `emb_mert[eid]` (max |diff| printed; `allclose(rtol=1e-3, atol=1e-4)` asserted; 0 is expected when both come from this session). Label-free; `extract_mert`, `emb_mert` and `emb_mert_v2.npz` unchanged.
- **§4.4f (now §7.2) candidates (25):** `L00`…`L12`, `W00-02`…`W10-12` (`W05-07` = deployed — asserted == `emb_mert` and then taken from it), `ALL`; references: the interpretable layer (the 30 features of §5.6 (now §10.2)) and `W05-07`.
- **Stage A (no ratings — piece / condition metadata only):** A1 same-piece nearest-neighbour share (cosine, self excluded; the §4.4e (now §7.1) shuffle scheme and seed, 10 000 shuffles keeping 3 per piece; silhouette by piece; descriptive; `W05-07` asserted == the §4.4e (now §7.1) MERT-full row) · A2 condition decoding across pieces — piece-centred embeddings (minus the piece's own 3-rendition mean; label-free), §4.0 (now §5.1) LOPO folds, a parameter-free nearest-centroid (cosine) classifier with MEC / EXP / EXG centroids from the training pieces; accuracy vs 1/3; 1000 within-piece permutations of the condition labels (seed `SEED`) · A3 share of pieces whose cosine d(MEC, EXG) is the largest of the three within-piece distances (vs 1/3; same permutations) · BH over the 25 for A2 and A3. **Pre-declared suggestion rule** (printed before the results): among the candidates with A2 q < 0.05 the lowest A1 share; if others are within 5 percentage points, prefer a 3-layer window (ties: A1, piece silhouette, candidate order); none passes → "no layer encodes condition beyond chance — keep W05-07". Printed as "SUGGESTED — not applied".
- **Stage B (descriptive; not used to choose):** each candidate's MERT block built as deployed — classification: the §2.1b PCA 95 % replicated (fit once on the 60, transductive, label-free, float32 rows as `bfv`); regression: the 768 dims via `MERT_REG_REPR` + nested ridge — through `ablate()` in tier `6-mert-layers` (+ the interpretable reference row), `piece` and `loo`, Δ vs W05-07 by `abl_vs_ref` (paired permutation, BH within tier × protocol), `ablation_master.csv` re-exported. **HARD CHECK:** the W05-07 rows reproduce the §4.4b (now §6.3) `MERT` row exactly (predictions, F1, r, R², RMSE). Reported per candidate: piece F1, LOOCV F1, the gap, R²v / R²a under both protocols, NestedSelect config counts, and (from §4.5c (now §5.3 / §6.9)) the empirical-null p of the piece F1 and its BH q across the 25. Pre-declared "beats W05-07 beyond the empirical null": F1 — ΔF1 > 0 with paired q < 0.05 **and** empirical-null q < 0.05; R² arousal — ΔR² > 0 with paired q < 0.05.
- **Nested layer selection (stability; `piece` only).** Outer = the §4.0 (now §5.1) piece folds; inner = LOPO over the 19 training pieces (the §4.0 (now §5.1) `get_cv('piece')` splitter applied to the training rows); **fixed estimators** — `SVM-Linear C=1` (balanced) on a per-candidate PCA 95 % fitted on the inner training rows only (scaler inside) for the classes; `StandardScaler → RidgeCV(RIDGE_ALPHAS)` on the 768 dims for arousal and valence separately; choice by pooled inner macro-F1 / SSE (ties → candidate order), refitted on the outer training fold. Reported: the nested outer F1 / R²a / R²v next to the same fixed estimators always on W05-07, the chosen candidate per fold and the frequency table. A selection-stability diagnostic — not comparable with the headline. Cached in `mert_layer_nested.csv` (+ `mert_layer_nested_pred.csv`).
- **Outputs** (`OUTPUT_DIR/mert_layers/`): `mert_layer_study.csv`, the two nested files, `fig_mert_layer_profile.png` (x = hidden state; windows at their centre layer, ALL at 13; references: interpretable, W05-07, chance, Dummy P95 and — redrawn by §4.5c (now §5.3 / §6.9) — the empirical P95); a one-line answer each to: where piece identity is lowest · where condition is encoded best · whether any candidate beats W05-07 on piece F1 / R²a beyond the empirical null · how stable the nested choice is.

**Other C1 changes.** `RUN_MANUAL_AUDIT = False` (§2.6 (now removed in C2) skipped, cells kept) · §4.4c (now §6.4)'s "per-layer MERT pending" text updated · the intro, §4.5 (now §6.6) and output-reference markdown updated.

**Runtimes (T4, 4 CPUs).** §2.1c ≈ 0.5–1 min · §4.4f (now §7.2) ≈ 40–55 min (Stage A 1–2, Stage B 35–45 with both protocols, nested 3–10) · §4.5c (now §5.3 / §6.9) ≈ 50–70 min (`RUN_NULL_LOO` ≈ +3 h) · total ≈ the B3 run (127 min) + 1.5–2 h. Quick run: `MERT_LAYER_PROTOCOLS = ('piece',)`, `RUN_MERT_LAYER_NESTED = False`, `NULL_DIMS = (25,)`.

### C1 checklist — run 2026-11-16 (outcome; numbers in §4)

- [x] **§4.5c (now §5.3 / §6.9):** computed. Empirical P95: d = 10 0.370, d = 25 0.459 (Dummy strict P95 0.323). 36 / 52 F1 beat the Dummy P95; 4 / 52 (3 distinct, all containing the reference-free timing block) beat the empirical null; the random control has p 0.58.
- [x] **§4.4f (now §7.2) Stage A:** the rule was printed; lowest A1 L10 93.3 %; best A2 L05 0.900 (25 / 25 with q < 0.05); best A3 L04 0.85; **SUGGESTED — not applied: W09-11**.
- [x] **§4.4f (now §7.2) Stage B:** none beats W05-07 on piece F1 beyond the empirical null (0 / 25 beat the null; W05-07 p 0.832). R²a: L04, W03-05 and W04-06 beat W05-07 (paired q < 0.05).
- [x] **§4.4f (now §7.2) nested:** classes L00 5 / 20 (9 distinct), R²a L11 9 / 20, R²v L11 16 / 20 → unstable.
- [x] **Asserts:** the run reached the §4.4f (now §7.2) summary and §4.5c (now §5.3 / §6.9), so the §2.1c consistency assert and the Stage B HARD CHECK did not fail.
- [ ] **Not transcribed** (the outputs were not reported in chat): §2.1c max |diff|, §5.6 (now §10.2) `alone_emp_p`, the §7.1 (now §11.1) column, the other guards, the total runtime — read them in the committed notebook.
- [ ] **The author's decision on `MERT_LAYERS`** (recommended: keep [5, 6, 7]) — record it here.

### B3 checklist (N = 116 + Phase B3) — run 2026-11-11; transcribed 2026-11-17 (numbers in §4, "Current results")

Not transcribed: the DTW forwarding count and the full calibration grids (deployed cells: 0.7 % vibrato windows / 2.0 % glides). A7 outcome: Méditation 19 / 20 / 21; Friedman p 0.765 (V7 PASS); note rate / IOI still significant in §5.1b (now §4.2 / §4.3b) (raw p, a duration effect); timing reading → B2 artefact.

Interactive run up to §2.6 (now removed in C2):
- [x] §2.2 banner "DISABLED"; Viterbi-source line and octave fixes unchanged.
- [x] §2.2 evidence table: detector OFF ≈ 616 / 596 / 592 (A2 may move them a little), ON ≈ 640 / 647 / 660 with ≈ 24 / 51 / 68 splits; 3 / 6 / 9 dB Mozart_VC4 rows; asserts ✅.
- [x] §2.2 condition-independence: totals, per-piece range, **Friedman n.s. ✅** (and the detector-on Friedman).
- [x] §2.2 vibrato: band-edge rejections (all / MEC / edge-only); rate never exactly 4.00 / 8.00; calibration deployed cell ≤ 5 %.
- [x] §2.2 close splits (A2 can add a few), portamento verdicts; §2.4 `dtw_features_v4.csv`, DTW forwarding 3–10; §2.5 asserts ✅.
- [x] §2.6 (now removed in C2): asserts ✅; Meditation_Thais **19 / 19 / 20**; V3 C♯6 ≤ 80 c; V2 / V3 vibrato with ≥ 80 % span; V4 INFO ≈ 17 / 16 / 20; V6 (≈ 5 MEC — known); **V7 PASS**.
Full run:
- [x] §5.1b (now §4.2 / §4.3b) note rate / IOI back to n.s.?; vibrato extent / coverage / rate z.
- [x] §5.6 (now §10.2) pre-declared timing reading (record above); commonality summary; figure lines / `*`.
- [x] §4.4e (now §7.1): asserts ✅; gaps; validation curve; NN shares; **which rule fires**.
- [x] §5.6b (now §10.3): runtime; I heatmaps; probes; correlations; **triangulation**; master table re-exported.
- [x] All other guards ✅; total runtime (≈ the B2 run + 15–20 min).

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

Variable names (`DTW_REFFREE`, `df_dtw`, `DTW_*`) and the lattice block name `DTW` were kept. Deviations from the literal Phase B spec (stated in the cells): vibrato notes ≥ 800 ms instead of 400 (presence resolution at fs = 100 Hz); tempogram statistics over interior frames only (edge frames ramp to zero and fake tempo variability); the raw diagnostic flux is computed on the gated frames as well; `feat_crepe_v2.csv` is recomputed from the cached tracks every run; the timing features raise if an excerpt has < 3 notes; §6.2 (now §7.5) takes its lattice-equivalent blocks from the §4.4b (now §6.3) stacks (identical arrays, so the §6.2b (now removed in C2) equality guard is not exposed to float32 rounding flipping a discrete inner-fold choice).

### Phase B Kaggle run (N = 118, 2026-10-09/10) — checklist outcome

- [x] §2.2 octave corrections 3 frames / 3 excerpts; vibrato rate 37 distinct values (> 20); half-extent 0–32.1 c, coverage 0–1; substitutions: no note ≥ 800 ms 1/60 (no ⚠️), no vibrato note 23; notes 391 / 259 / 198 at 600 / 800 / 1000 ms; F0 range max 3100 c (old 5343 c); 17 close splits.
- [x] §2.2 portamento extents 80–1907 c; glides on 24.3 % of transitions (median 62 ms) → confirmed false alarms by the 2026-11-05 audit (Viterbi ramp) → fixed in Phase B2 (B9b).
- [x] §2.3 r(flux, LUFS) 0.874 raw → 0.508 normalised; DR max 34.7 dB, median 6.2 % gated (max 26 %); DR sensitivity ρ down to 0.79 → under audit.
- [x] §2.3c 0/60 conclusive at FLAT_MIN 0.2 (9 at 0.1), 0 flags, session map not provided → under audit.
- [x] §3.3b timing manipulation check: 0/3 variability features non-decreasing; tempo log2-SD falls MEC → EXG → author's explanation + under audit.
- [x] §4.0 (now §5.1) 4 CPUs, parallel == serial ✅.
- [x] NestedSelect beats the strict P95 under `piece` only on X_B (0.361); X_C 0.302 does not.
- [x] MERT-full in the lattice regression heads (`n_features_reg` 768–795), rebuild ✅ — and every MERT + X head scored like MERT alone (→ block weighting).
- [x] §4.9 (now §8.1) B-INC: Δvalence incremental q 0.037 (Interpretable, MERT-full, Interp + MERT-PCA); Δarousal q 0.070; ceilings V 0.455, A 0.678.
- [x] §5.1b (now §4.2 / §4.3b) vibrato extent / coverage escalate (z +5.14 / +4.90); §5.6 (now §10.2) vibrato alone weak; §6.1b (now §7.3b) vibrato coverage R² 0.48, rate −0.02 (`CREPE_UNIQUE2` → pending decision).
- [x] Runtime ≈ 97 min.

### B2 checklist (N = 116 + block weighting + Phase B2 + §2.6 (now removed in C2) audit) — run in full by the author; kept for reference (superseded by the B3 checklist above)

Interactive run up to §2.6 (now removed in C2) first (the §1 items already held in the 2026-11-03 run):
- [ ] §1.1 prints 2 excluded (`questionnaire_5_P012`, `questionnaire_5_P020`), 118 → 116 participants, 2360 → 2320 rows, per form 20 / 21 / 26 / 19 / 30; the three asserts pass.
- [ ] §1.1b maps ≈ V 2.94 + 1.57z, A 3.39 + 1.53z; §1.2b anchors V 0.342 → 0.338, A 0.276 → 0.253; 8/60 class changes; §1.2 classes HA_LV 20, LA_LV 17, HA_HV 11, Neutral 7, LA_HV 5 (`Schindlers_List_EXP` HA_LV → LA_LV); §1.4 (now §4.3a) prints the exclusions.
- [ ] §2.2 (B2): Viterbi-source line ("12 - abs" found); `rms10_v3.npz` computed (length assert ✅); segmentation old vs new counts (total / per condition / per excerpt), re-articulation splits, `REART_DIP_DB` 3 / 6 / 9 sensitivity incl. Mozart VC4 (score 19 each), close splits; vibrato analysed notes at 500 / 650 / 800 ms, windows / vibrato windows, substitution counts (⚠️ > 10 with no note ≥ 500 ms), rate inside 4–8 Hz; portamento verdict counts (expect many "viterbi-limited jump"), glide rise / excess / extent; **calibration grids: deployed false-positive rates ≤ 5 %? (⚠️ otherwise — discuss the constants before the full run; nothing is auto-tuned).**
- [ ] §2.4: timing recomputed with the B2 onsets (`dtw_features_v3.csv`); DTW forwarding count 3–10.
- [ ] §2.6 (now removed in C2): every recompute / detail-consistency assert passes (windows → coverage / rate / extent, window fit → R², transitions → portamento); declared substitutions listed; LUFS diff − 3 dB ≈ 0; listen to the players (onset clicks, beat clicks, glide tone); read the overview (vibrato windows, re-articulation splits, env_db) / zoom figures (fitted sinusoids), the per-piece ✓/✗ tables and the 20-piece exceptions; **AUTHOR VALIDATION V1–V6 all PASS** (+ the F♯5→A5 / A4→B4 old vs new verdicts); fill `manual_audit/audit_checklist.csv`.
Full run:
- [ ] §4.0 (now §5.1) class prevalences 11 / 20 / 5 / 17 / 7 + analytic uniform F1; block-weighting guard ✅; printed weights (MERT-full ≈ 0.51, CREPE / Essentia ≈ 4.46, DTW ≈ 5.33).
- [ ] §4.4b (now §6.3) `[w]` on the 7 MERT-full multi-block rows; MERT + X R²_aro no longer equal to MERT alone?; α grid-edge report.
- [ ] §4.4c (now §6.4) tier 4E: Δ weighted − unweighted (r, R², q) per set and protocol; deployed == lattice ✅; α edge hits.
- [ ] §4.4d (now §6.5) radius rule; all other guards; NestedSelect vs strict P95; §4.9 (now §8.1) B-INC; §5.5 (now §10.1) CoE; §5.6 (now §10.2); §6.1b (now §7.3b); total runtime.
- [ ] **§5.1b (now §4.2 / §4.3b) dose-response re-check for the vibrato family** (the abstract claims vibrato extent / coverage escalate MEC → EXP → EXG — Phase B z +5.14 / +4.90 with the faulty detector); §5.6 (now §10.2) Vibrato / Portamento rows (Phase B: removing Portamento raised F1); §6.1b (now §7.3b) vibrato / portamento probe and `CREPE_UNIQUE2`.

### Phase A Kaggle run (2026-10-09) — checklist outcome

- [x] §2.1 restore message printed; the embeddings turned out identical to the pre-fix run (false alarm).
- [x] §2.3c: 0 LUFS outliers, 0 clipping, 21 noise-floor flags (15 inconclusive) — estimator replaced in Phase B (B-NF).
- [x] §4.0 (now §5.1) equivalence assert passed; Dummy floors vary across folds (uniform 0.272 / stratified 0.183 under loo).
- [x] §4.5b (now §5.2 / §6.6): no default-classifier row beats the strict P95 under `piece` (NB below chance everywhere).
- [x] §4.4b (now §6.3) lattice, §4.4d (now §6.5) label-reproduction, §6.2b (now removed in C2) guards passed.
- [x] Tier 4D: balanced SVM + inner-CV C beats NB under `piece` on all three matrices.
- [x] §4.9 (now §8.1): Interpretable beats Condition-only descriptively for both targets (untested → B-INC).
- [x] §5.5 (now §10.1): CoE negative under `piece` (interpretable wins).
- [x] §6.1b (now §7.3b): α hit the grid edge (count superseded: 34 outer folds in the Phase B run).
- [x] Runtime: 45.3 → 75.5 min (+30 min), dominated by §4.1 (now §6.1) inner-CV tuning and the 4D tier.

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

---

## 13. Literature hooks and thesis implications (2026-11-17)

Candidate references for findings and methods that arose in Phases B2–C1. **Verify each one (authors, year, venue) and check it against the thesis `.bib` before citing.** Several may already be there: Ch. 6 covers nested CV, chance floors and Freedman–Lane.

| Where it is used in the pipeline / thesis | Candidate reference(s) |
|---|---|
| CREPE and its Viterbi decoder (slew ≤ 11 bins = 220 c per 10 ms → every jump becomes a ramp; B9b glide rule) | Kim, Salamon, Li & Bello 2018, "CREPE: A Convolutional Representation for Pitch Estimation", ICASSP (code: `crepe/core.py`, `to_viterbi_cents`) |
| Violin vibrato also modulates the amplitude through the body resonances → why an energy envelope cannot find re-articulations (B3 negative result) | Fletcher & Sanders 1967, "Quality of violin vibrato tones", JASA · Mellody & Wakefield 2000, "The time-frequency characteristics of violin vibrato: modal distribution analysis and synthesis", JASA |
| Vibrato rates / widths of string players (the 4–8 Hz band, the ±3 c floor) | MacLeod 2008, "Influences of dynamic level and pitch register on the vibrato rates and widths of violin and viola players", JRME · Geringer & Allen 2004, vibrato among high-school / university violin and cello students, JRME |
| Tempo / timing as primary emotion cues (why only the timing block beats the empirical null) | Juslin & Laukka 2003, Psychological Bulletin (communication of emotions in vocal expression and music performance) · Gabrielsson & Lindström 2010, Handbook of Music and Emotion (structure and emotional expression) |
| Theoretical chance levels exceeded "by chance" in small samples → empirical null of the whole pipeline | Combrisson & Jerbi 2015, "Exceeding chance level by chance…", J. Neurosci. Methods · Ojala & Garriga 2010, "Permutation tests for studying classifier performance", JMLR · Varoquaux 2018, "Cross-validation failure: small sample sizes lead to large error bars", NeuroImage |
| Selection bias, nested model selection | Cawley & Talbot 2010, JMLR · Varma & Simon 2006, BMC Bioinformatics |
| Piece / artist / album leakage; "horses" in MIR (piece-identity shortcut; why LOPO) | Flexer 2007, artist filters for genre classification, ISMIR · Sturm 2014, "A simple method to determine if a music information retrieval system is a 'horse'", IEEE TMM |
| Shortcut learning in deep networks (MERT §4.4e (now §7.1) / §4.4f (now §7.2)) | Geirhos et al. 2020, "Shortcut learning in deep neural networks", Nature Machine Intelligence |
| Layer-wise probing of music SSL models / MERT depths | Castellon, Donahue & Liang 2021, ISMIR (Jukebox representations for MIR) · Li et al., MERT (arXiv 2023; ICLR 2024) |
| Multiple testing (§5.1 (now §9.1; its head → §4.1), §5.1b (now §4.2 / §4.3b), the ablation tiers) | Benjamini & Hochberg 1995, JRSS-B |
| Shared vs unique variance (§5.6 (now §10.2) commonality) | Nimon et al. 2008, commonality analysis (Behavior Research Methods) |
| Binary relevance vs multiclass for single-label data | Tsoumakas & Katakis 2007, "Multi-label classification: an overview" · Zhang & Zhou 2014, IEEE TKDE (multi-label review) |
| Page's trend test (§5.1b (now §4.2 / §4.3b)) | Page 1963, JASA |
| Freedman–Lane permutation (§4.9 (now §8.1) B-INC) | Freedman & Lane 1983 · Winkler et al. 2014, NeuroImage |

**Thesis wording to update once these results are final:**
- **Abstract:** "against explicit chance floors" → name the empirical null of the pipeline.
- **Ch. 6:** add the empirical-null subsection next to the Dummy floors.
- **Ch. 9:**
  - classification is reported against the empirical null (only the timing block passes);
  - regression / trend tests carry the claims;
  - the MERT shortcut (§4.4e (now §7.1) + §4.4f (now §7.2));
  - the common-cause limit (§5.6b (now §10.3));
  - the timing artefact caught by the manual audit, as a methodological lesson.
- **The "all-positive ≈ 0.322" floor** cited in the thesis: drop it (superseded).