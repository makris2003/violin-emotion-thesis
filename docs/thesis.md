# Violin Emotion Thesis — Pipeline Overview

> **Latest (2026-11-20): Phase C2 — APPLIED, NOT RUN** ("Phase C2" in §10 of CLAUDE.md / the "Phase C2" section of docs/thesis.md). The notebook was restructured so its order is the story order — **12 parts**: setup · ground truth · features · tool validation · **manipulation check (Part 4)** · **evaluation yardstick (Part 5: CV engine, Dummy floors, the empirical null — drawn before any F1)** · prediction · MERT diagnostics · within-piece · which techniques · cost of explainability · synthesis (claims table). Content: predictive (27) / descriptive (30) interpretable layer (replaces `DTW_DEV_IN_INTERP`), BH in the Page tests, an **exact-width 1000-draw primary null** with BH in a pre-declared family, tempogram ⚑ + sensitivity rows, deployed nested ridge in §6.2 / §6.8 / Fig 5, rewritten §7.4 / §7.5 / §9.3, §11.1 claims table; §2.6 audit, the 8-classifier Wilcoxon and §6.2b removed. 118 cells. **No result number below has changed — nothing has run**; the numbers the C2 run will replace are flagged in "Current results" and listed in the C2 run checklist. Old section numbers in the historical text are annotated "(now §x.y)". **Next:** the C2 run (the primary null takes ≈ 15 h → two sessions, resumable).
>
> **Revision status (2026-11-17): Phase B3 (2026-11-11) and C1 (2026-11-16) have been RUN — results
> transcribed in "Current results" below.** Highlights:
> - **Classification:** read against the §4.5c (now §5.3 / §6.9) empirical null (P95 0.370 at 10 columns / 0.459 at 25; Dummy
>   0.323), only the sets with the reference-free timing block beat chance (DTW 0.398, CREPE + DTW 0.460,
>   Essentia + DTW 0.373, piece).
> - **Regression** carries the claims (interpretable R²a 0.648, R²v 0.258 on unseen pieces).
> - **MERT:** the piece-identity shortcut is present at every depth (§4.4f (now §7.2), 93–100 %) → recommended: keep
>   `MERT_LAYERS = [5, 6, 7]` (author to confirm).
> - **Timing:** the B2 F1 0.542 was the vibrato artefact (B3 0.398).
> - **Overlap:** the families share a common cause (§5.6b (now §10.3)).
> Consolidated limitations / future work / remaining code obligations: "Limitations, future work and remaining
> code obligations"; literature: "Literature hooks". **Next: Phase C2.**
>
> **Earlier (2026-11-16): Phase C1 — APPLIED (since RUN)** (see "Phase C1" below): an
> **empirical chance null** for the deployed classification pipeline (§4.5c (now §5.3 / §6.9)) and a **DIAGNOSTIC MERT
> per-layer study** (§2.1c per-layer cache + §4.4f (now §7.2), tier `6-mert-layers`); `RUN_MANUAL_AUDIT = False`.
> **`MERT_LAYERS` stays `[5, 6, 7]`** — the author decides by hand after reading §4.4f (now §7.2). Notebook now
> **114 cells**. **Next: the full run** (no interactive stop at §2.6 (now removed in C2) needed — the audit is skipped).
>
> **Phase B3 — RUN on Kaggle 2026-11-11** (papermill 08:29 → 10:36 UTC, ≈ 127 min, 108 cells, 0 errors;
> outputs committed in `58cb2e7`). The B3 numbers that motivate C1 are quoted in "Phase C1" below; the
> rest of the B3 checklist is not yet transcribed into these docs (the outputs are in the notebook).
>
> **Revision status (2026-11-03): Phase B has been RUN on N = 118** (Kaggle, Tesla T4, papermill
> 2026-10-09 23:45 → 2026-10-10 01:22 UTC, ≈ 97 min, 101 cells, 0 errors; outputs saved in the
> committed notebook). Its results (below) are **N = 118 / pre-block-weighting**.
>
> **Committed notebook = an N = 116 run of 2026-11-03** (papermill 15:26 → 17:11 UTC; §1.1 shows
> 118 → 116, F5 32 → 30) with (1) the participant exclusions in §1.1 (`EXCLUDED_PARTICIPANTS`),
> (2) equal-share block weighting of the multi-block MERT-full regression heads
> (`REG_BLOCK_WEIGHTING = 'equal'`, tier `4E-block-weighting`) and (3) the **temporary §2.6 (now removed in C2) manual
> descriptor audit**. Its §2.6 (now removed in C2) outputs are the evidence of the **manual audit of 2026-11-05**; its
> Part 2–7 numbers are **not recorded here** and are superseded by Phase B2 (they still use the
> Phase B CREPE descriptors).
>
> **Phase B2 — descriptor repairs after the manual audit — APPLIED 2026-11-05 and RUN IN FULL on
> Kaggle by the author (N = 116; the run's outputs are NOT committed — the committed notebook still
> carries the 2026-11-03 outputs for the unchanged cells).** Its key numbers are quoted in the
> "Phase B3" section below (evidence (a)–(e)).
>
> **Phase B3 — APPLIED 2026-11-10 (since RUN, 2026-11-11 — see above)** (see the section below): the B2 re-articulation
> detector is **disabled** for every feature (`REART_ENABLED = False`, kept report-only), the plateau
> merge re-checks the 80 c range, vibrato windows at the 4 / 8 Hz grid edge are rejected, caches
> `feat_crepe_v4.csv` / `dtw_features_v4.csv`, and two new diagnostic sections: **§4.4e (now §7.1)** MERT
> generalisation (train vs test, validation curve, NN identity) and **§5.6b (now §10.3)** family overlap (pair
> removal tier `3-subfamily-pairs`, cross-family probes, correlations). Notebook now **108 cells**.
> **Next: run interactively up to §2.6 (now removed in C2) (A7 acceptance criteria), then the full run (§4.4e (now §7.1) / §5.6b (now §10.3)
> rules, the pre-declared timing reading).** The thesis text (Ch. 7 Dataset) already reports N = 116
> numbers; details in `docs/thesis.md` §10–§12.

## Project
Undergraduate **diploma thesis**, School of Electrical and Computer Engineering, NTUA (supervisor
G. Stamou): *Expressive Techniques in Solo Violin Performance and Their Effect on Perceived Emotion*.
Chain: **expressive technique → acoustic result → perceived audience emotion**, studied with an
**interpretable** ML pipeline. Two categorisation deliverables (tools for detecting technique;
modalities for measuring audience effect). Framing is **categorisation + interpretable attribution +
cost of explainability**, NOT a leaderboard — MERT is one tool category (the opaque SSL reference).
Questions: (1) which techniques escalate MEC → EXP → EXG and which reach listeners as valence /
arousal; (2) can perceived emotion be attributed interpretably, and what does explainability cost;
(3) where do the interpretable features and the SSL representation agree or complement each other.

## Dataset
60 excerpts = 20 pieces × 3 conditions (MEC mechanical / EXP expressive / EXG exaggerated),
recorded by the author playing violin. 118 responses across 5 **linked (ring)** Google Forms,
**116 after screening** (exclusions applied in §1.1 since 2026-11-03), rated **valence 0–6,
arousal 0–6** (midpoint 3.0; `0` is a real rating, not NA) + 1–3 emotion tags (7 GEMS terms + Neutral).
Ring: each form = all 20 pieces, one rendition each (8 MEC, 8 EXP, 4 EXG). Renditions follow a
cyclic rotation of (MEC, MEC, EXG, EXP, EXP) over F1–F5, shifted by one form for each of 5 groups
of 4 pieces → every MEC/EXP excerpt is in 2 adjacent forms (40 anchors, 8 shared per neighbour,
F5↔F1), every EXG excerpt in 1 form (4 unique per form). Forms assigned by a uniform random URL
redirect (Google Apps Script), no balancing, no log.
Ground truth = **5 classes** (4 circumplex quadrants + Neutral) from **normalized** VA means.
Stimuli were **NOT loudness-normalized** for the participants (level is part of the stimulus).

## Pipeline (notebook/thesis_pipeline.ipynb, Kaggle GPU, 118 cells since Phase C2 — applied, NOT RUN)

Runs top-to-bottom; **the notebook order is the story order** (Phase C2 — old → new map in the "Phase C2" section).
`§` = the in-notebook section banners. 118 cells (67 code, 51 markdown).

| Part | § | Content |
|---|---|---|
| **0 Setup** | 0.1–0.4 | install / imports / **config** (`VA_MID=3.0`, `NEUTRAL_RADIUS=0.75`, `MERT_LAYERS=[5,6,7]`, tag maps; C2: `TEMPOGRAM_UNRELIABLE`, `NULL_N_DRAWS_PRIMARY = 1000`, `CREPE_UNIQUE_R2_MAX = 0.2`, the shared `bh_fdr`, the unsupervised forwarding gate) / audio manifest |
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
| | 5.3 | empirical null of NestedSelect on noise: 100-draw map (d = 10 / 25, `null_f1_draws.csv`) + **exact-width 1000-draw primary null** (A–D, Interpretable (27), DTW, CREPE + DTW; `null_f1_draws_primary.csv`, resumable) · `emp_null_ref` / `emp_null_mark` |
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

- **Cleaning (outside the notebook):** `questionnaire_cleaning_v3.py` (qc_pack, not in the repo)
  supersedes the participant screening: D1 within-form LOO person-total r < 0.10 · D2 MCD distance
  on form-centred ratings > simulated 99.9th pct (14.55) · D3 scale reversal · D4 mean IRV outside
  [0.25, 2.20] · D5 LongString ≥ 8 in answer order; exclude at ≥ 2 flags → `questionnaire_5_P012`
  (D1+D2), `questionnaire_5_P020` (D1+D3, reversed valence). `scripts/questionnaire_cleaning.py`
  and `scripts/excerpt_cleaning.py` are the original (older) versions. **Decision: the excerpt QC
  (ICC, r_wg, bimodality coefficient, Kruskal–Wallis, crossed additive model) stays OUTSIDE the
  notebook** (offline scripts; numbers in `docs/thesis.md` §11).

## Evaluation protocol (§4.0 (now §5.1) — read before quoting any Part 4–6 number)
- **`PRIMARY_PROTOCOL='piece'`** (leave-one-piece-out, `LeaveOneGroupOut` on `PIECE_GROUPS`);
  `loo` is always reported next to it as the optimistic secondary; `PROTOCOLS=('piece','loo')`,
  printed piece first, with the optimism gap. `get_cv(protocol)` is the ONLY splitter builder
  (`'piece5'` = GroupKFold(5) on pieces, used only by the §6.1b (now §7.3b) probe).
- **Default classifier: nested model selection**, `DEFAULT_CLF_NAME='NestedSelect'`,
  `cv_predict_default(X, Y, protocol)` = `cv_predict_nested_select(…, configs=SELECT_CONFIGS)`.
  Inside every outer training fold an inner GroupKFold(5) over the training pieces scores each of
  the 13 `SELECT_CONFIGS` (NB · SVM-Linear balanced · LogReg balanced, C ∈ `logspace(-3,2,6)`) by
  pooled inner OOF macro-F1; ties → list order; the winner is refitted on the outer training fold.
  sha1 memo + joblib/loky over outer folds (`NMS_N_JOBS = os.cpu_count()`), parallel == serial
  asserted on X_B.
- **NaiveBayes** (`NB_CLF_NAME`, `make_nb_clf`) = the Phase A pre-declared default; failed under
  `piece`; stays a reported row.
- `cv_predict_multilabel(make_est, X, Y, protocol, tune_grid=None)` — binary relevance, optional
  per-label inner-CV C; asserted equal to the old `cross_val_predict(SafeMultiOutput(GaussianNB))`.
- `nested_ridge_predict(X, y, protocol, alphas, make_pipe=None, block_sizes=None)` — α ∈
  `RIDGE_ALPHAS=logspace(-2,4,13)` chosen inside each outer training fold by inner GroupKFold(5)
  (pooled SSE; ties → largest α). `NestedRidgeOperator(X, protocol, alphas, block_sizes=None)` =
  its closed-form twin.
- **Block weighting (new, 2026-11-03, not yet run):** `block_weights(sizes)` → w_b = √(P/(B·p_b))
  (P = Σ p_b columns, B blocks); `BlockScaler(sizes)` = stateless transformer applied AFTER
  StandardScaler (`StandardScaler → BlockScaler → Ridge`); `block_sizes` threads through
  `_ridge_pipe`, `select_ridge_alpha`, `nested_ridge_predict`, `NestedRidgeOperator`. B = 1 →
  identical to unweighted. Total standardised variance stays P (α grid keeps its meaning); group
  ridge with α_b ∝ B·p_b/P. Weights depend only on column counts — leakage-free. Config
  `REG_BLOCK_WEIGHTING = 'equal'` (§0.3; `'none'` restores the Phase B heads). **Scope:** only
  regression heads where MERT-full is concatenated with ≥ 1 other tool block (MERT / CREPE /
  Essentia / DTW): the §4.4b (now §6.3) lattice regression heads (via `_abl_reg_kw`) and `X_C_REG` in §4.4d (now §6.5).
  Unchanged: single-block heads, interpretable-only sets, PCA-MERT sets, every classification head,
  §4.2 (now §6.2), §4.9 (now §8.1), §5.4 (now §9.2), §5.5 (now §10.1). §4.0 (now §5.1) guard on `X_C_REG`: B = 1 exact, Σ p_b·w_b² = P, equal share,
  BlockScaler stateless, weighted operator == weighted sklearn path.
- **B-MERTFULL:** the regression heads see MERT at full dimensionality (`MERT_REG = X_MERT_FULL`);
  the classification heads keep the PCA-forwarded `X_A`.
- `freedman_lane_targets` (B-INC) and `split_half_ceiling` (B-NC) live in the HELPERS block.
- Metrics: macro-F1; r **and** `oos_r2` (baseline = training-fold mean) **and** `rmse`.
- **Chance floors:** Dummy seeded `base + 1000·fold + label`, 200 repeats → mean / SD / P95;
  `CHANCE_FLOOR_P95` = max P95 over strategies (strict "beats chance" line). §4.0 (now §5.1) now prints the
  class prevalences of `Y` and the analytic uniform/stratified F1 derived from them (nothing
  hardcoded).
- **Empirical null (C1, §4.5c (now §5.3 / §6.9)):** the Dummy floors ignore the selection procedure and are too low
  for NestedSelect (B3: 25 noise columns → piece F1 0.354 > Dummy strict P95 0.323). `EMP_NULL_F1`
  = {(protocol, d): {mean, p95, p99, draws}} from `NULL_N_DRAWS = 100` i.i.d. N(0, 1) matrices per
  d ∈ `NULL_DIMS = (25, 10)` through `cv_predict_nested_select` (`NULL_SEED = 0`, draw i → seed
  0 + i; `NULL_PROTOCOLS = ('piece',)`, `RUN_NULL_LOO = False`). Read every classification F1 with
  `emp_null_p(f1, protocol, n_features)` against the nearest d (`emp_null_dim`: ≤ 17 → 10, ≥ 18 → 25,
  > 25 extrapolated); "beats the empirical null" = p < 0.05; a protocol without draws → NaN / "n/a".
  The `CHANCE_FLOOR_*` names are unchanged.

## Key variables
- `df_long` — per-rating, **screened** (§1.1); `EXCLUDED_PARTICIPANTS` (§0.3), `N_PARTICIPANTS_RAW`
  (= 118, §1.1). `PARTICIPANT_DATA = df_long` (§1.4 (now §4.3a); the old `df_long_clean` branch is gone).
- `df_agg`/`df_all` — per-excerpt; `valence_mean/arousal_mean` = NORMALIZED (ground truth),
  `*_raw` kept in parallel. `emotion_labels` → `Y`; `top_tags` descriptive only.
- `Y` (60×5), `mlb.classes_`. Full blocks: `df_mert`, `df_crepe`, `df_essentia`, `df_dtw`.
  Forwarded: `emb_mert_fwd`, `df_crepe_fwd`, `df_ess_fwd`, `df_dtw_fwd`; lists `*_FORWARD`.
- CREPE: `CREPE_CANDIDATES` (10), `CREPE_LABELS`, `CREPE_DIAGNOSTIC = ['crepe_vibrato_n_notes']`;
  `CREPE_TRACKS`, `CREPE_DIAG`, `CREPE_CENTS`, `CREPE_VOICED`, **`CREPE_ENV_DB`** (10 ms RMS dB on
  the CREPE grid, B2; report-only since B3), `CREPE_VITERBI_SOURCE_OK` (report only);
  `VIB_N_SUBST_NONOTES` / `VIB_N_SUBST_NOVIB`; `_med_rate` (the declared median vibrato-rate
  substitute, read by §2.6 (now removed in C2)). B3: `REART_ENABLED` (HELPERS, `False`), `CREPE_REART_REPORT`
  ({dB: per-excerpt note counts with the detector forced on}), `CREPE_NOTE_COUNTS` (pieces × MEC/EXP/EXG,
  deployed), `CREPE_NOTECOUNT_FRIEDMAN` (dict: `statistic, p, ok, statistic_on, p_on, totals, totals_on,
  range_median, range_max, n_pieces` — read by §2.6 (now removed in C2) V7).
  `CREPE_DIAG[e]` keys (B2 + B3): `n_octave_fixed, n_vib_notes, n_windows, n_vib_windows,
  n_edge_windows, n_edge_only, n_notes_60, n_notes_60_v2, n_reart_splits (0 while disabled),
  n_close_splits, n_transitions, n_glides, glide_durs, glide_exts, glide_excess, transitions,
  vib_windows (r2, extent, edge), notes_by_min` (the Phase B key `n_vib` is gone).
- Diagnostic-only columns: `CREPE_DIAGNOSTIC`, `ESSENTIA_DIAGNOSTIC =
  ['essentia_spectral_flux_raw_mean']`, `TIMING_DIAGNOSTIC = ['timing_octave_jump_frac']`.
- Assembly: `X_A` (MERT-fwd), `X_B` (CREPE-fwd), `X_C` (all), `X_D` (MERT+Essentia), `X_F`
  (MERT+DTW), `X_MERT_FULL`; regression twins `MERT_REG`, `X_C_REG`, `X_D_REG`, `X_F_REG` (stored
  raw — block weighting happens inside the folds). `STRICT_BLOCKS=('MERT','Essentia')`.
- Timing families: `DTW_REFFREE` (`timing_*`, condition-independent) vs `DTW_DEV` (`dtw_*`, 0/1 at
  MEC by construction); `DTW_MODEL_COLS` (→ X_C/X_F, family 1 only, `DTW_DEV_IN_MODELS=False`) vs
  `df_dtw_attr` (→ `df_interp`, both families, `DTW_DEV_IN_INTERP=True`).
- `df_interp` + `INTERP_FEAT_LABELS` (30 features: 10 CREPE + 10 Essentia-fwd + 10 DTW-fwd, §5.1 (now §9.1; its head → §4.1)).
- Results: `all_recs`, `DEFAULT_RECS`, `NB_RECS`, `DEFAULT_SET='C: Combined'`, `REG_RES`,
  `PRED_QUADS`, `INTERP_CV_R`, `INTERP_CV_F1`, `COE`, `BLOCK_F1`, `df_within_piece`,
  `df_wp_ceiling` / `WP_CEILING`, `df_delta_tests`, `df_reclevel`, `df_inner_skips`, `df_abl_bw`
  (tier 4E). Ablation rows now also carry `block_weighting` ('equal' / 'none' / 'n/a'),
  `n_alpha_edge_val`, `n_alpha_edge_aro`, `n_outer_folds`.
- B3 diagnostics: §4.4e (now §7.1) `MG_X_INTERP`, `MG_X_RANDOM`, `MG_BLOCKS`, `df_mg_gap`, `df_mg_curve`,
  `df_mg_nn`, `MG_VERDICT`, constants `MG_GAP_MARGIN=0.10`, `MG_CURVE_RISE=0.05`, `MG_NN_FOLD=3`,
  `MG_N_PERM=10000`, `MG_RANDOM_SEED=0` (outputs in `OUTPUT_DIR/mert_diagnostics/`) · §5.6 (now §10.2)
  `df_abl_commonality` · §5.6b (now §10.3) `RUN_PAIR_ABLATION_LOO=False`, `OV_I_TOL=0.02`, `OV_PROBE_MIN=0.30`,
  `OV_RHO_MIN=0.40`, `df_ov_pairs`, `df_ov_probe`, `df_ov_probe_fam`, `df_ov_corr`, `df_ov_corr_fam`
  (§5.6b (now §10.3) re-exports `df_ablation` / `ablation_master.csv` with the pair tier).
- C1: §2.1c `MERT_N_HIDDEN = 13`, `extract_mert_layers`, `emb_mert_layers` ({eid: 13 × 768}),
  `CACHE_MERT_LAYERS` · §4.4f (now §7.2) flags `RUN_MERT_LAYER_STAGE_B = True`, `RUN_MERT_LAYER_NESTED = True`,
  `MERT_LAYER_PROTOCOLS = ('piece', 'loo')`, constants `ML_N_PERM_COND = 1000`, `ML_SUGGEST_TOL = 0.05`,
  `ML_TIER = '6-mert-layers'`; names `ML_DIR`, `ML_DEPLOYED` (`'W05-07'`), `ML_CANDIDATES`,
  `ML_CAND_NAMES`, `ML_SET_NAME`, `ML_INTERP_LABEL` / `ML_INTERP_NAME`, `ML_NESTED_CLF`
  (`'SVM-Linear C=1'`), `ML_NESTED_SCORES`, `ML_SUGGESTED`, `df_mert_layer_a` / `_b` / `_deltas` /
  `_nested` / `_nested_freq`, `df_mert_layer_study`, `ml_plot_profile` (outputs in
  `OUTPUT_DIR/mert_layers/`) · §4.5c (now §5.3 / §6.9) `NULL_*`, `NULL_CACHE`, `EMP_NULL_F1`, `emp_null_dim`,
  `emp_null_p`, `df_null_draws`, `df_emp_null_readout`, `df_mert_layer_study_null` · §5.6 (now §10.2)
  `df_abl_subfamily['alone_emp_p']`. Temporaries are prefixed `_mlc_` / `_ml_` / `_null_`.

## Ablation study (§4.4b (now §6.3) → §5.6 (now §10.2), one engine)
`ablate()` in **§4.4b (now §6.3)** is the ONLY scorer; every tier appends to `ABL_ROWS`/`ABL_PRED`;
`df_ablation` → `ablation_master.csv` at the end of §5.6 (now §10.2) and again (with the pair tier) at the end
of §5.6b (now §10.3). `ablate(…, protocols=…)` already restricts the protocols (used by the pair tier).
- **Protocol:** default classifier (nested selection) → F1-macro; nested piece-grouped ridge on
  `X_reg` (the `MERT_REG`-based matrix whenever MERT is in the set) → r, R², RMSE, **with the
  deployed block weighting on multi-block MERT-full heads** (`ablate(reg_block_sizes=…,
  block_weighting=…)`, built by `_abl_reg_kw(names, weighting=None)`); scaler inside each fold;
  `abl_alpha_edge_report` prints the α grid-edge hits of the weighted heads (never widens the grid).
- **Both CV protocols, `piece` first.** Stats: bootstrap 95% CI · paired permutation p · BH-FDR
  within (tier, protocol); `abl_vs_ref(…, pcols=…)` can restrict the tested metrics (tier 4E =
  regression metrics only). `beats_chance` = F1 > `ABL_FLOOR_P95`.
- **Tiers:** `1-blocks` (15 subsets; unique vs marginal — the unique R² uses the deployed
  weighting) · `3-subfamily` (§5.6 (now §10.2)) · **`3-subfamily-pairs`** (§5.6b (now §10.3), B3: the 28 family pairs
  removed together + the reference row, `piece` only unless `RUN_PAIR_ABLATION_LOO`) · `4A-*` +
  `4B-gate` + `4D-balancing` (§4.4c (now §6.4)) ·
  **`4E-block-weighting`** (§4.4c (now §6.4): the 7 multi-block MERT-full subsets fitted unweighted and
  weighted; Δ = weighted − unweighted; the deployed member must equal its Tier-1 row) ·
  `4C-neutral-radius` + `4C-ground-truth` (§4.4d (now §6.5); `X_C_REG` rows weighted as deployed) ·
  **`6-mert-layers`** (§4.4f (now §7.2), C1, DIAGNOSTIC: the 25 MERT layer candidates built as deployed — §2.1b PCA
  replicated for classification, MERT-full for regression — + the interpretable reference row; Δ vs
  `MERT W05-07 (deployed)`; the deployed row must equal the Tier-1 `MERT` row; §4.4f (now §7.2) re-exports
  `ablation_master.csv`, §5.6 (now §10.2) / §5.6b (now §10.3) include the tier).
- **Guards:** §4.0 (now §5.1) engine equivalence (GaussianNB) · §4.0 (now §5.1) NestedSelect parallel == serial · **§4.0 (now §5.1)
  block-weighting guard** · §4.4b (now §6.3) lattice rebuilds `X_C/X_D/X_F` and `X_C_REG/X_D_REG/X_F_REG`
  (RAW matrices) · §4.4c (now §6.4) 4D nested-on-X_A equals the lattice `MERT` row · **§4.4c (now §6.4) 4E deployed ==
  lattice** · §4.4d (now §6.5) label re-derivation reproduces `Y` · §4.9 (now §8.1) closed-form operators equal
  `nested_ridge_predict` · §6.2b (now removed in C2) additive == lattice · §5.5 (now §10.1) interpretable r equals §5.4 (now §9.2) · §2.5 no
  retired cache name (pre-Phase-B, `_v2`, `_v3`) and the v4 names · **§2.6 (now removed in C2) audit recompute asserts**
  (temporary) · B3: §2.2 detector-on notes = deployed + splits, no deployed split, deployed
  calibration cell = the MEC vibrato-window share · §4.4e (now §7.1) refitted per-fold configs == NestedSelect,
  test F1 / ridge predictions == the §4.4b (now §6.3) / §4.4c (now §6.4) rows, validation curve == `cv_predict_multilabel` ·
  §5.6b (now §10.3) `MG_X_INTERP` == `_X_fam(_interp_cols)`, pair-tier reference == §5.6 (now §10.2) row,
  `NestedRidgeOperator` == `nested_ridge_predict` · **C1:** §2.1c mean of the `MERT_LAYERS` rows ==
  `emb_mert` (allclose rtol 1e-3 / atol 1e-4, max |diff| printed) · §4.4f (now §7.2) `W05-07` == `emb_mert` /
  `X_MERT_FULL`, its replicated PCA block == `X_A` (≤ 1e-5), its A1 == the §4.4e (now §7.1) MERT-full row, **HARD
  CHECK: the `W05-07` rows reproduce the §4.4b (now §6.3) `MERT` row exactly** (predictions, F1, r, R², RMSE, both
  protocols), inner LOPO never contains the test piece · §4.5c (now §5.3 / §6.9) draw 0 (d = 25) == the §4.4e (now §7.1) random
  control (matrix and F1) · §7.1 (now §11.1) one read-out row per headline cell · new cache names not in
  `OLD_CACHE_NAMES`.
- **NOT run:** CREPE step_ms, Essentia frame sizes, CREPE voicing threshold / the B2 descriptor constants as an ablation tier (cheap from
  `crepe_tracks_v2.npz`; §2.2 prints report-only sensitivities and the MEC calibration grids
  instead; the re-articulation detector is disabled since B3 and only reported), tempogram window /
  tempo prior. (Per-MERT-layer — ON HOLD on 2026-11-03 — is now run as the C1 DIAGNOSTIC §4.4f (now §7.2); the
  deployed `MERT_LAYERS` is unchanged.)

## §2.6 (now removed in C2) — TEMPORARY manual descriptor audit (2026-11-03; updated for B2 on 2026-11-05 and for B3 on 2026-11-10)
**C1 (2026-11-16): `RUN_MANUAL_AUDIT = False`** — both audit code cells only print "skipped"; the cells
are kept (set `True` to run the audit again). The §2.6 (now removed in C2) markdown says so.
**B3:** every recompute follows the deployed path (no re-articulation split; the merge range check; the
band-edge rule — windows carry `edge` / `reason`, purple in the zooms); V2 / V3 vibrato checks also need
the matched note to cover ≥ 80 % of the annotated span; **V4 is INFO** (repeated notes without a silence
are counted once — known limitation); **new V7** reads `CREPE_NOTECOUNT_FRIEDMAN` (PASS iff p ≥ 0.05).
The description below is the B2 one; where it mentions re-articulation splits / magenta markers, they
are absent while `REART_ENABLED = False`.
One markdown cell + two code cells between `# >>> AUDIT` / `# <<< AUDIT` markers (cells 41–43 since C1,
39–41 before; the B3 Kaggle run regenerated the ids: `9767b524`, `c78e74f3`, `2c5ee847`), right after §2.5.
`RUN_MANUAL_AUDIT` (`False` since C1), `AUDIT_EXCERPTS = Meditation_Thais_{MEC,EXP,EXG} + Mozart_VC4_{MEC,EXP,EXG}`;
every other name is `_aud_*`; nothing downstream reads it; writes only to `OUTPUT_DIR/manual_audit/`;
installs pyloudnorm (AUDIT only). It recomputes all 30 interpretable features with the pipeline's own
B2 helpers (+ `CREPE_ENV_DB`) and asserts equality with `df_crepe/df_essentia/df_dtw` (CREPE/DTW 1e-6,
Essentia 1e-5; declared §2.2 substitutions recognised); asserts that the vibrato windows reproduce
coverage / rate / extent, every window's `vibrato_window_fit` reproduces its R², the transition table
reproduces glide / transition counts and the portamento rate, and the per-frame tempo and Essentia
frame loops reproduce the helpers. Produces audio players (+ onset clicks, + beat clicks, + glide
tone), a 6-panel overview per excerpt (panel 2: notes, trend, shaded vibrato windows, magenta
re-articulation splits, reason codes incl. V = viterbi-limited jump; panel 3: + the 10 ms `env_db`
and the splits), zooms (the 3 longest analysed notes with every 500 ms window and its fitted
sinusoid + legacy FFT verdict; every glide / rejected transition with rise, t_min, excess), tables
(notes with segmentation source / windows / vibrato fraction / median R² / legacy FFT columns;
`*_vibrato_windows.csv`; transitions with t_min / excess / B2 verdict; key / LUFS), per-piece
direction tables (✓/✗), the 20-piece direction check, `audit_checklist.csv` (SQ8 row now counts
vibrato **windows**) and the **AUTHOR VALIDATION** table V1–V6 (`audit_author_validation.csv`,
PASS / FAIL printed, never raised) + the Méditation EXP F♯5→A5 / A4→B4 old vs new verdicts.
**Remove after the audit.**

## Phase B2 — descriptor repairs after the manual audit (2026-11-05; APPLIED, run in full — the re-articulation splits are disabled since B3)
**Audit findings (from the §2.6 (now removed in C2) outputs of the 2026-11-03 N = 116 run, checked by eye and ear):**
- **(a) Portamento false alarms = crepe's Viterbi ramp.** Méditation MEC D6→D5 (6.57 s, −1177 c) and
  A5→A4 (9.36 s, −1198 c) were "glides" (rise 62.5 ms, sign 1.0) — but crepe's Viterbi decoder
  (`crepe/core.py`, `to_viterbi_cents`, transition weight max(12 − |Δbin|, 0) over 20-cent bins)
  moves at most 11 bins = 220 c per 10 ms frame, so every octave jump becomes a ≈ 55 ms ramp. The
  real EXG glides are 175–312 ms (B5→C♯6 312.5, C♯6→D6 175, E5→A5 200).
- **(b) The 70 c boundary rule merges a note with its slide.** Méditation EXP "D6" 6.38–8.30 s
  (1920 ms, dev −12.5 c, FFT presence 0.08) is really C♯6 + slide + D6; EXG C♯6 5.61–6.64 s
  contains slides.
- **(c) The FFT presence test rejects real vibrato.** Méditation EXP D5 8.40–10.00 s (presence 0.533,
  5.58 Hz, ±4.15 c) and B4 12.73–16.12 s (presence 0.841, 5.63 Hz, ±4.40 c) were rejected only by the
  5 c half-extent rule; the parabolic peak stepped outside the band (rates 1.53 / 2.55 / 2.38 Hz on
  MEC negatives). The author confirmed no vibrato in SQ8 exrpt1 (MEC/EXP/EXG) and exrpt3 (EXP/EXG).
- **(d) Repeated notes are not separated.** Mozart VC4 has 19 notes in the score; the pipeline
  found 17 (MEC) / 16 (EXP) / 19 (EXG); repeated D6s merged (MEC 1.39–3.09 s, EXP 1.38–3.43 s,
  EXG 1.70–3.04 s) — a re-articulation without a pitch change has no trend boundary.
- **(e) Still open (documented, NOT changed in B2):** integrated LUFS Essentia − pyloudnorm =
  **+3.055 LU on all 6** audited excerpts (the dual-mono duplication; a constant offset, harmless for
  correlations but not an absolute LUFS) · `es.Key` gives **G minor for Mozart_VC4_EXP** (D major for
  the other five) · Méditation tempo log2-SD **MEC 0.256 > EXP 0.158 > EXG 0.135** and pulse clarity
  lowest at MEC (✗ vs the metronomic instruction — the tempogram reacts to articulation) · Mozart
  log-attack time **not sharper in the spiccato EXP** (✗; LAT on Essentia waveform onsets).

**New definitions (HELPERS:descriptors, §2.2):**
- **Segmentation** `segment_notes(cents, voiced, min_note_ms, step_ms=10, env_db=None,
  reart_dip_db=REART_DIP_DB)` (same return contract; `segment_notes_detail` adds `kinds` /
  `splits`): per voiced run, greedy **plateaus** of the 15-frame median trend (|trend − running
  median| ≤ `PLATEAU_TOL_CENTS=40`, range ≤ `PLATEAU_MAX_RANGE_CENTS=80` — added guard so a slow ramp
  cannot pass as a plateau), candidates ≥ 60 ms; adjacent plateaus < `PLATEAU_MERGE_CENTS=50` and
  ≤ `PLATEAU_MERGE_GAP_MS=50` apart merged; **re-articulation** split at a local minimum t of
  `env_db` with max(env[t−150 ms : t]) − env[t] ≥ `REART_DIP_DB=6` and max(env[t : t+60 ms]) − env[t]
  ≥ 6 dB (`REART_PRE_MS=150`, `REART_POST_MS=60`; candidates accepted deepest first, parts ≥ 60 ms);
  drop < min_note_ms. `env_db_on_grid(y, sr, n_frames)`: rectangular 25 ms window (`ENV_WIN_MS`)
  centred on the CREPE frame times i·10 ms, 16 kHz mono (the CREPE input), zeros outside, floor
  −120 dB → `rms10_v3.npz`. The Phase B rule stays as `_segment_notes_v2` (report only).
  Re-articulation splits are note onsets for the timing family.
- **Portamento (B9b)** `portamento_features(..., env_db=None)`: transitions between consecutive notes
  of one run; t_min = 0.8·|I| / `CREPE_VITERBI_MAX_CENTS_PER_FRAME` (220) · 10 ms ·
  `GLIDE_RISE_SCALE` (1.25); excess = rise − t_min; glide iff rise ∈ `GLIDE_DUR_MS=(80, 500)`, excess
  ≥ `GLIDE_MIN_EXCESS_MS=50`, |I| ∈ 80–2000 c, sign ≥ 0.80. Verdicts (order): across an unvoiced gap ·
  interval out of range · no 90 % crossing · too short · viterbi-limited jump · too slow · not
  monotone · glide. Same output keys + per-glide `excess` + `transitions` (with `t_min_ms`,
  `excess_ms`, `passes_except_excess`). §2.2 checks `inspect.getsource(crepe.core.to_viterbi_cents)`
  for `"12 - abs"` (prints, never raises).
- **Vibrato (B8b)** `vibrato_features(cents, voiced, min_note_ms=VIB_MIN_NOTE_MS, env_db=None)`:
  notes ≥ `VIB_MIN_NOTE_MS=500` (sensitivity 500/650/800), `VIB_WIN_MS=500` windows every
  `VIB_HOP_MS=250` (last window shifted to the note end), degree-2 detrend, LS a·sin + b·cos over
  `VIB_RATE_BAND=(4, 8)` Hz step 0.05, R², argmax + parabolic refinement clamped to the grid,
  half-extent √(a² + b²) (`vibrato_window_fit`); vibrato window iff R² ≥ `VIB_FIT_R2_MIN=0.50` and
  half-extent ≥ `VIB_MIN_EXTENT=3.0`; coverage = union of vibrato-window time / analysed-note time;
  rate / extent = duration-weighted medians over vibrato windows; same declared substitutions and ⚠️.
  Column names unchanged. `vibrato_note` = legacy FFT detector (report only; constants
  `VIB_BAND`, `VIB_PRESENCE_MIN`, `VIB_LEGACY_MIN_EXTENT=5`).
- **Calibration (report only, label-free):** MEC = negative control (no vibrato, no portamento by
  instruction). §2.2 prints the % of MEC windows passing R² ∈ {0.3, 0.4, 0.5, 0.6, 0.7} × half-extent
  ∈ {2, 3, 4, 5} c and the % of MEC transitions that would be glides at `GLIDE_MIN_EXCESS_MS` ∈
  {25, 50, 75, 100}; ⚠️ if the deployed setting exceeds 5 %. Audio-derived only — never a rating,
  never `Y`; nothing is chosen automatically (the constants stay as declared).
- **Caches:** `feat_crepe_v2.csv` → `feat_crepe_v3.csv`, `dtw_features_v2.csv` → `dtw_features_v3.csv`
  (the onsets changed), new `rms10_v3.npz`; both v2 names added to `OLD_CACHE_NAMES` (§0.3) and
  asserted in §2.5. `crepe_tracks_v2.npz` unchanged (CREPE itself is not re-run).
- Call sites: `crepe_descriptors(f0_hz, voiced, env_db=None)`, `note_onset_times(...,
  env_db=None)`, `timing_note_features(..., env_db=None)`, `extract_timing_reffree(y, sr, cents,
  voiced, env_db)` (§2.4, env_db required), `n_close_splits(notes, meds, kinds=None)`.

| Phase B | Phase B2 |
|---|---|
| boundary where the trend moves > 70 c over 5 frames | trend plateaus (±40 c, range ≤ 80 c) + merge < 50 c / ≤ 50 ms + re-articulation dips ≥ 6 dB (`_segment_notes_v2` kept, report only) |
| glide 40–400 ms rise, sign ≥ 0.8 | 80–500 ms rise, excess over the Viterbi ramp ≥ 50 ms, sign ≥ 0.8 |
| vibrato notes ≥ 800 ms, FFT peak in [3,10] Hz, presence ≥ 0.30, half-extent ≥ 5 c (Hilbert) | notes ≥ 500 ms, 500 ms windows / 250 ms hop, LS sinusoid 4–8 Hz, R² ≥ 0.50, half-extent ≥ 3 c (√(a²+b²)) |
| coverage = vibrato-note duration / notes ≥ 800 ms | union of vibrato-window time / analysed-note time |
| rate / extent = duration-weighted medians over vibrato notes | … over vibrato windows (rate clamped to 4–8 Hz) |
| `CREPE_DIAG` `n_vib` | `n_vib_notes`, `n_windows`, `n_vib_windows`, `n_notes_60_v2`, `n_reart_splits`, `glide_excess`, `transitions`, `vib_windows` |
| `feat_crepe_v2.csv`, `dtw_features_v2.csv` | `feat_crepe_v3.csv`, `dtw_features_v3.csv`, + `rms10_v3.npz` |

## Phase B3 — re-articulation detector disabled + MERT / family-overlap diagnostics (2026-11-10; APPLIED; RUN 2026-11-11)
**Evidence — the full B2 run (N = 116, block weighting; quoted by the author, outputs not committed):**
- **(a) Re-articulation splits follow vibrato, not bow changes.** The B2 detector split a plateau where
  the 10 ms RMS envelope dips ≥ 6 dB — but on the violin vibrato also modulates the amplitude (the
  frequency sweeps through the body resonances); the 25 ms envelope resolves it and the 6 dB rule took
  it for a re-attack. Splits per condition **MEC 24 / EXP 51 / EXG 68** (same scores);
  Meditation_Thais 0 / 3 / 11 (the EXG F♯5 cut into 7 pieces of 140–370 ms = 1–2 vibrato periods);
  Shostakovich_Sym5_Mvt3_EXG 11. Note totals per condition: Phase B rule 575 / 533 / 518 → B2 plateaus
  only 616 / 596 / 592 → B2 with splits 640 / 647 / 660. `REART_DIP_DB` sensitivity Mozart_VC4
  MEC / EXP / EXG = 26/23/28 (3 dB), 22/21/23 (6 dB), 19/17/22 (9 dB) vs 19 in the score — no threshold
  fixes all three. Consequences: §5.1b (now §4.2 / §4.3b) note-based timing turned significant (note rate z = +2.06,
  p = 0.040; IOI mean −2.53, p = 0.011; IOI CV +2.85, p = 0.004 — all n.s. in Phase B); §5.6 (now §10.2)
  "Timing (ref-free)" alone F1 (piece) 0.417 → 0.542 and its removal loss −0.012 → +0.110 (q = 0.037);
  long EXG vibrato notes fragmented below the 500 ms analysis minimum; the portamento share counted
  repeated-note pairs in its denominator.
- **(b) The merge broke the plateau range invariant.** Adjacent plateaus (< 50 c, ≤ 50 ms) were merged
  without re-checking the 80 c range → Meditation_Thais_EXG C♯6 (≈ 5.76–6.62 s) had a trend range of
  128 c (V3 FAIL); merged notes also absorbed the transition frames.
- **(c) Vibrato rate pinned at the band edge.** ≥ 1 excerpt had a weighted-median rate of exactly
  4.00 Hz (the lower grid edge): the best fit lay outside the band — a residual slow trend.
- **(d) MERT generalisation.** F1 LOOCV → piece: MERT 0.623 → 0.310 (−50 %, below the P95 0.323) vs
  CREPE 0.500 → 0.333, Essentia 0.480 → 0.356, DTW 0.570 → 0.542; adding MERT to an interpretable block
  lowered the piece F1 in 6/7 combinations and raised the LOOCV F1 in 7/7; unique F1 of MERT (piece)
  −0.122 (q = 0.034); NestedSelect chose C = 0.001 in 12/20 piece folds and C = 10 in 43/60 LOOCV folds.
  MERT-PCA25 ridge R² (piece) valence −0.18, arousal −0.25; MERT-full ridge arousal 0.45, valence 0.01;
  the valence α of the MERT combos hit the upper grid edge. The MERT PCA is fit once on all 60
  (label-free). No train score anywhere → capacity overfitting vs a piece-identity shortcut untested.
- **(e) Family overlap (§5.6 (now §10.2), piece).** Full interpretable layer (30 features, 8 families) F1 0.449,
  R²a 0.627; Σ family-alone R²a ≈ 1.55, Σ removal losses ≈ 0.23 → most of the arousal R² is shared.
  The §5.6 (now §10.2) F1 figure drew "alone" bars from 0 (chance mean 0.272, P95 0.323). "Small removal loss +
  high alone score" can be redundancy, noise (only 2 of 24 removal losses q < 0.05), dilution (Timing
  alone 0.542 > all 30 at 0.449) or a different NestedSelect choice per subset.

**Decision (author): DISABLE the re-articulation detector for the features; keep it report-only.**
`REART_ENABLED = False` (HELPERS:descriptors). Defensible because, without it, a repeated same-pitch
note with no silence between counts as one note — a bias set by the **score**, ≈ the same in the three
renditions of a piece, so it cancels in within-piece comparisons; the vibrato artefact grew with
expressiveness and did not cancel. Only 4 of the 7 timing features use note onsets (`timing_note_rate`,
`timing_ioi_mean_s`, `timing_ioi_sd_s`, `timing_ioi_cv`); tempo variability / local tempo / pulse clarity
come from the tempogram and are unaffected. Vibrato and portamento never needed the detector.

**Changes (all in the notebook; HELPERS stay numpy / scipy / bisect only):**
- **A1** `REART_ENABLED = False`; `segment_notes_detail(…, reart_enabled=None)` and
  `segment_notes(…, reart_enabled=None)` gained an optional switch (None → `REART_ENABLED`; `True`
  forces the detector on for the report; default keeps every caller unchanged). Every feature path
  (`segment_notes`, `note_onset_times`, `timing_note_features`, `vibrato_features`,
  `portamento_features`, `crepe_descriptors`, `extract_timing_reffree`) still accepts `env_db` (still
  length-checked) and ignores it. `_reart_splits`, `env_db_on_grid`, `REART_*`, `rms10_v3.npz` kept.
- **A2** `_merge_plateaus(…, max_range=PLATEAU_MAX_RANGE_CENTS)`: merge only if the merged note's trend
  range (gap frames included) ≤ 80 c. (May raise the report-only close-split count a little.)
- **A3** `vibrato_window_fit` returns `edge` (R² argmax before refinement = first / last grid point);
  new `vibrato_window_reason(f)` → 'band edge' / 'low R²' / 'small extent' / 'vibrato' (rule order);
  windows carry `edge` + `reason`; a band-edge window is never vibrato; `vibrato_features` returns
  `n_edge_windows` / `n_edge_only` (edge windows that would otherwise have passed); `CREPE_DIAG`
  `vib_windows` = (r2, extent, edge); §2.2 prints the edge rejections (all excerpts, the 20 MEC) and the
  calibration grid applies the rule in every cell (assert: deployed cell = MEC vibrato-window share).
- **A4** §2.2 banner ("re-articulation detector DISABLED (REART_ENABLED = False) — see Phase B3"),
  report-only evidence table (detector off vs on at 6 dB: totals + splits per condition, per excerpt,
  3 / 6 / 9 dB sensitivity with Mozart_VC4), condition-independence check of the deployed note counts
  (per-condition totals, per-piece range, `scipy.stats.friedmanchisquare` on the 20 × 3 table, also on
  the detector-on counts; ✅ p ≥ 0.05 / ⚠️, printed) → `CREPE_NOTECOUNT_FRIEDMAN`.
- **A5** caches `feat_crepe_v3.csv` → **`feat_crepe_v4.csv`**, `dtw_features_v3.csv` →
  **`dtw_features_v4.csv`**; both v3 names in `OLD_CACHE_NAMES`; §2.5 asserts the v2 and v3 names are
  retired and the v4 names are in use. `crepe_tracks_v2.npz`, `rms10_v3.npz`, MERT / Essentia unchanged.
- **A6** §2.6 (now removed in C2): equality asserts on the disabled path (+ assert: no deployed split); no split markers;
  band-edge windows purple with their reason; notes table `n_edge_windows`; **V4 → INFO** (known
  limitation, expected ≈ 17 / 16 / 20); every V2 / V3 "has vibrato" check also needs the matched note to
  cover ≥ 80 % of the annotated span; **new V7** = the §2.2 Friedman result (PASS / FAIL).
- **§4.4e (now §7.1) (new, after §4.4d (now §6.5))** MERT generalisation diagnostics — B1 train vs test F1 (per-fold
  NestedSelect configs read from the §4.0 (now §5.1) memo — the §4.4b (now §6.3) rows store only counts — refitted with
  §4.0 (now §5.1)'s code; test asserted = NestedSelect and = the §4.4b (now §6.3) / §4.4c (now §6.4) rows) and train vs out-of-sample R²
  (nested ridge) for MERT-PCA25, MERT-full, CREPE, Essentia, DTW, the 30-feature layer and a 25-column
  N(0, 1) control (seed 0), `piece` then `loo`; B2 validation curve (`piece`, SVM-Linear balanced at each
  fixed C of `SELECT_C_GRID`) for MERT-PCA25 / interpretable / random; B3 nearest-neighbour identity
  (same-piece vs chance 2/59, same-condition vs 19/59, 10 000 label shuffles, silhouettes) in MERT-full
  (cosine), MERT-PCA25, CREPE, Essentia, DTW, interpretable (z-scored). **Pre-declared rules:**
  CAPACITY OVERFITTING iff MERT-PCA25's piece gap ≥ interpretable gap + 0.10 and ≥ random gap + 0.10
  AND every C with a risen train F1 (≥ +0.05 over the smallest C; ≥ 1 such C) keeps the test F1 ≤ the
  strict P95; PIECE-IDENTITY SHORTCUT iff in a MERT space the same-piece share ≥ 3 × chance, p < 0.05,
  above every interpretable space, while the same-condition share has p ≥ 0.05. Both can fire.
  (Named §4.4e (now §7.1) because §4.4d (now §6.5) = the target-side ablation already exists.)
- **§5.6 (now §10.2) (edited)** F1 panel: chance mean + strict P95 lines; `*` on removal losses with q < 0.05;
  commonality summary per protocol (F1 above the chance mean, R²a, R²v: Σ alone, Σ loss, full,
  shared ≈ full − Σ loss; not exactly additive) → `ablation_subfamily_commonality.csv`; new column
  `q_loss_r2_val`; the pre-declared timing reading printed (below).
- **§5.6b (now §10.3) (new, after §5.6 (now §10.2))** C1 tier `3-subfamily-pairs` (28 pairs + reference, `piece`;
  `RUN_PAIR_ABLATION_LOO = False`; `ablate(protocols=…)` existed, no engine change), I(A,B) =
  loss(A∪B) − loss(A) − loss(B) for F1 / R²a / R²v, read only if the pair loss q < 0.05 (I > 0.02
  substitutes, |I| ≤ 0.02 each its own, I < −0.02 only together), 3 heatmaps, runtime printed,
  `ablation_master.csv` re-exported; C2 probes (nested ridge via `NestedRidgeOperator`, `piece`):
  each feature of A from all non-A features and from each family B (8 × 8), RAW and WITHIN-PIECE
  (feature minus its piece mean — label-free); C3 Spearman 30 × 30 RAW / WITHIN-PIECE ordered by family
  + 8 × 8 mean |ρ|; **C4 pre-declared triangulation**: A and B overlap only if ≥ 2 of (1) I > 0 with
  pair q < 0.05 (any target), (2) within-piece probe R²(A←B) or R²(B←A) ≥ 0.30, (3) within-piece family
  mean |ρ| ≥ 0.40; prints the passing pairs and the one-test pairs; C5 `overlap_pairs.csv`,
  `overlap_probe(_family).csv`, `overlap_corr(_family).csv`, 3 figures.

**A7 — acceptance criteria for the next run (documented here, NOT asserted):** Meditation_Thais
19 / 19 / 20 notes; per-condition totals ≈ 616 / 596 / 592 (small changes from A2 allowed); Friedman
n.s. (V7 PASS); §5.1b (now §4.2 / §4.3b) note rate / IOI dose-response back to n.s. **PRE-DECLARED timing reading:** in
§5.6 (now §10.2) compare "Timing (ref-free)" alone F1 (piece) with 0.417 (Phase B) and 0.542 (B2) — near 0.54 → the
timing signal is real; back near 0.42 → the B2 value was the vibrato artefact. Record which; quote only
the B3 value in the thesis. *As printed by the 2026-11-11 run (committed outputs, for the author to
confirm): Timing (ref-free) alone F1 (piece) = 0.398 → closer to 0.417 — the B2 value was the vibrato
artefact; deployed note-count Friedman χ²(2) = 0.535, p = 0.765 (V7 PASS). The other A7 items are not
transcribed yet.*

**Known limitations (B3, documented — NOT fixed):** repeated notes without a silence count as one
(Mozart_VC4 ≈ 17 / 16 / 20 vs 19) · slow slides can create short "ramp plateaus" (a glide counted twice,
e.g. B5→C6→C♯6 in Meditation_Thais_EXG) · the 80 ms glide minimum rejects the EXP C♯6→D6 (rise 75 ms) ·
residual MEC detections (V6: 5 MEC — the glides look like position shifts) · tempo log2-SD / pulse
clarity / log-attack time / `es.Key` (Mozart_VC4_EXP read as G minor) still open.

## Phase C1 — empirical chance null + DIAGNOSTIC MERT layer study (2026-11-16; APPLIED; RUN 2026-11-16 — see "Current results")
**Why the empirical null — the B3 random-control finding (run of 2026-11-11, `piece`).** §4.4e (now §7.1)'s capacity
control (25 i.i.d. N(0, 1) columns, seed 0) through NestedSelect scored **piece F1 0.354**, above the Dummy
strict P95 **0.323** (uniform mean 0.272); on the validation curve the noise reached **0.468 at C = 0.001**
(NestedSelect chose C = 0.001 in 14/20 piece folds for the noise). Likely mechanism: at a tiny C the balanced
linear SVC is ≈ a class-mean-difference rule whose threshold follows the training-label composition of each
LOPO fold, and the labels cluster within pieces. The Dummy floor ignores the selection procedure, so it is
not an adequate null for the deployed classifier (regression is unaffected: noise R² ≈ 0). Context from the
same run: same-piece nearest neighbour MERT-full 100 %, MERT-PCA25 90 %, interpretable 57 % (chance 3.4 %) —
the **piece-identity shortcut rule fired** (capacity overfitting did not); F1 LOOCV 0.623 → piece 0.310.

**How the null is read (§4.5c (now §5.3 / §6.9), new cell after the §4.5 (now §6.6) figures).** `NULL_N_DRAWS = 100` draws per
d ∈ `NULL_DIMS = (25, 10)` (25 = MERT-PCA25 / interpretable layer; 10 = CREPE / Essentia; DTW 7 → 10), each an
i.i.d. standard-normal 60 × d matrix (seed `NULL_SEED` + draw, `NULL_SEED = 0`) through the deployed
`cv_predict_nested_select` path (same `Y`, §4.0 (now §5.1) folds, 13 configs, inner GroupKFold(5), tie rule), pooled
test F1-macro as the headline, NestedSelect config counts recorded; `NULL_PROTOCOLS = ('piece',)`,
`RUN_NULL_LOO = False`. Cache `null_f1_draws.csv` (reused iff the settings hash — draws, dims, seed,
protocols, `Y`, configs, inner splits, tie rule, groups — matches; the cell prints which). Draw 0 at d = 25
is asserted to be the §4.4e (now §7.1) random control (matrix and F1). `EMP_NULL_F1[(protocol, d)] = {mean, p95, p99,
draws}` is printed next to the Dummy floors with the share of draws above the Dummy P95. Every reported
classification F1 — §4.1 (now §6.1) A–D, every §4.4b (now §6.3) lattice row (`MERT + DTW` = `X_F`; §4.1 (now §6.1) has no X_F row), the
§4.4e (now §7.1) blocks, the §4.4f (now §7.2) candidates — gets the empirical one-sided p = (1 + #draws ≥ F1) / (1 + 100)
against the nearest d (≤ 17 → 10, ≥ 18 → 25, > 25 extrapolated) and "beats empirical null" (p < 0.05) →
`emp_null_readout.csv`. §5.6 (now §10.2) draws the empirical P95 (d = 25) on the F1 panel and prints `alone_emp_p`;
§7.1 (now §11.1) adds "Empirical-null p" / "Beats empirical null" per protocol ('n/a' where not drawn).

**MERT layer study (DIAGNOSTIC — `MERT_LAYERS` stays `[5, 6, 7]` until the author decides by hand).**
- **§2.1c** (new, after §2.1b — §2.1b was already the MERT PCA forwarding cell) `extract_mert_layers`:
  one forward pass exactly as `extract_mert` → (13, 768) float32 time-means of every hidden state
  (0 = transformer input, 1–12 layers), asserted 13; cache `emb_mert_layers_v3.npz`; the mean of the
  `MERT_LAYERS` rows must equal `emb_mert` (max |diff| printed, `allclose(rtol=1e-3, atol=1e-4)`).
  Label-free; `extract_mert` / `emb_mert` / `emb_mert_v2.npz` unchanged.
- **§4.4f (now §7.2)** (new, after §4.4e (now §7.1)) candidates: `L00`…`L12`, `W00-02`…`W10-12` (`W05-07` = deployed — asserted ==
  `emb_mert`, then the deployed vectors are used), `ALL`; references: the 30-feature interpretable layer
  (`MG_X_INTERP`) and `W05-07`. **Stage A (no ratings):** A1 same-piece NN share (cosine; §4.4e (now §7.1) shuffle
  scheme, 10 000, seed `SEED`; silhouette; descriptive; `W05-07` asserted == §4.4e (now §7.1)) · A2 condition decoding
  across pieces (piece-centred — label-free, the piece's own renditions —, §4.0 (now §5.1) LOPO folds,
  parameter-free nearest-centroid cosine classifier, accuracy vs 1/3, 1000 within-piece condition
  permutations) · A3 share of pieces with d(MEC, EXG) the largest within-piece cosine distance (vs 1/3, same
  permutations) · BH across the 25 for A2 and A3. **Pre-declared suggestion rule** (printed first): among
  A2 q < 0.05 the lowest A1 share; within 5 pp prefer a 3-layer window (ties: A1, piece silhouette, order);
  none → "keep W05-07"; printed as "SUGGESTED — not applied". **Stage B (descriptive):** every candidate
  through `ablate()` in tier `6-mert-layers` (classification = §2.1b PCA 95 % replicated, transductive on
  the 60; regression = MERT-full via `MERT_REG_REPR` + nested ridge), both protocols, Δ vs W05-07
  (`abl_vs_ref`, paired, BH); **HARD CHECK** W05-07 == the §4.4b (now §6.3) `MERT` row; per candidate piece / LOOCV
  F1, the gap, R²v / R²a both protocols, NestedSelect configs; §4.5c (now §5.3 / §6.9) adds the empirical-null p and BH q
  across the 25. Pre-declared "beats W05-07 beyond the empirical null": F1 — ΔF1 > 0, paired q < 0.05 AND
  empirical-null q < 0.05; R²a — ΔR² > 0, paired q < 0.05. **Nested layer selection** (`piece`): outer
  §4.0 (now §5.1) folds, inner LOPO over the training pieces (the `get_cv('piece')` splitter on the training rows),
  FIXED estimators — `SVM-Linear C=1` (balanced) on a per-candidate PCA 95 % fitted on the inner training
  rows; `StandardScaler → RidgeCV(RIDGE_ALPHAS)` on the 768 dims per target — choice by pooled inner
  macro-F1 / SSE (ties → order), outer score + the same fixed estimators on W05-07, the choice per fold and
  its frequency table; a stability diagnostic, not comparable with the headline. Cache
  `mert_layer_nested.csv` + `mert_layer_nested_pred.csv` (settings hash).
- Outputs `mert_layers/mert_layer_study.csv`, `fig_mert_layer_profile.png` (x = hidden state, windows at
  their centre, ALL at 13; references interpretable / W05-07 / chance / Dummy P95 / empirical P95), a
  one-line summary per question.

**Flags / caches.** `RUN_MANUAL_AUDIT = False` (§2.6 (now removed in C2)) · §4.4f (now §7.2) `RUN_MERT_LAYER_STAGE_B = True`,
`RUN_MERT_LAYER_NESTED = True`, `MERT_LAYER_PROTOCOLS = ('piece', 'loo')` · §4.5c (now §5.3 / §6.9) `NULL_N_DRAWS`,
`NULL_DIMS`, `NULL_SEED`, `NULL_PROTOCOLS`, `RUN_NULL_LOO` · new caches `emb_mert_layers_v3.npz`,
`null_f1_draws.csv`, `mert_layers/mert_layer_nested(.csv, _pred.csv)`; nothing retired
(`emb_mert_v2.npz` stays). **Changed cells:** intro (md), §2.6 (now removed in C2) md + §2.6a (now removed in C2) code (flag), §4.4c (now §6.4) md + code (the
"per-layer MERT pending" text), §4.5 (now §6.6) md, §5.6 (now §10.2) md + code, §7.1 (now §11.1) code, output reference (md); **new:** §2.1c
(md + code), §4.4f (now §7.2) (md + code), §4.5c (now §5.3 / §6.9) (md + code) → 114 cells.

**Expected runtimes (T4, 4 CPUs; the B3 run took ≈ 127 min).** §2.1c ≈ 0.5–1 min (B3 extracted `emb_mert`
in 26 s) · §4.4f (now §7.2) Stage A ≈ 1–2 min, Stage B ≈ 35–45 min (both protocols; ≈ 10–15 min `piece` only),
nested ≈ 3–10 min · §4.5c (now §5.3 / §6.9) ≈ 50–70 min (2 × 100 NestedSelect `piece` runs at ≈ 15–20 s; `RUN_NULL_LOO`
≈ +3 h) · total ≈ +1.5–2 h → ≈ 3.5–4 h. Quick run: `MERT_LAYER_PROTOCOLS = ('piece',)`,
`RUN_MERT_LAYER_NESTED = False`, `NULL_DIMS = (25,)` (the caches make reruns cheap).

## C1 checklist — run 2026-11-16 (outcome; numbers in "Current results")
- [x] **§4.5c (now §5.3 / §6.9):** computed. Empirical P95: d = 10 0.370, d = 25 0.459 (Dummy strict P95 0.323). 36 / 52 F1 beat the Dummy P95; 4 / 52 (3 distinct, all containing the reference-free timing block) beat the empirical null; the random control has p 0.58.
- [x] **§4.4f (now §7.2) Stage A:** the rule was printed; lowest A1 L10 93.3 %; best A2 L05 0.900 (25 / 25 with q < 0.05); best A3 L04 0.85; **SUGGESTED — not applied: W09-11**.
- [x] **§4.4f (now §7.2) Stage B:** none beats W05-07 on piece F1 beyond the empirical null (0 / 25 beat the null; W05-07 p 0.832). R²a: L04, W03-05 and W04-06 beat W05-07 (paired q < 0.05).
- [x] **§4.4f (now §7.2) nested:** classes L00 5 / 20 (9 distinct), R²a L11 9 / 20, R²v L11 16 / 20 → unstable.
- [x] **Asserts:** the run reached the §4.4f (now §7.2) summary and §4.5c (now §5.3 / §6.9), so the §2.1c consistency assert and the Stage B HARD CHECK did not fail.
- [ ] **Not transcribed** (the outputs were not reported in chat): §2.1c max |diff|, §5.6 (now §10.2) `alone_emp_p`, the §7.1 (now §11.1) column, the other guards, the total runtime — read them in the committed notebook.
- [ ] **The author's decision on `MERT_LAYERS`** (recommended: keep [5, 6, 7]) — record it here.

### B3 checklist (2026-11-10) — run on 2026-11-11; transcribed 2026-11-17 (numbers in "Current results")
Not transcribed: the DTW forwarding count and the full calibration grids (deployed cells 0.7 % / 2.0 %).
A7: Méditation 19 / 20 / 21; Friedman p 0.765 (V7 PASS); note rate / IOI still significant (raw p; a
duration effect); timing reading → B2 artefact.
Interactive run up to §2.6 (now removed in C2) first:
- [x] §2.2 banner "re-articulation detector DISABLED"; octave fixes / Viterbi-source line unchanged.
- [x] §2.2 evidence table: detector OFF totals ≈ 616 / 596 / 592 (A2 may move them a little) vs ON
      ≈ 640 / 647 / 660 with splits ≈ 24 / 51 / 68; 3 / 6 / 9 dB Mozart_VC4 rows; asserts ✅.
- [x] §2.2 condition-independence: per-condition totals, per-piece range, **Friedman n.s. (✅)** — and
      the detector-on Friedman for comparison.
- [x] §2.2 vibrato: band-edge rejections (all / MEC) and how many were rejected ONLY by the edge rule;
      rate range strictly inside 4–8 Hz (no 4.00); calibration grid (deployed cell ≤ 5 %?).
- [x] §2.2 portamento verdict counts; close splits (⚠️ > 5 %? A2 can add some).
- [x] §2.4 `dtw_features_v4.csv` written; DTW forwarding count 3–10; §2.5 cache asserts ✅.
- [x] §2.6 (now removed in C2): recompute asserts ✅; Meditation_Thais **19 / 19 / 20** notes; V3 C♯6 range ≤ 80 c; V2 / V3
      vibrato checks with ≥ 80 % span coverage; V4 INFO ≈ 17 / 16 / 20; V5; V6 (≈ 5 MEC fail — known);
      **V7 PASS**; purple band-edge windows in the zooms.
Full run:
- [x] §5.1b (now §4.2 / §4.3b): note rate / IOI mean / IOI CV dose-response back to n.s.? vibrato extent / coverage z.
- [x] §5.6 (now §10.2): **pre-declared timing reading** (Timing alone F1 vs 0.417 / 0.542 — record which);
      commonality summary; chance lines and `*` markers in the figure.
- [x] §4.4e (now §7.1): B1 asserts ✅; MERT-PCA25 / interpretable / random gaps (piece); validation curve; NN
      identity shares + p; **which rule fires** (capacity / shortcut / both / neither).
- [x] §5.6b (now §10.3): tier runtime; I heatmaps; probes raw vs within-piece; correlations; **triangulation pairs**;
      `ablation_master.csv` re-exported with `3-subfamily-pairs`.
- [x] Everything else as before (all guards ✅, NestedSelect vs P95, §4.9 (now §8.1) B-INC, §5.5 (now §10.1) CoE, §6.1b (now §7.3b), total
      runtime ≈ B2 run + ≈ 15–20 min for §4.4e (now §7.1) + §5.6b (now §10.3)).

### B2 checklist (2026-11-05) — run in full by the author; kept for reference
First, interactive run up to §2.6 (now removed in C2) (the §1 items already held in the 2026-11-03 run):
- [ ] §1.1: "2 excluded — questionnaire_5_P012, questionnaire_5_P020"; 118 → 116 participants,
      2360 → 2320 rows; per form F1 20→20 · F2 21→21 · F3 26→26 · F4 19→19 · F5 32→30; asserts pass.
- [ ] §1.1b maps ≈ V 2.94 + 1.57z, A 3.39 + 1.53z · §1.2b anchors V 0.342 → 0.338, A 0.276 → 0.253 ·
      8/60 class changes raw → norm · §1.2 classes HA_LV 20, LA_LV 17, HA_HV 11, Neutral 7, LA_HV 5
      (`Schindlers_List_EXP` HA_LV → LA_LV) · §1.4 (now §4.3a) prints the exclusions.
- [ ] §2.2 (B2): the Viterbi-source line ("12 - abs" found → assumption holds; ⚠️ otherwise);
      `rms10_v3.npz` computed once (60 excerpts, length assert ✅); octave fixes unchanged (3 frames).
- [ ] §2.2 segmentation: old (Phase B rule) vs new note counts — total, per condition, per excerpt;
      re-articulation splits; `REART_DIP_DB` 3 / 6 / 9 sensitivity of the total and of Mozart VC4
      MEC/EXP/EXG (score: 19 each); close splits (⚠️ > 5 % of notes).
- [ ] §2.2 vibrato: analysed notes at 500 / 650 / 800 ms; windows / vibrato windows; substitutions
      (no note ≥ 500 ms: ___ — ⚠️ above 10; no vibrato window: ___ — expected ≈ every MEC + SQ8);
      rate inside 4–8 Hz (assert), half-extent / coverage ranges.
- [ ] §2.2 portamento: glides / transitions; verdict counts (expect many "viterbi-limited jump");
      glide rise / excess / extent ranges.
- [ ] §2.2 **calibration grids** (MEC negative control): false-positive vibrato windows at the
      deployed R² 0.50 / 3 c (*) and false-positive glides at excess 50 ms; **⚠️ if either > 5 %** —
      then discuss the constants with the author before the full run (nothing is auto-tuned).
- [ ] §2.4: timing features recomputed with the B2 onsets (`dtw_features_v3.csv` — retired in B3); DTW forwarding
      count (3–10).
- [ ] §2.6 (now removed in C2): every recompute / detail-consistency assert passes (windows → coverage / rate / extent,
      window fit → R², transitions → portamento); the declared substitutions listed; LUFS
      diff_minus_3dB ≈ 0; listen to the players; look at the overview (shaded vibrato windows,
      magenta splits, env_db) and zoom figures (fitted sinusoids); **AUTHOR VALIDATION V1–V6 all
      PASS?** (V1 Méditation MEC no glide · V2 Méditation EXP C♯6 / D6 separate, C♯6→D6 glide, D5 / B4
      ≥ 50 % vibrato windows, F♯5 / B5 vibrato · V3 Méditation EXG C♯6 trend range ≤ 80 c, F♯5 / B5 / D6
      / D5 vibrato · V4 Mozart VC4 19 notes × 3 · V5 SQ8 zero vibrato windows · V6 every MEC coverage
      ≤ 0.05 and portamento ≤ 0.05) and the F♯5→A5 / A4→B4 old vs new verdicts; fill
      `audit_checklist.csv`; read the 20-piece direction exceptions. Then set `RUN_MANUAL_AUDIT` as
      wanted and launch the full run.
Full run:
- [ ] §4.0 (now §5.1): class prevalences 11/20/5/17/7 and the analytic uniform F1; block-weighting guard ✅ and
      the printed weights (expected MERT-full w ≈ 0.51, CREPE / Essentia ≈ 4.46, DTW ≈ 5.33 for
      795 = 768 + 10 + 10 + 7 columns); parallel == serial ✅; runtime.
- [ ] §4.4b (now §6.3): `[w]` on the 7 MERT-full multi-block rows; do MERT + X rows now differ from MERT alone
      (R²_aro was 0.45–0.46 for all of them at N = 118)? α grid-edge report (warn > 0).
- [ ] §4.4c (now §6.4) tier 4E: Δ weighted − unweighted (r, R², q) per set, both protocols; deployed == lattice
      assert ✅; α edge hits.
- [ ] §4.4d (now §6.5): X_C rows weighted; NEUTRAL_RADIUS rule (did r = 1.0 again beat 0.75 on both columns?).
- [ ] Everything else at N = 116: NestedSelect vs strict P95 (piece), §4.9 (now §8.1) B-INC q, §5.5 (now §10.1) CoE,
      §5.6 (now §10.2), §6.1b (now §7.3b), the §5.1 (now §9.1; its head → §4.1) BH count (180 tests). All guards ✅. Total runtime (≈ 97 min at N = 118
      + 4E ≈ a few minutes + §2.6 (now removed in C2)).
- [ ] **§5.1b (now §4.2 / §4.3b) dose-response re-check for the vibrato family** — the thesis abstract claims that
      vibrato extent / coverage escalate MEC → EXP → EXG (Phase B: z +5.14 / +4.90). With the B2
      vibrato (MEC ≈ no vibrato windows by construction of the negative control) re-read the Page z
      of extent, coverage and rate before keeping the claim; also §5.6 (now §10.2) Vibrato / Portamento rows
      (Phase B: removing Portamento *raised* F1 — consistent with the false alarms) and §6.1b (now §7.3b) for the
      vibrato / portamento descriptors (re-check `CREPE_UNIQUE2`).

## Phase C2 — applied 2026-11-20, NOT RUN

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
  `NULL_N_DRAWS_PRIMARY = 1000` (§0.3); same seeding (map draws 0–99 reused); cache `null_f1_draws_primary.csv`
  (per-row settings hash, **resumable**, checkpoint every 50 draws, round-robin over widths,
  `NULL_PRIMARY_BUDGET_MIN = 480`); `emp_null_ref` / `emp_null_mark` (labels "exact-width", "map", "null borrowed from
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

### C2 run checklist (what to look at; which C1 numbers change)

Runtime: the primary null is the new cost — 7 widths × 1000 draws, minus the 200 map draws reused ≈ 6 800
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
      will change with exact widths and 1000 draws).
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

## Current results — B3 run (2026-11-11) + C1 run (2026-11-16), N = 116

> ⚠️ **Phase C2 (applied 2026-11-20, NOT RUN) will replace** these numbers: everything that used the 30-feature interpretable layer in a model (now 27: Interpretable F1 ≈ 0.445, R²v ≈ 0.334, R²a ≈ 0.636 in the C1 run) — §6.2 (deployed rows), §7.1 / §7.2 interpretable references, §9.2 betas, §9.3 SHAP, §10.1 CoE, §10.2 families (7 instead of 8), §10.3 pairs (21 instead of 28; test 1 with I > 0.02); every empirical-null p / ✓ (exact-width, 1000-draw primary null, BH in the family — §6.9); the §4.2 stars (BH q instead of raw p); the new §6.3 tempogram rows, §6.8 (deployed nested ridge, chance references), §7.4 (signed, BH, 3 panels), §7.5 (lattice reads, CREPE-unique rule). Unchanged by construction: §4.2 / §9.1 values (descriptive 30), §8.1, the lattice, MERT rows, the Dummy floors, the 100-draw map. Section numbers below are pre-C2 with "(now §x.y)".

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
- **Alone, F1:** Timing 0.398 · Timbre 0.393 · Dynamics 0.327 · Pitch 0.323 · Vibrato 0.316 · DTW-dev 0.314 · Tonal 0.295 · Portamento 0.273. Against the empirical d = 10 null (P95 0.370), only Timing and Timbre are above.
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

**§4.5c (now §5.3 / §6.9) empirical chance null (C1; piece, 100 draws).**

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

## Historical findings — Phase B run on N = 118 (pre-block-weighting; superseded by the B3 + C1 runs above)
All guards passed (101 cells, 0 errors). Part 1 numbers are N = 118 (the N = 116 values are in
`docs/thesis.md` §11).
- **§2.2 CREPE:** octave fixes 3 frames / 3 excerpts · notes ≥ 800 ms: 259 (600 ms: 391, 1000 ms:
  198), no qualifying note: 1 excerpt (`Shostakovich_SQ8_exrpt3_MEC`) · 121 vibrato notes · rate
  undefined (no vibrato note) → median 5.811 Hz for 23 excerpts (every MEC with notes, plus SQ8
  exrpt1 EXP/EXG and exrpt3 EXP/EXG — confirmed by ear as no vibrato) · rate 37 distinct values ·
  half-extent 0–32.1 c · coverage 0–1 · portamento 139 glides in 572 transitions (24.3 %), duration
  50–312 ms (median 62), extent 80–1907 c · F0 range max 3100 c · 1626 notes ≥ 60 ms, 17 close splits.
- **§2.3 Essentia:** DR gate median 6.2 % gated (max 26 %), DR max 34.7 dB; 6 of 9 gate variants ρ < 0.9
  (0.79–0.85) · r(flux, LUFS) raw 0.874 → normalised 0.508 · 10 forwarded (short-term SD dropped).
- **§2.3c:** FLAT_MIN 0.2 → 0/60 conclusive (60/60 inconclusive; 9 at 0.1); 0 flags; LUFS MEC −26.8,
  EXP −24.1, EXG −22.6; session map not provided.
- **§2.4/§3.3b timing:** DTW forwarded 10 (7 ref-free + 3 dev; inside 3–10) · tempo log2-SD MEC 0.261
  > EXP 0.232 > EXG 0.209; IOI CV / SD and pulse clarity not monotone (0/3).
- **§4.0 (now §5.1) floors:** piece mean 0.2715 / strict P95 0.3274; loo 0.2720 / 0.3202. 4 CPUs, parallel ==
  serial ✅.
- **§4.1 (now §6.1) NestedSelect (piece):** A 0.282, **B 0.361 (only set above the strict P95)**, C 0.302, D 0.281;
  loo A 0.517, B 0.476, C 0.626, D 0.540. NB (piece) A 0.112, B 0.260, C 0.162, D 0.126.
- **§4.2 (now §6.2) (piece):** Ridge-nested · MERT-full R²v +0.016, R²a +0.446 (r_a 0.63).
- **§4.4b (now §6.3) lattice (piece):** CREPE+Essentia F1 0.389; Essentia+DTW 0.386 with **R²v 0.29, R²a 0.67**
  (best regression); every set containing MERT-full has R²a 0.446–0.459 and R²v ≈ 0.01–0.02 — the
  MERT block dominates the single-α ridge (→ block weighting). Full set F1 0.302 (below P95). Unique
  F1: MERT −0.053, CREPE −0.045, Essentia −0.051, DTW +0.036 (none significant).
- **§4.4c (now §6.4) (piece):** 4A-mert-pca R²a PCA −0.25 vs full +0.45 (Δ +0.70, q = 0.0015), R²v −0.17 vs
  +0.02 (q = 0.039), F1 0.28 vs 0.33 (n.s.) · Essentia whole block F1 0.26 vs curated 0.34 · DTW
  ref-free 0.37 / deviation-only 0.27 / both 0.32 · gate 0.80 → 0.385 (n.s.) · **4D:** Interp +
  SVM-Linear balanced 0.417 [0.346, 0.475] (Δ +0.063 vs nested, q = 0.13); NB below nested on all
  three matrices (q ≤ 0.017).
- **§4.4d (now §6.5) (piece):** NEUTRAL_RADIUS 1.0 beats 0.75 on BOTH columns for X_A (0.362 / 0.419 vs 0.282 /
  0.353) and X_C (0.374 / 0.383 vs 0.302 / 0.377) — the rule triggers (pending decision). Raw GT
  F1 0.34 vs normalised 0.30.
- **§4.5 (now §6.6)–§4.8 (now §6.6) (X_C, piece):** LA_HV and Neutral collapse (recall 0); per condition MEC 0.24 /
  EXP 0.29 / EXG 0.32; quadrant accuracy 0.25 (fixed Ridge α = 10 on X_A); Krippendorff α HA_HV
  0.31, LA_LV 0.31, others ≤ 0.03.
- **§4.9 (now §8.1) within-piece (piece):** ceilings (split-half, centred) V 0.455 [0.088, 0.718], A 0.678
  [0.520, 0.813]. ΔValence: Condition-only R² 0.047; Interpretable 0.174 (p = 0.006); **B-INC
  incremental ΔR² Interpretable +0.129 (q = 0.037), MERT-full +0.128 (q = 0.037), Interp + MERT-PCA
  +0.117 (q = 0.037)**, MERT-PCA +0.077 (q = 0.070). ΔArousal: Condition-only 0.408; Interpretable
  0.423; incremental +0.024 (p = 0.044, q = 0.070); MERT sets do not add. R²/ceiling Interp V 0.38,
  A 0.62.
- **§5.1 (now §9.1; its head → §4.1):** 180 tests, 10 raw p < .05 vs 9.0 expected, **0 q < .05** → not evidence.
- **§5.1b (now §4.2 / §4.3b) dose-response (Page, 20 pieces):** flux (normalised) z +5.22, **vibrato extent +5.14,
  coverage +4.90**, LUFS +4.74, DR +4.11, DTW deviation +4.59 (built in), rolloff −2.85, **tempo
  log2-SD −2.21 and local tempo −2.06 (both decrease)**; vibrato rate +1.90 n.s.; portamento n.s.
  Audience: arousal z +4.11, valence +1.26 n.s.
- **§5.4 (now §9.2) read-out (piece):** V r 0.25 / R² 0.12; A r 0.77 / R² 0.62; class F1 0.335.
- **§5.5 (now §10.1) CoE (piece; negative = interpretable wins):** interpretable F1 0.353, R²v 0.120, R²a 0.620 vs
  MERT-PCA 0.282 / −0.171 / −0.252 and MERT-full 0.328 / 0.016 / 0.446. Headline ΔR v −0.181,
  ΔR a −0.140, ΔR² v −0.104, ΔR² a −0.174, ΔF1 −0.071; vs MERT-PCA ΔR²a −0.872.
- **§5.6 (now §10.2) (piece):** removing Portamento RAISES F1 to 0.465 (loss −0.112, q = 0.015) and removing Timbre
  to 0.424 (q = 0.009); removing Dynamics costs R²a 0.090 (q = 0.11); Dynamics alone R²a 0.44;
  Vibrato alone F1 0.28, R²a 0.04.
- **§6.1b (now §7.3b) probe:** flux 0.86, F0 mean 0.80, centroid 0.73, HPCP entropy 0.73, pulse clarity 0.71,
  note rate 0.65, vibrato coverage 0.48; LUFS 0.08; vibrato rate −0.02; α hit the grid edge in **34**
  outer folds.
- **§6.2 (now §7.5) (piece):** + all CREPE +0.057, + DTW +0.052, + Essentia −0.001, + 2 CREPE-unique −0.019;
  §6.2b (now removed in C2) exact. Master ablation table 138 rows / 9 tiers.

### Phase B checklist — outcome of the N = 118 run
- [x] §2.2 octave corrections (3 frames); vibrato rate 37 distinct values (> 20 ✅); half-extent 0–32 c;
      substitutions: no note ≥ 800 ms 1/60 (no ⚠️), no vibrato note 23; F0 range max 3100 c.
- [x] §2.2 portamento extents within 80–2000 c; glides on 24 % of transitions — but median 62 ms and
      ≥ 12/20 MEC with a glide → **confirmed false alarms by the 2026-11-05 audit (crepe's Viterbi
      ramp) — fixed in Phase B2 (B9b)**.
- [x] §2.3 normalised flux correlates less with LUFS (0.51 vs 0.87); DR gate sensitivity ⚠️ (ρ 0.79).
- [x] §2.3c flatness test inconclusive for 60/60 at FLAT_MIN 0.2 → **under audit**.
- [x] §2.4/§3.3b timing manipulation check fails (tempo log2-SD falls MEC → EXG) → explained by the
      author (below) + **under audit**; DTW forwarded 10 (no warning).
- [x] §4.0 (now §5.1) 4 CPUs, parallel == serial ✅. NestedSelect beats the strict P95 under piece only on X_B.
- [x] §4.2 (now §6.2) / lattice MERT-full in the regression heads (`n_features_reg` 768–795), rebuild ✅ → found the
      MERT-dominance problem (→ block weighting).
- [x] §4.9 (now §8.1) B-INC: Δvalence incremental q = 0.037 (Interpretable, MERT-full); Δarousal q = 0.070.
- [x] §5.1b (now §4.2 / §4.3b) vibrato extent / coverage now escalate strongly; §6.1b (now §7.3b) α edge 34 folds.
- [x] Runtime ≈ 97 min.

## Decisions recorded (2026-11-03)
- **Excerpt QC stays OUTSIDE the notebook** (ICC, r_wg, bimodality coefficient, Kruskal–Wallis,
  crossed additive model — offline scripts).
- **Per-MERT-layer caching is ON HOLD.** *(Superseded 2026-11-16 by Phase C1: the supervisors asked for a
  per-layer study — §2.1c caches every hidden state and §4.4f (now §7.2) runs it as a DIAGNOSTIC; `MERT_LAYERS` stays
  `[5, 6, 7]` until the author decides by hand.)*
- **Timing pattern — the author's explanation:** in EXG he tended to play faster and more
  virtuosically in fast pieces and took more time in slow pieces. This is the likely reason for the
  local-tempo / duration-ratio directions (piece-dependent, no single MEC → EXG direction).
  `timing_tempo_logsd` and pulse clarity are still checked visually in §2.6 (now removed in C2).
- Block weighting of the multi-block MERT-full regression heads: deployed `'equal'`; tier 4E is
  descriptive.

## Open issues
- **Closed by Phase C2 (applied 2026-11-20, not run):** `DTW_DEV_IN_INTERP` (→ predictive / descriptive split), `CREPE_UNIQUE2` (→ probe rule), the "all-positive" floor (→ empirical null), `NEUTRAL_RADIUS` (stays 0.75), Fig 5 and the fixed α = 10 (→ deployed nested ridge), `MERT_LAYERS` (stays [5, 6, 7]). Still open: the report-only re-articulation code and legacy FFT helpers.
- **PENDING DECISIONS (unchanged):**
  - `DTW_DEV_IN_INTERP=True` vs the thesis CoE definition (condition-encoding features out of both
    sides of the cost of explainability).
  - `CREPE_UNIQUE2` hard-coded `[f0_range, vibrato_rate]` in §6.2 (now §7.5) (read off the pre-Phase-B probe;
    in the N = 118 Phase B probe F0 range is well represented, R² 0.43).
  - The "all-positive" chance floor (≈ 0.322) cited in the thesis but not computed in the notebook →
    superseded by the §4.5c (now §5.3 / §6.9) empirical null (P95 0.370 / 0.459); drop it from the thesis.
  - The §4.4d (now §6.5) NEUTRAL_RADIUS rule triggers (under piece, r = 1.0 beats 0.75 on both columns for X_A
    and X_C).
  - Fig 5 (§4.7 (now §6.8)) omits the MEC points (`cc` holds only EXP/EXG) / figure dedup.
  - §4.2 (now §6.2) / §4.7 (now §6.8) use a fixed Ridge α = 10 while the rest uses nested ridge.
  - **NEW (C1): whether to change `MERT_LAYERS`** — the author decides by hand after reading §4.4f (now §7.2)
    (Stage A suggestion, Stage B profile vs the empirical null, nested stability). Nothing is applied
    automatically. **C1 outcome:** the shortcut is in every layer, 0 / 25 beat the empirical null, the nested
    choice is unstable → recommended: keep [5, 6, 7] (record the author's decision).
- **Addressed in C1 (run 2026-11-16 — outcome in "Current results"):**
  - **Chance floor too low for the deployed classifier** — the Dummy floors ignore NestedSelect's
    selection procedure; B3: 25 noise columns → piece F1 0.354 > Dummy strict P95 0.323 (0.468 at
    C = 0.001). → §4.5c (now §5.3 / §6.9) empirical null (NestedSelect on noise, d = 25 / 10, 100 draws); every reported
    F1 is read against it (`emp_null_readout.csv`, §5.6 (now §10.2), §7.1 (now §11.1)). Outcome: 4 / 52 beat it (3 distinct, all
    with the timing block); the final form (exact widths, 1000 draws, a primary family, BH) is a C2 obligation.
  - **Per-MERT-layer study** (supervisors' request; was ON HOLD) → §2.1c + §4.4f (now §7.2) (diagnostic). Outcome: the
    shortcut is in every layer; the condition is decodable once the piece mean is removed (A2 0.90); no layer
    closes the gap to the interpretable layer.
- **Audit of 2026-11-05 — confirmed and fixed in Phase B2:** portamento false alarms (crepe's
  Viterbi ramp; B9b) · note segmentation merging notes with slides and repeated notes (B2 plateaus
  + re-articulation splits; the Phase B worry "note splitting by wide vibrato" is now covered by the
  plateau tolerance + merge rule and checked via close splits / V2–V4) · the FFT presence test
  rejecting real vibrato (B8b).
- **Audit findings still OPEN (documented, not changed):**
  - integrated LUFS Essentia − pyloudnorm = **+3.055 LU on all 6 audited excerpts** (dual-mono
    duplication — a constant offset; decide whether to subtract 3.01 LU or report as is);
  - `es.Key`: **Mozart_VC4_EXP = G minor**, the other five D major (key strength is the only key
    feature used, but the estimate is unstable);
  - `timing_tempo_logsd` Méditation **MEC 0.256 > EXP 0.158 > EXG 0.135** and pulse clarity lowest at
    MEC (✗) — the tempogram reacts to articulation, not only to rubato;
  - log-attack time **not sharper in the spiccato Mozart EXP** (✗) — LAT computed on Essentia
    waveform onsets;
  - DTW / duration ratio computed over whole files incl. silence;
  - DR gate sensitivity (6 of 9 variants ρ < 0.9);
  - §2.3c flatness noise-floor test inconclusive for 60/60.
- **Not approved / deferred (unchanged):** SHAP fitted in-sample (§5.2 (now §9.3)) · §5.3 (now §7.4) embedding-distance
  sign · §1.4 (now §4.3a) mixed model · BH in §5.1b (now §4.2 / §4.3b) · Phase C augmentation (minority classes; pitch ±1 semitone
  or tempo ±5 %, training folds only and only with the source piece in the fold; caveats: no new
  piece diversity, perturbs studied variables, unrated copies, phase-vocoder artefacts).
- "8 emotions" is mechanically 5 classes (report as 5) · the ablation lattice is a DESCRIPTIVE map
  — never select the winning subset on the same CV and then report its score.
- **`LA_HV` has n=5 from only 2 pieces** (La_Vita ×3, Meditation_Thais MEC+EXP; same at N = 116).
- **Session map not provided yet** (`SESSION_MAP_CSV`); gain equality across the 3 recording
  sessions is not independently verified.
- **Parameter ablations still pending:** CREPE (step_ms; voicing threshold and the B2 constants
  cheap from the cached tracks), Essentia frame sizes, tempogram window / tempo prior.
- **MERT generalisation / family overlap (B3):** no train score existed before B3 → §4.4e (now §7.1) (capacity
  overfitting vs piece-identity shortcut, pre-declared rules) and §5.6b (now §10.3) (redundancy vs noise vs
  dilution, pre-declared triangulation) — read their verdicts before writing Ch. 9. (B3 run: the
  piece-identity shortcut fired in MERT-full and MERT-PCA25, capacity overfitting did not.)
- §5.5 (now §10.1)'s headline CoE is a best-of over two black boxes (max(MERT-PCA, MERT-full)) — conservative;
  the two separate gaps are saved in `coe_gaps.csv`.
- B2 / B3 detectability limits: a glide is only seen between two plateau notes (a slide that never
  settles within ±40 c is part of a transition, not a note); **since B3 no re-articulation is split at
  all** (a repeated same-pitch note without a silence is one note); vibrato on notes < 500 ms or with a
  half-extent < 3 c is invisible; a window whose best rate is on the 4 / 8 Hz grid edge is not vibrato.
- **Known limitations (Phase B3, documented — not fixed):** repeated notes without a silence count as
  one (Mozart_VC4 ≈ 17 / 16 / 20 vs 19 — §2.6 (now removed in C2) V4 is INFO) · slow slides can create short "ramp
  plateaus" (a glide counted twice, e.g. B5→C6→C♯6 in Meditation_Thais_EXG) · the 80 ms glide minimum
  rejects the Méditation EXP C♯6→D6 (rise 75 ms) · residual MEC detections (V6: 5 MEC — the glides look
  like position shifts) · tempo log2-SD / pulse clarity / log-attack time / `es.Key` (Mozart_VC4_EXP
  read as G minor) — see "Audit findings still OPEN".
- **Re-articulation detector = documented negative result (B3):** splits followed the vibrato
  amplitude modulation (MEC 24 / EXP 51 / EXG 68); kept report-only in §2.2 (`REART_ENABLED = False`).

## Limitations, future work and remaining code obligations (2026-11-17)
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
> **Status 2026-11-20 (Phase C2 applied, NOT RUN):** 1 ✅ · 2 ✅ · 3 ✅ (keep flagged) · 4 ⏸ ON HOLD · 5a ✅ · 5b ✅ · 6 ✅ · 7 ✅ (re-articulation / FFT helpers left in place — pending) · 8 ✅ except the helpers · 9 open (the C2 run) · 10 open. Details: "Phase C2".

1. **Final form of the empirical null.**
   - A pre-declared primary family: headline A–D, interpretable (30), DTW, CREPE + DTW, and the multiclass model of item 4.
   - An exact-width null per primary set, with `NULL_N_DRAWS = 1000` for that family (cached), and BH within the family.
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

## Quick answers added 2026-11-17
- **"Does any feature set beat chance in classification?"** Against the *empirical* null of the deployed pipeline (§4.5c (now §5.3 / §6.9): NestedSelect on noise, P95 0.370 at 10 columns / 0.459 at 25), only the sets that contain the reference-free timing block do (DTW 0.398, CREPE + DTW 0.460, Essentia + DTW 0.373, all piece). The Dummy floor (P95 0.323) is too low for a nested-selection pipeline: 25 noise columns score 0.354. Regression is unaffected (noise R² ≈ 0), and the thesis claims rest on it.
- **"Why does MERT fail on unseen pieces?"** It recognises the piece. The nearest neighbour of every excerpt in MERT-full is another rendition of the same piece (100 %), at every depth (93–100 %, §4.4f (now §7.2)). Capacity overfitting did not fire (§4.4e (now §7.1)). MERT does carry condition information once the piece mean is removed (A2 0.90 > interpretable 0.77), but the piece identity masks it.
- **"Which technique drives arousal?"** Not identifiable with this design. The performer escalated the techniques together, so within a piece dynamics, vibrato and the DTW deviation predict each other (§5.6b (now §10.3)), and ≈ 0.35 of the 0.65 arousal R² is shared across families. Only Dynamics has a significant unique loss (q 0.028). The remedy is a factorial, one-technique-at-a-time recording.

## Conventions
- Scale **0–6**, `VA_MID=3.0`, `NEUTRAL_RADIUS=0.75` (un-tuned), `MERT_LAYERS=[5,6,7]`.
- `SEED=42`, `N_DUMMY_REPEATS=200`, `WP_N_PERM=500`, `NC_N_SPLITS=200`, `SELECT_C_GRID=
  logspace(-3,2,6)` → 13 `SELECT_CONFIGS`, `MERT_REG_REPR='full'`, **`REG_BLOCK_WEIGHTING='equal'`**,
  **`EXCLUDED_PARTICIPANTS`**, `SESSION_MAP_CSV`, `OLD_CACHE_NAMES` (§0.3); `RIDGE_ALPHAS=
  logspace(-2,4,13)`, inner `GroupKFold(5)`, `NMS_N_JOBS=os.cpu_count()`, `NMS_TIE_TOL=1e-12` (§4.0 (now §5.1));
  `TUNE_GRID_C={'C': logspace(-3,2,6)}` (§4.1 (now §6.1)); probe `ALPHAS=[1,10,100,1000,10000]` (§6.1b (now §7.3b)).
- CREPE (§2.2, B2 + B3): 10 ms step, `viterbi=True`, voiced = confidence > 0.5; octave fix ±10 frames /
  [1000,1400] c; trend 15 frames; plateaus ±40 c (range ≤ 80 c), merge < 50 c / ≤ 50 ms **only if the
  merged range stays ≤ 80 c (B3)**; **`REART_ENABLED = False` (B3)** — re-articulation dip 6 dB
  (150 ms pre / 60 ms post; sensitivity 3/6/9) report only; env 25 ms rectangular on
  the 10 ms grid; notes ≥ 60 ms (timing/portamento) and ≥ `VIB_MIN_NOTE_MS=500` (vibrato;
  500/650/800); vibrato windows 500 ms / hop 250 ms, LS sinusoid 4–8 Hz step 0.05, R² ≥ 0.50,
  half-extent ≥ 3 c, best rate not on the grid edge (B3); glides rise 80–500 ms, excess over t_min ≥ 50 ms (Viterbi 220 c/frame),
  80–2000 c, sign ≥ 80 %, rise × 1.25. Legacy (report only): boundary lag 5 / 70 c (also the
  close-split threshold), FFT presence ±0.5 Hz / [1,15] Hz ≥ 0.30, half-extent ≥ 5 c.
- Essentia: DR gate `DR_ABS_GATE_DB=-60`, `DR_REL_GATE_DB=40`; flux on L1-normalised spectra.
  §2.3c: `FLAT_MIN=0.2` (0.1/0.2/0.3) over 50 Hz–8 kHz.
- Timing/DTW: `DTW_SR=22050`, `DTW_HOP=512`, `DTW_TG_WIN=151` (≈ 3.5 s), log-normal tempo prior at
  100 BPM within 40–200 BPM, interior frames only (≥ 20), chroma-CQT alignment.
- Excerpt IDs `PieceName_ConditionCode`; participant IDs `questionnaire_{n}_P{idx:03d}`.
- `OUTPUT_DIR = '/kaggle/working/pipeline_outputs'`; caches `emb_mert_v2.npz`,
  **`emb_mert_layers_v3.npz` (C1, §2.1c)**, `crepe_tracks_v2.npz`, `rms10_v3.npz`, `feat_crepe_v4.csv`,
  `essentia_features_v2.csv/.npz`, `rms_frames_v2.npz`, `dtw_features_v4.csv`, **`null_f1_draws.csv`
  (C1, §4.5c (now §5.3 / §6.9))**, **`mert_layers/mert_layer_nested.csv` + `_pred.csv` (C1, §4.4f (now §7.2))** — the C1 caches are
  reused only when their settings hash / shape checks match (retired: pre-Phase-B names,
  `feat_crepe_v2/_v3.csv`, `dtw_features_v2/_v3.csv`; C1 retires nothing); the §2.6 (now removed in C2) audit writes only to
  `OUTPUT_DIR/manual_audit/`; §4.4e (now §7.1) writes to `OUTPUT_DIR/mert_diagnostics/`; §4.4f (now §7.2) to
  `OUTPUT_DIR/mert_layers/`. C1 constants: `NULL_N_DRAWS=100`, `NULL_DIMS=(25,10)`, `NULL_SEED=0`,
  `NULL_PROTOCOLS=('piece',)`, `RUN_NULL_LOO=False`; `ML_N_PERM_COND=1000`, `ML_SUGGEST_TOL=0.05`,
  `MERT_LAYER_PROTOCOLS=('piece','loo')`; `RUN_MANUAL_AUDIT=False`.
- Forwarding gate is UNSUPERVISED (reads only features, never `Y`); StandardScaler (and the
  BlockScaler) stay inside each CV fold. No cell other than §4.0 (now §5.1) builds a CV splitter; no cell
  rebinds a §4.0 (now §5.1) or a HELPERS:descriptors name. The `# >>> HELPERS:cv_engine … # <<<` block (§4.0 (now §5.1))
  and the `# >>> HELPERS:descriptors … # <<<` block (§2.2) are self-contained.
- **Editing the notebook:** the `.ipynb` (≈ 7 MB without the §2.6 (now removed in C2) audio outputs, ≈ 35 MB with them;
  single-line JSON) exceeds the Read tool's notebook
  limit, so NotebookEdit cannot be used on it. Edit cell by cell by splitting the JSON at the
  `{"cell_type"` boundaries, decoding / re-encoding the cell source with sed/awk under `LC_ALL=C`,
  clearing the outputs of changed code cells, and verifying the round trip byte for byte. Claude
  Code cannot run the notebook (no audio/GPU) → surgical edits + static checks only.

## Audit (Oct 2026)
Full audit of the 2026-08-16 run's outputs; corrections in three phases.
- **Phase A — pipeline corrections (applied and run 2026-10-09).** §4.0 (now §5.1) CV engine · A1 Dummy floors ·
  A2 leave-one-piece-out PRIMARY + nested ridge + §4.9 (now §8.1) · A3 MERT pos-conv restore (false alarm:
  embeddings unchanged) · A4 GaussianNB pre-declared default (failed under piece → B-NMS) · A5
  balancing + inner-CV C (+ tier 4D) · A6 out-of-sample R² / RMSE · A7 group-aware MERT probe · A14
  BH-FDR over all §5.1 (now §9.1; its head → §4.1) tests · A-QC §2.3c. Phase A headline (now historical): NB below chance under
  piece on every set (C 0.168); arousal R² X_C 0.50; 4D Interp + SVM balanced 0.385.
- **Why A2:** 95% (V) / 91% (A) of the between-excerpt variance is between-piece, condition 1% / 4%.
- **Phase B — descriptor repairs + analysis upgrades (applied 2026-10-10, run on N = 118 —
  results above).** B-NMS nested model selection · B-INC Freedman–Lane · B-NC noise ceiling ·
  B-MERTFULL · B-shared CREPE tracks + octave fix + note segmentation · B8 vibrato · B9 portamento ·
  B10 F0 range · B11 normalised flux · B12 DR gate · B13 `timing_*` + 151-frame tempogram ·
  B-NF flatness floor + session map · B-cache `_v2`.
- **Phase B2 — descriptor repairs after the manual audit (applied 2026-11-05, run in full)** — see the
  "Phase B2" section above (the table below is Phase A → Phase B; the B2 column changes are there).
- **Phase B3 — re-articulation detector disabled, merge range check, vibrato band edge, §4.4e (now §7.1) MERT
  diagnostics, §5.6b (now §10.3) family overlap (applied 2026-11-10, run 2026-11-11)** — see the "Phase B3" section above.
- **Phase C1 — empirical chance null (§4.5c (now §5.3 / §6.9)) + DIAGNOSTIC MERT layer study (§2.1c, §4.4f (now §7.2)) +
  `RUN_MANUAL_AUDIT = False` (applied 2026-11-16, run 2026-11-16)** — see the "Phase C1" section and
  "Current results" above. **Phase C2 (next):** the remaining code obligations.

| Old (Phase A) | New (Phase B) |
|---|---|
| `crepe_vibrato_rate_hz` (argmax inside the [4,8] Hz band) | `crepe_vibrato_rate_hz` (per note ≥ 800 ms, [3,10] Hz, parabolic peak, duration-weighted median over vibrato notes) |
| `crepe_vibrato_depth_cents` | `crepe_vibrato_extent_cents` (half-extent, Hilbert) + `crepe_vibrato_coverage` |
| `crepe_portamento_count` | `crepe_portamento_rate` (glides / voiced s) + `crepe_portamento_share` (glides / transitions) |
| `crepe_portamento_mean_ext_cents` | same name, new glide detector (B9) |
| `crepe_f0_range_cents` / `_cv` / `crepe_f0_mean_hz` | same names, on the octave-corrected track; range = P95 − P5 |
| — | `crepe_vibrato_n_notes` (diagnostic) |
| `essentia_spectral_flux_mean` / `_std` | `essentia_spectral_flux_norm_mean` / `_std` (+ diagnostic `essentia_spectral_flux_raw_mean`) |
| `essentia_dynamic_range_db_mean` | same name, silence-gated |
| `dtw_local_tempo_mean_bpm` | `timing_local_tempo_bpm` |
| `dtw_tempo_variability_std` / `_cv` | dropped → `timing_tempo_logsd` (SD of log2 local tempo) |
| `dtw_onset_rate` | `timing_note_rate` (CREPE notes / total duration) |
| `dtw_ioi_mean_s` / `dtw_ioi_std_s` / `dtw_ioi_cv` | `timing_ioi_mean_s` / `timing_ioi_sd_s` / `timing_ioi_cv` |
| `dtw_pulse_clarity` | `timing_pulse_clarity` (new tempogram) |
| — | `timing_octave_jump_frac` (diagnostic) |
| `dtw_global_tempo_ratio` | `dtw_duration_ratio` ("Duration ratio vs MEC (>1 = slower)") |
| `make_default_clf` / `DEFAULT_CLF_NAME='NaiveBayes'` | `make_nb_clf` / `NB_CLF_NAME`; `DEFAULT_CLF_NAME='NestedSelect'` |
| caches `feat_crepe.csv`, `essentia_features.csv/.npz`, `dtw_features.csv` | `feat_crepe_v2.csv`, `crepe_tracks_v2.npz`, `essentia_features_v2.csv/.npz`, `rms_frames_v2.npz`, `dtw_features_v2.csv` |

## Vibrato descriptor — read before quoting any vibrato result
Pre-Phase-B vibrato results (e.g. the old "vibrato n.s." dose-response) came from a broken
descriptor and are evidence about the descriptor only. Phase B (B8) repaired it (per-note analysis on
time-contiguous segments, [3,10] Hz with a presence test, extent and coverage split). In the N = 118
Phase B run **vibrato extent (z +5.14) and coverage (z +4.90) escalate MEC → EXP → EXG**, the rate
does not (z +1.90 n.s.); Vibrato alone predicts little (F1 0.28, R²a 0.04). The 2026-11-05 audit
then showed that B8's FFT presence test rejected real narrow vibrato and its notes merged with
slides → **Phase B2 (B8b)**: windowed sinusoid fit on B2 plateau notes ≥ 500 ms. **The Phase B
vibrato numbers above are superseded** and the escalation claim must be re-read in the next run
(§5.1b (now §4.2 / §4.3b)). Remaining caveats: CREPE's 10 ms resolution, only notes ≥ 500 ms and half-extent ≥ 3 c,
rates clamped to 4–8 Hz, the declared median substitutions (counts printed in §2.2). **Phase B3:** a
window whose best rate is on the 4 / 8 Hz grid edge is no longer vibrato (the B2 run had an excerpt at
exactly 4.00 Hz), and the re-articulation splits that fragmented long EXG vibrato notes below the
500 ms minimum are disabled — quote only the B3 vibrato numbers. **B3 run:**
- **Escalation:** extent z +5.45 and coverage +4.66 (both p < .0001); rate 0.00 (n.s.).
- **Méditation EXP:** coverage 0.38 → 0.70; the narrow D5 / B4 vibrato is now detected.
- **False positives:** 0.7 % of the windows in the 20 MEC excerpts.
- **Predictive weight:** Vibrato alone F1 0.316 (below the empirical d = 10 P95 of 0.370), R²a 0.050. The
  performer escalates it strongly, but on its own it predicts little; within a piece it co-varies with
  dynamics / DTW-dev (probe R² 0.32).

## Thesis document (LaTeX, NTUA template) — status 2026-10-30
Written separately from the notebook (folder `thesis_work`, `main.tex`, not in this repo).
Chapters 1–7 written (73 pages of a 120-page cap); Ch. 8 Framework, Ch. 9 Experiments and Ch. 10
Conclusion wait for the N = 116 Phase B run. Full status: `docs/thesis.md` §12 and the planning doc
"Διπλωματική: validation και πλάνο συγγραφής". When pipeline numbers change, the thesis chapters
that quote them are Ch. 7 (Dataset) and, later, Ch. 9.

## Literature hooks and thesis implications (2026-11-17)
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