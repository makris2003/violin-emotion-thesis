# Violin Emotion Thesis — Pipeline Overview

> **Revision status: Phase B applied — results pending the next Kaggle run.** Phase A was applied
> and run on Kaggle on 2026-10-09; its results (below) are **pre-Phase-B** wherever they depend on
> CREPE / Essentia / timing features or on the default classifier (all of Parts 4–6).

## Project
Master's thesis: **expressive violin techniques → acoustic result → perceived audience emotion**,
studied with an **interpretable** ML pipeline. Two categorisation deliverables (tools for detecting
technique; modalities for measuring audience effect). Framing is **categorisation + interpretable
attribution**, NOT a leaderboard — MERT is one tool category (the deep/SSL exemplar).α

## Dataset
60 excerpts = 20 pieces × 3 conditions (MEC mechanical / EXP expressive / EXG exaggerated),
recorded by the author playing violin. 118 participants across 5 **linked (ring)** forms rated
**valence 0–6, arousal 0–6** (midpoint 3.0; `0` is a real rating, not NA) + emotion tags.
Ring: 20/form, 8 anchors shared with each neighbour, 4 unique, 40 anchors total.
Ground truth = **5 classes** (4 circumplex quadrants + Neutral) from **normalized** VA means.
Stimuli were **NOT loudness-normalized** for the participants (level is part of the stimulus).

## Pipeline (notebook/thesis_pipeline.ipynb, Kaggle P100, 101 cells, runs top-to-bottom)
- **Part 1 — ground truth:** parse → **per-rater normalization (§1.1b, new ground truth)** →
  aggregate/derive labels → validation (§1.2b) → label matrix → rating stats → lift.
- **Part 2 — features:** MERT (768-dim, layers 5–7; **§2.1 restores the positional-conv
  weight-norm tensors from the checkpoint and asserts them**, cache `emb_mert_v2.npz`) · **CREPE
  (§2.2, Phase B): tracks cached once (`crepe_tracks_v2.npz`), octave correction, note
  segmentation, then vibrato / portamento / F0 descriptors from the tracks** · Essentia
  (dynamics/timbre/tonal; **L1-normalised flux, silence-gated dynamic range**) · **§2.3c
  recording-level sanity check** (report only: LUFS robust z, within-piece LUFS, **spectral-
  flatness noise-floor** chain test, clipping, optional session map) · **timing + DTW (§2.4/§2.4b):
  `timing_*` reference-free family from CREPE note onsets + a 151-frame tempogram, `dtw_*`
  deviation family vs MEC**. Each extracted in full, then **forwarded through one shared
  UNSUPERVISED gate** `unsup_forward_select` (never sees Y). madmom/MusiCNN remain skip stubs.
  The descriptor helpers live in the `# >>> HELPERS:descriptors … # <<<` block of §2.2.
- **Part 3 — EDA/validation** (incl. §3.3 CREPE sanity, §3.3b timing sanity + manipulation check).
- **Part 4 — prediction:** **§4.0 = the one CV engine** (protocols, **nested model selection =
  default classifier**, nested ridge, metrics, chance floors, Freedman–Lane + noise-ceiling
  helpers) · §4.1 NestedSelect (★default) + 8-classifier **sensitivity table** (never used for
  selection) · §4.2 VA regression (+ `Ridge-nested · MERT-full` row) · §4.4 + **§4.4b/c/d ablation
  study** · §4.5/§4.5b per-class + re-seeded Dummy floors (+ the NB Phase A default row) ·
  §4.6–§4.8 on the default classifier / X_C · **§4.9 within-piece prediction** (piece-centred
  targets & features, piece CV only, within-piece permutation, **B-INC Freedman–Lane incremental
  test, B-NC split-half noise ceiling**).
- **Part 5 — attribution:** condition-delta (**BH-FDR over all §5.1 tests**) · dose-response
  (Page's trend) · SHAP · embedding distance · interpretable linear read-out · cost of
  explainability (both protocols; headline + two separate gaps) · **§5.6 sub-family ablation**.
- **Part 6 — tool characterization:** MERT↔feature correspondence probe (**GroupKFold(5) over
  pieces, α nested**) · tool complementarity (default classifier, both protocols; blocks taken
  from the §4.4b lattice stacks) · §6.2b guard.
- **Part 7 — synthesis/save** (NestedSelect F1 per protocol + chosen configs, chance mean/P95, NB
  Phase A default F1, mean F1 of the 8 sensitivity classifiers).
- **Cleaning (separate scripts):** questionnaire_cleaning.py (participant screening, MCD/
  Mahalanobis etc.), excerpt_cleaning.py (stimulus QC, ICC, manipulation check). Exclusions
  NOT yet applied (pipeline runs on all 118).

## Evaluation protocol (§4.0 — read before quoting any Part 4–6 number)
- **`PRIMARY_PROTOCOL='piece'`** (leave-one-piece-out, `LeaveOneGroupOut` on `PIECE_GROUPS`);
  `loo` is always reported next to it as the optimistic secondary; `PROTOCOLS=('piece','loo')`,
  printed piece first, with the optimism gap. `get_cv(protocol)` is the ONLY splitter builder
  (`'piece5'` = GroupKFold(5) on pieces, used only by the §6.1b probe).
- **Default classifier (Phase B, B-NMS): nested model selection**, `DEFAULT_CLF_NAME='NestedSelect'`,
  `cv_predict_default(X, Y, protocol)` = `cv_predict_nested_select(X, Y, protocol,
  configs=SELECT_CONFIGS)`. Inside every outer training fold an inner GroupKFold(5) over the
  training pieces scores each of the 13 `SELECT_CONFIGS` (§0.3: NB · SVM-Linear balanced ·
  LogReg balanced, C ∈ `logspace(-3,2,6)`, same C for all labels) by **pooled inner OOF
  macro-F1**; ties → list order (NB, then smallest C, SVM before LogReg); the winner is refitted
  on the whole outer training fold. Returns the chosen config per outer fold; every call prints
  the config frequencies. Rationale: choosing the classifier on the CV that reports it is
  selection bias; nesting makes the score an unbiased estimate of the select-then-fit procedure.
  Runtime: sha1 content-hash memo for byte-identical calls + joblib/loky over outer folds
  (`NMS_N_JOBS = os.cpu_count()`, printed), each fold under `threadpool_limits(1)`; **§4.0
  asserts parallel == serial exactly** on X_B (predictions, chosen configs, inner scores).
- **NaiveBayes** (`NB_CLF_NAME`, `make_nb_clf() → GaussianNB()`) = the Phase A pre-declared
  default; it failed under `piece` and stays as a reported row ("NaiveBayes (pre-declared Phase A
  default — failed under piece)") in §4.1 / §4.5b / §7.1 / tier 4D.
- `cv_predict_multilabel(make_est, X, Y, protocol, tune_grid=None)` — manual outer loop, binary
  relevance, `SafeMultiOutputClassifier` semantics; with `tune_grid`, C is chosen per label
  inside the training fold by inner GroupKFold(5) on pieces (pooled OOF F1; ties → strongest
  regularisation; skipped → C=1.0 if <5 training positives or positives in <2 pieces).
  Asserted equal to the old `cross_val_predict(SafeMultiOutput(GaussianNB))` on X_A (the guard
  tests the engine, not the default).
- `nested_ridge_predict(X, y, protocol)` — α ∈ `RIDGE_ALPHAS=logspace(-2,4,13)` chosen inside each
  outer training fold by inner GroupKFold(5) on pieces (pooled SSE; ties → largest α); replaces
  every `RidgeCV`. `NestedRidgeOperator` = its closed-form twin (used only for §4.9's permutation
  refits; equivalence asserted on the observed targets).
- **B-MERTFULL:** the regression heads see MERT at full dimensionality (`MERT_REG = X_MERT_FULL`
  when `MERT_REG_REPR='full'`, §0.3); the classification heads keep the PCA-forwarded `X_A`.
- `freedman_lane_targets(y, Z, perms)` (B-INC) and `split_half_ceiling(raters, items, values,
  item_order, n_splits, seed, item_groups)` (B-NC) live in the §4.0 HELPERS block.
- Metrics: macro-F1; r **and** `oos_r2` (baseline = training-fold mean) **and** `rmse`.
  Negative LOOCV r with weak predictors is an artefact — read R² (printed note).
- **Chance floors (A1):** Dummy seeded `base + 1000·fold + label`, `N_DUMMY_REPEATS=200` repeats →
  mean / SD / P95. `CHANCE_FLOOR_MEAN` (strongest strategy's mean → `marginal_f1`),
  `CHANCE_FLOOR_P95` = **max P95 over strategies** (the strict "beats chance" line).

## Key variables
- `df_long` — per-rating (+ `valence_norm/arousal_norm` after §1.1b).
- `df_agg`/`df_all` — per-excerpt; `valence_mean/arousal_mean` = NORMALIZED (ground truth),
  `*_raw` kept in parallel. `emotion_labels` → `Y`; `top_tags` descriptive only.
- `Y` (60×N), `mlb.classes_`. Full blocks: `df_mert`, `df_crepe`, `df_essentia`, `df_dtw`.
  Forwarded: `emb_mert_fwd`, `df_crepe_fwd`, `df_ess_fwd`, `df_dtw_fwd`; lists `*_FORWARD`.
- **CREPE (§2.2):** `CREPE_CANDIDATES` (10 descriptors), `CREPE_LABELS`, `CREPE_DIAGNOSTIC =
  ['crepe_vibrato_n_notes']`; `CREPE_TRACKS`, `CREPE_DIAG`, `CREPE_CENTS`, `CREPE_VOICED`;
  `VIB_N_SUBST_NONOTES` / `VIB_N_SUBST_NOVIB` (declared substitution counts, read by §5.6).
- **Diagnostic-only columns** (never forwarded, never in `df_interp`, excluded from the 4A whole-
  block gate and the 4B gate): `CREPE_DIAGNOSTIC`, `ESSENTIA_DIAGNOSTIC =
  ['essentia_spectral_flux_raw_mean']`, `TIMING_DIAGNOSTIC = ['timing_octave_jump_frac']`.
- Assembly: `X_A` (MERT-fwd), `X_B` (CREPE-fwd), `X_C` (MERT+CREPE+Essentia+DTW),
  `X_D` (MERT+Essentia), `X_F` (MERT+DTW — the complementarity contrast in §6.2),
  `X_MERT_FULL` (768-dim, the representational analyses and the regression heads).
  Regression twins (B-MERTFULL): `MERT_REG`, `X_C_REG`, `X_D_REG`, `X_F_REG` (MERT block replaced
  by `MERT_REG`). `STRICT_BLOCKS=('MERT','Essentia')` → missing block raises, never silent
  zero-fill; DTW is non-strict but a missing row still warns loudly in `bfv`.
- **Timing has two families** (§2.4): (1) `DTW_REFFREE` = the `timing_*` reference-free family
  (graded for all 60 incl. MEC, condition-INdependent); (2) `DTW_DEV` = the `dtw_*` within-piece
  warp-path deviation vs the piece's own MEC, which is the self-alignment identity (0 deviation /
  1 ratio, `DTW_DEV_SELF`) at MEC **by construction** and so partly encodes condition. The
  variable names (`DTW_REFFREE`, `df_dtw`, `DTW_*`) and the lattice block name `DTW` were kept.
  §2.4b splits them: `DTW_MODEL_COLS` (→ X_C/X_F, family 1 only by default,
  `DTW_DEV_IN_MODELS=False`) vs `df_dtw_attr` (→ `df_interp`, both families,
  `DTW_DEV_IN_INTERP=True`). Flip either constant to change the regime.
- Interpretable frame + labels for Part 5: `df_interp` + `INTERP_FEAT_LABELS`
  (CREPE candidates + Essentia + DTW, built in §5.1; §5.1b/§5.2/§5.4/§5.5/§5.6 derive from it).
- Results containers: `all_recs` (8 clf + NestedSelect × 4 sets × 2 protocols, `protocol` key;
  NestedSelect rows carry `selected` / `chosen`), `DEFAULT_RECS[(set, protocol)]` (NestedSelect),
  `NB_RECS[(set, protocol)]`, `DEFAULT_SET='C: Combined'`, `REG_RES[protocol]` (`reg_res` =
  primary; incl. `REG_MERTFULL`), `PRED_QUADS`, `INTERP_CV_R[protocol][dim]`,
  `INTERP_CV_F1[protocol]`, `COE[protocol]`, `BLOCK_F1[protocol]`, `df_within_piece`,
  `df_wp_ceiling` / `WP_CEILING`, `df_delta_tests` (+ `q_ME/EX/MG`), `df_reclevel`,
  `df_inner_skips`.

## Ablation study (§4.4b → §5.6, one engine)
`ablate()` in **§4.4b** is the ONLY scorer; every tier appends to `ABL_ROWS`/`ABL_PRED` and the
consolidated `df_ablation` → `ablation_master.csv` is written at the end of §5.6.
- **Protocol** (identical to §5.5/§6.2): default classifier (**nested selection**) → F1-macro;
  nested piece-grouped ridge on `X_reg` (default: X itself; the `MERT_REG`-based matrix whenever
  MERT is in the set, `ABL_BLOCKS_REG` / `_abl_stack_reg`) → r, `r2_val/r2_aro`,
  `rmse_val/rmse_aro`; scaler inside each fold. Rows record `selected` (config frequencies),
  `n_configs_selected`, `n_features_reg`. Reads `Y` only to SCORE, never to select — forwarding
  stays unsupervised.
- **Both CV protocols, `piece` first** (primary) — quote `piece` for generalisation claims.
- **Stats:** bootstrap 95% CI over excerpts · **paired permutation** p (predictions swapped per
  excerpt, metric recomputed — valid for F1, r and R²) · **BH-FDR within (tier, protocol)** over
  F1, r_val, r_aro, R²_val, R²_aro. Runs with different ground truth are `paired=False` → NaN p
  (§4.4d). `beats_chance` = F1 > `ABL_FLOOR_P95`.
- **Tiers:** `1-blocks` (all 15 subsets of MERT/CREPE/Essentia/DTW → unique vs marginal
  contribution, §4.4b) · `3-subfamily` (leave-one-family-out, §5.6; Vibrato and Portamento now 3
  features each) · `4A-*` + `4B-gate` (config knobs, §4.4c; `4A-mert-pca` unchanged — it is the
  evidence for B-MERTFULL; **`4A-crepe-portamento` removed in Phase B**, the B9 features are
  rates/shares by construction) · **`4D-balancing`** (§4.4c: **nested selection (deployed, ref)**
  / NB empirical priors (Phase A default) / NB balanced priors / SVM-Linear C=1 unbalanced /
  SVM-Linear balanced + inner-CV C, on X_A, X_C, interpretable; classification only; per-matrix
  reference via `abl_vs_ref(..., ref_map=)`) · `4C-neutral-radius` + `4C-ground-truth` (§4.4d;
  regression on `MERT_REG` / `X_C_REG`).
- **Guards:** §4.0 engine equivalence (GaussianNB) · §4.0 NestedSelect parallel == serial · §4.4b
  lattice rebuilds `X_C/X_D/X_F` **and `X_C_REG/X_D_REG/X_F_REG`** · §4.4c 4D nested-on-X_A equals
  the lattice `MERT` row · §4.4d label re-derivation reproduces `Y` · §4.9 closed-form operators
  equal `nested_ridge_predict` (reference and Condition + Features) · **§6.2b** §6.2's additive
  numbers equal the §4.4b lattice rows (both protocols) · §5.5 interpretable r equals §5.4 exactly
  · §2.5 no cache path carries a pre-Phase-B name. Gate sweeps write audits to
  `OUTPUT_DIR/ablation_gate_audit/`.
- **NOT run:** CREPE step_ms, Essentia frame sizes, **per-MERT-layer** (§2.1 averages layers 5–7 —
  have it cache per-layer once and future layer ablations become free) need re-extraction; the
  CREPE voicing threshold and the B8/B9 descriptor constants are now cheap from
  `crepe_tracks_v2.npz` but not approved yet; tempogram window / tempo prior not swept.

## Phase B — what changed (applied 2026-10-10, results pending the next Kaggle run)
Analysis upgrades: **B-NMS** nested model selection (default classifier) · **B-INC** Freedman–Lane
incremental test in §4.9 (`r2_cond_plus`, `d_r2_incremental`, `p_incremental`, `q_incremental`;
reduced model = OLS on the centred condition one-hot; BH over the 8 set × target tests) ·
**B-NC** split-half noise ceiling in §4.9 (`NC_N_SPLITS=200` rater splits, Spearman–Brown,
centred + uncentred, `r2_over_ceiling`, `within_piece_noise_ceiling.csv`) · **B-MERTFULL**
regression heads on MERT-full. Descriptor repairs (re-extraction): **B-shared** CREPE tracks +
`fix_octave_errors` + `segment_notes` · **B8** vibrato · **B9** portamento · **B10** F0 range ·
**B11** normalised flux · **B12** DR silence gate · **B13** `timing_*` family + 151-frame
tempogram with log-normal prior · **B-NF** flatness noise floor + optional session map ·
**B-cache** `_v2` names + assert.

| Old (Phase A) | New (Phase B) |
|---|---|
| `crepe_vibrato_rate_hz` (argmax inside the [4,8] Hz band) | `crepe_vibrato_rate_hz` (per note ≥ 800 ms, [3,10] Hz, parabolic peak, duration-weighted median over vibrato notes) |
| `crepe_vibrato_depth_cents` | `crepe_vibrato_extent_cents` (half-extent, Hilbert) + `crepe_vibrato_coverage` (vibrato-note duration / notes ≥ 800 ms) |
| `crepe_portamento_count` | `crepe_portamento_rate` (glides / voiced s) + `crepe_portamento_share` (glides / transitions) |
| `crepe_portamento_mean_ext_cents` | same name, new glide detector (B9) |
| `crepe_f0_range_cents` / `_cv` / `crepe_f0_mean_hz` | same names, on the octave-corrected track; range = P95 − P5 |
| — | `crepe_vibrato_n_notes` (diagnostic) |
| `essentia_spectral_flux_mean` / `_std` | `essentia_spectral_flux_norm_mean` / `_std` (+ diagnostic `essentia_spectral_flux_raw_mean`) |
| `essentia_dynamic_range_db_mean` | same name, silence-gated |
| `dtw_local_tempo_mean_bpm` | `timing_local_tempo_bpm` |
| `dtw_tempo_variability_std` / `_cv` | dropped → `timing_tempo_logsd` (SD of log2 local tempo, octaves) |
| `dtw_onset_rate` | `timing_note_rate` (CREPE notes / total duration) |
| `dtw_ioi_mean_s` / `dtw_ioi_std_s` / `dtw_ioi_cv` | `timing_ioi_mean_s` / `timing_ioi_sd_s` / `timing_ioi_cv` |
| `dtw_pulse_clarity` | `timing_pulse_clarity` (new tempogram) |
| — | `timing_octave_jump_frac` (diagnostic) |
| `dtw_global_tempo_ratio` | `dtw_duration_ratio` ("Duration ratio vs MEC (>1 = slower)") |
| `make_default_clf` / `DEFAULT_CLF_NAME='NaiveBayes'` | `make_nb_clf` / `NB_CLF_NAME`; `DEFAULT_CLF_NAME='NestedSelect'` |
| caches `feat_crepe.csv`, `essentia_features.csv/.npz`, `dtw_features.csv` | `feat_crepe_v2.csv`, `crepe_tracks_v2.npz`, `essentia_features_v2.csv/.npz`, `rms_frames_v2.npz`, `dtw_features_v2.csv` (`OLD_CACHE_NAMES` asserted absent) |

Deviations from the literal Phase B spec (all stated in the cells): vibrato notes ≥ 800 ms (not
400; decision 1); tempogram statistics over **interior** frames only (frames whose window lies
inside the excerpt — edge frames ramp to zero and fake tempo variability); the raw diagnostic flux
is computed on the gated frames too; `feat_crepe_v2.csv` is recomputed from the tracks every run;
the timing features raise if an excerpt has < 3 notes; §6.2 takes its lattice-equivalent blocks
from the §4.4b stacks (identical arrays, so the §6.2b equality guard cannot trip over float32
rounding interacting with the discrete nested choice).

## What to check after the Phase B Kaggle run
- [ ] §2.2: octave corrections per excerpt; vibrato rate has **> 20 distinct values** (§3.3 ✅/⚠️);
      coverage / half-extent ranges plausible (half-extent typically ~10–40 cents); **declared
      substitution counts** (no note ≥ 800 ms: ___ / 60 — ⚠️ fires above 10; no vibrato note:
      ___) and the 600/800/1000 ms sensitivity print; F0 range max (expected well below the old
      5343 c); segmentation diagnostic (adjacent notes < 70 c).
- [ ] §2.2: portamento extents within 80–2000 c and glides **not** on every transition.
- [ ] §2.3: r(flux, LUFS) raw vs normalised (expected: normalised correlates less); DR max and %
      gated (⚠️ > 30 %); DR gate sensitivity ρ (⚠️ < 0.9).
- [ ] §2.3c: flatness distribution, flatness-floor flags and their FLAT_MIN 0.1/0.2/0.3 sensitivity;
      "session map not provided" until `session_map.csv` is uploaded.
- [ ] §2.4/§3.3b: timing manipulation check on the new `timing_*` features; DTW forwarding count
      (the "outside 3–10" warning).
- [ ] §4.0: `os.cpu_count()` / NMS_N_JOBS, parallel == serial guard ✅; runtime per section.
- [ ] Nested-selection config frequencies (per call); **does NestedSelect beat the strict chance
      P95 under piece** (§4.1/§4.5b/§7.1); NB row stays visible.
- [ ] §4.2 / lattice: MERT-full in the regression heads (`n_features_reg`), regression rebuild
      assert ✅.
- [ ] §4.9: B-INC incremental p / q for Δarousal and Δvalence; noise ceilings (centred and
      uncentred) and R² / ceiling; redraw count.
- [ ] §5.1b dose-response for the repaired vibrato / timing features; §5.6 vibrato family (3
      features) and its printed caveats; §6.1b probe for the new CREPE descriptors (re-check the
      `CREPE_UNIQUE2` pair of §6.2).

## Current findings — Phase A Kaggle run of 2026-10-09 (GPU T4; 101 cells, 0 errors) — **pre-Phase-B**
Every number below that depends on CREPE / Essentia / timing features or on the default classifier
(all of Parts 4–6, §2.3c) is **pre-Phase-B** and is superseded by the next run; Part 1 and the §1.x
statistics are unchanged by Phase B.
- **All guards passed:** §4.0 engine equivalence, §4.4b lattice rebuild, §4.4d label
  reproduction, §6.2b, §4.9 closed-form permutation equivalence, §5.5↔§5.4 cross-check.
- **Dummy floors fixed:** uniform mean 0.272 (loo) / 0.2715 (piece); stratified 0.183 / 0.161;
  strict P95 0.320 (loo) / 0.327 (piece).
- **MERT (A3) was a false alarm:** the restore ran, but the embeddings are identical to the pre-fix
  run (§6.1 correspondence table and every MERT-based F1 match to 4 decimals). The transformers
  "newly initialized" warning was spurious; pre-audit MERT numbers were valid. The assert stays.
- **Classification, piece (PRIMARY):** the pre-declared default GaussianNB is BELOW chance on every
  set (F1-macro A 0.112, B 0.287, C 0.168, D 0.113); on X_C three classes collapse (HA_HV, LA_HV,
  Neutral recall 0); quadrant accuracy 0.25; Krippendorff α ≤ 0 for all labels except LA_LV
  (0.41). LOOCV for comparison: C 0.686.
- **Tier 4D (piece):** SVM-Linear balanced + inner-CV C beats NB deployed on X_A (+0.143,
  q=0.023), X_C (+0.156, q=0.005), Interp (+0.151, q=0.023). Interp + SVM balanced = 0.385
  [0.327, 0.434] — the only configuration above the strict chance P95. NB balanced priors: no
  effect. → Decision (applied in Phase B): nested model selection replaces the single default; NB
  stays reported as "pre-declared Phase A default, failed under piece".
- **Regression, piece (nested ridge):** arousal R² — X_C 0.50, CREPE+Essentia 0.52, Essentia
  0.40; valence R² ≈ 0 (max 0.23, CREPE+DTW). MERT-PCA: R²v −0.17, R²a −0.25; MERT-full: R²a 0.45
  (Δ vs PCA +0.70, q=0.0015) → PCA-95 discards the arousal-relevant variance and keeps piece
  identity (a tool-taxonomy finding; → B-MERTFULL).
- **Cost of explainability, piece (negative = interpretable wins):** headline ΔR valence −0.215,
  ΔR arousal −0.054, ΔF1 −0.127; vs MERT-PCA ΔR² arousal −0.756.
- **§4.9 within-piece (piece-centred, LOPO):** Δarousal — Condition-only R² 0.41, Interpretable
  0.47 (p=0.002), MERT-full 0.38, MERT-PCA 0.27. Δvalence — Condition-only 0.047 (p=0.034),
  Interpretable 0.20 (p=0.004), MERT-full 0.175, MERT-PCA 0.134. Within-piece SD: V 0.234,
  A 0.285 (scale 0–6). The "beats condition" difference was not tested (→ B-INC, pending).
- **§5.1 BH:** 0 of 174 tests have q < .05 (12 raw p < .05 vs 8.7 expected) → the condition-delta
  correlations are not evidence; do not cite them as findings.
- **§6.1b grouped probe:** well represented across pieces — F0 mean 0.80, spectral centroid 0.73,
  HPCP entropy 0.73, pulse clarity 0.59; loudness only 0.08; most reference-free timing features
  ≤ 0. α hit the grid edge (10000) in 32 folds (= shrink to the mean; harmless for those targets).
- **§2.3c recording level:** 0 LUFS outliers, 0 clipping, 21 noise-floor flags (15 excerpts
  inconclusive). EXG conclusive floor mean −66.2 dBFS vs MEC −71.4 / EXP −72.5; EXG LUFS +4.2 LU
  vs MEC. Author: 3 recording sessions with position markers; in EXG the performer deliberately
  played louder. Gain equality across sessions is NOT independently verified (limitation); the
  5th-percentile floor estimator was unreliable when an excerpt has no silence (→ B-NF, pending).
  A session map is not available yet.
- Unchanged from earlier runs: dose-response (§5.1b) — flux/loudness rise strongly, vibrato n.s.
  (with the OLD, broken vibrato descriptor and the raw flux — both replaced in Phase B);
  normalization (§1.2b) raw↔norm r = 0.996 (V) / 0.993 (A), 10/60 class changes, 4/60 quadrant
  changes; variance structure 95% (V) / 91% (A) between-piece, condition 1% / 4%. Pre-audit LOOCV
  CoE (+0.112 / +0.051 / +0.145) is superseded.

## Conventions
- Scale **0–6**, `VA_MID=3.0`, `NEUTRAL_RADIUS=0.75` (un-tuned), `MERT_LAYERS=[5,6,7]`.
- `SEED=42`, `N_DUMMY_REPEATS=200`, `WP_N_PERM=500`, `NC_N_SPLITS=200`, `SELECT_C_GRID=
  logspace(-3,2,6)` → 13 `SELECT_CONFIGS`, `MERT_REG_REPR='full'`, `SESSION_MAP_CSV`,
  `OLD_CACHE_NAMES` (§0.3); `RIDGE_ALPHAS=logspace(-2,4,13)`, inner `GroupKFold(5)`,
  `NMS_N_JOBS=os.cpu_count()`, `NMS_TIE_TOL=1e-12` (§4.0); `TUNE_GRID_C={'C': logspace(-3,2,6)}`
  (§4.1); probe `ALPHAS=[1,10,100,1000,10000]` (§6.1b).
- CREPE (§2.2, unchanged extraction): 10 ms step, `viterbi=True`, voiced = confidence > 0.5.
  Descriptors: octave fix ±10 frames / [1000,1400] c; trend 15 frames, boundary lag 5 / 70 c;
  notes ≥ 60 ms (timing/portamento) and ≥ `VIB_MIN_NOTE_MS=800` (vibrato; sensitivity 600/800/1000);
  vibrato band [3,10] Hz, N=4096, presence ±0.5 Hz / [1,15] Hz ≥ `VIB_PRESENCE_MIN=0.30` (empirical
  noise of the ratio ≈ 0.15–0.17, P95 ≈ 0.27 at 0.8 s) and half-extent ≥ 5 c; glides 40–400 ms,
  80–2000 c, sign ≥ 80 %, duration = 10–90 % rise × 1.25.
- Essentia: DR gate `DR_ABS_GATE_DB=-60`, `DR_REL_GATE_DB=40`; flux on L1-normalised spectra.
  §2.3c: `FLAT_MIN=0.2` (sensitivity 0.1/0.2/0.3) over `FLAT_BAND_HZ=(50,8000)` (clamped to
  Nyquist), RMS < P20, ≥ 20 noise-like frames else inconclusive; session "confined" = ≥ 80 %.
- Timing/DTW: `DTW_SR=22050`, `DTW_HOP=512`, **`DTW_TG_WIN=151`** (≈ 3.5 s), log-normal tempo prior
  centred on 100 BPM within 40–200 BPM, interior frames only (≥ 20), chroma-CQT alignment.
- Excerpt IDs: `PieceName_ConditionCode` (e.g. `Bach_Adagio_S1_MEC`).
- Participant IDs: `questionnaire_{n}_P{idx:03d}` (alias `S{n}_P{idx:03d}`).
- `OUTPUT_DIR = '/kaggle/working/pipeline_outputs'`; caches `emb_mert_v2.npz`,
  `crepe_tracks_v2.npz`, `feat_crepe_v2.csv`, `essentia_features_v2.csv/.npz`,
  `rms_frames_v2.npz`, `dtw_features_v2.csv`.
- Forwarding gate is UNSUPERVISED (reads only features, never `Y`); StandardScaler stays inside
  each CV fold. No cell other than §4.0 builds a CV splitter; no cell rebinds a §4.0 or a
  HELPERS:descriptors name.
- Claude Code cannot run the notebook (no audio/GPU) → surgical edits + static checks + local
  synthetic tests only. The `# >>> HELPERS:cv_engine … # <<<` block in §4.0 and the
  `# >>> HELPERS:descriptors … # <<<` block in §2.2 are self-contained so they can be extracted
  and tested outside the notebook.

## Open issues
- "8 emotions" is mechanically 5 classes (report as 5) · figure dedup partially pending ·
  exclusions not applied · the ablation lattice is a DESCRIPTIVE map — never select the winning
  subset on the same CV and then report its score · DTW forwarded 11 columns at N=60 in the
  2026-08-16 run (the §2.4b "outside 3–10" warning) — re-check with the Phase B timing family.
- **`LA_HV` has n=5 from only 2 pieces** (La_Vita ×3, Meditation_Thais MEC+EXP) — under `piece`
  it is learned from one piece at a time; inner-CV tuning is skipped for it in many folds.
- **Session map not provided yet** (`SESSION_MAP_CSV`, columns `excerpt_id, session`); gain
  equality across the 3 recording sessions is not independently verified (the performer reports
  playing louder in EXG on purpose).
- **Parameter ablations still pending:** CREPE (step_ms re-extraction; voicing threshold and the
  B8/B9 constants now cheap from the cached tracks), Essentia frame sizes, tempogram window /
  tempo prior, per-MERT-layer.
- **Phase C (deferred idea) — augmentation:** supervisor's suggestion to augment the minority
  positive-valence classes (pitch shift ±1 semitone, tempo ±5%, never combined); copies used
  **only in training folds, and only when their source piece is in that training fold** (test =
  originals only; gate/PCA fitted on originals; scoring/permutations on the 60 originals).
  Caveats: cannot add piece diversity (LA_HV comes from 2 pieces); perturbs studied variables (F0
  mean, tempo, note rate) while keeping labels; copies' labels were never rated; phase-vocoder
  artefacts affect Essentia timbre. Augmentation later as an ablation tier.
- **Not approved yet:** SHAP fitted in-sample (§5.2) · §5.3 embedding-distance sign · §1.4 mixed
  model · BH in §5.1b.
- **Known, not fixed (out of scope):** Fig 5 VA scatter (§4.7) never draws MEC points (`cc` holds
  only EXP/EXG); README points to `docs/thesis_handoff.md`, the file is `docs/thesis.md`.
- §5.5's headline CoE is a best-of over two black boxes (max(MERT-PCA, MERT-full)) —
  conservative; the two separate gaps are saved in `coe_gaps.csv`.
- B9 detectability limit: a glide is only seen if its trend crosses the 70-cent boundary rule,
  i.e. slopes below ≈ 1.4 cents/ms over 50 ms are not segmented into a transition.
- Runtime: nested selection (full 13-config grid, ~115 distinct calls after the memo) needs
  ≈ 3 h of serial CPU work by local measurement (≈ 40 s per serial `piece` call, ≈ 3× that for
  `loo`); with parallel outer folds that is an estimated **≈ 45 min extra on 4 cores, ≈ 1.5 h
  on 2 cores** — check §4.0's printed `os.cpu_count()` and the run time.

## Audit (Oct 2026)
Full audit of the 2026-08-16 run's outputs; corrections run in three phases.
- **Phase A — pipeline corrections (APPLIED and run 2026-10-09).** A-shared §4.0 CV engine · A1
  DummyClassifier floors (re-seeded per fold × label, 200 repeats — were collapsing to F1=0) · A2
  leave-one-piece-out as PRIMARY protocol everywhere + nested piece-grouped ridge + §4.9
  within-piece prediction + CoE handles a negative gap · A3 MERT positional-conv weight restore +
  assert, cache v2 (false alarm: embeddings unchanged) · A4 GaussianNB as the single pre-declared
  default classifier, all best-of selection removed (failed under piece → B-NMS) · A5 class
  balancing + inner-CV C for the sensitivity classifiers (+ tier `4D-balancing`) · A6
  out-of-sample R² and RMSE next to r (+ R² in the ablation BH family) · A7 group-aware MERT probe
  with nested α · A14 BH-FDR over all §5.1 tests · A-QC recording-level sanity check (§2.3c).
- **Why A2:** 95% (V) / 91% (A) of the between-excerpt variance is between-piece, condition 1% /
  4% → LOOCV mostly measures piece recognition.
- **Phase B — descriptor repairs + analysis upgrades (APPLIED 2026-10-10, results pending the
  next Kaggle run)** — see "Phase B — what changed". **Phase C** — see Open issues.

## Vibrato descriptor — read before quoting any vibrato result
**Pre-Phase-B results** (e.g. the dose-response "vibrato n.s.") were obtained with a broken
descriptor: `crepe_vibrato_rate_hz` was the argmax of a spectrum inside the same [4,8] Hz band the
signal was already bandpassed to (it could never report "no vibrato"), both features ran over
concatenated non-contiguous voiced frames, and depth conflated extent with coverage — those nulls
are evidence about the descriptor only. **Phase B (B8) repaired it:** per-note analysis on
time-contiguous segments, rate in [3,10] Hz with an explicit presence test, extent and coverage
split. A vibrato null from the Phase B run is a **musical result at this N**, with the remaining
caveats: CREPE's 10 ms resolution, only notes ≥ 800 ms analysed, and the declared median
substitutions (§2.2 prints the counts; ⚠️ if > 10 excerpts have no qualifying note — then the
vibrato family is unreliable at this note length).
