# Violin Emotion Thesis — Pipeline Overview

> **Revision status (2026-11-03): Phase B has been RUN on N = 118** (Kaggle, Tesla T4, papermill
> 2026-10-09 23:45 → 2026-10-10 01:22 UTC, ≈ 97 min, 101 cells, 0 errors; outputs saved in the
> committed notebook). Its results (below) are **N = 118 / pre-block-weighting**.
>
> **Committed notebook = an N = 116 run of 2026-11-03** (papermill 15:26 → 17:11 UTC; §1.1 shows
> 118 → 116, F5 32 → 30) with (1) the participant exclusions in §1.1 (`EXCLUDED_PARTICIPANTS`),
> (2) equal-share block weighting of the multi-block MERT-full regression heads
> (`REG_BLOCK_WEIGHTING = 'equal'`, tier `4E-block-weighting`) and (3) the **temporary §2.6 manual
> descriptor audit**. Its §2.6 outputs are the evidence of the **manual audit of 2026-11-05**; its
> Part 2–7 numbers are **not recorded here** and are superseded by Phase B2 (they still use the
> Phase B CREPE descriptors).
>
> **Phase B2 — descriptor repairs after the manual audit — APPLIED 2026-11-05, NOT RUN** (see the
> section below): plateau note segmentation + re-articulation splits, Viterbi-aware portamento,
> windowed sinusoid-fit vibrato, caches `feat_crepe_v3.csv` / `dtw_features_v3.csv` / `rms10_v3.npz`.
> Changed cells (outputs cleared): §0.3, §2.2 (md + code), §2.4 (md + code), §2.5, §2.6 (md + both
> code cells), §4.4c (one print), §5.6 (md + caveat print), output reference. **Next: run
> interactively up to §2.6, read the calibration grids and the AUTHOR VALIDATION table, then the
> full run.** The thesis text (Ch. 7 Dataset) already reports N = 116 numbers; details in
> `docs/thesis.md` §11–§12.

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

## Pipeline (notebook/thesis_pipeline.ipynb, Kaggle GPU, 104 cells, runs top-to-bottom)
- **Part 1 — ground truth:** parse → **exclude `EXCLUDED_PARTICIPANTS` (§1.1, asserts 116
  participants / 2,320 rows / 60 excerpts)** → per-rater normalization (§1.1b, derived on the
  screened panel) → aggregate/derive labels → validation (§1.2b) → label matrix → rating stats
  (§1.4 prints the 2 exclusions) → lift.
- **Part 2 — features:** MERT (768-dim, layers 5–7; §2.1 restores the positional-conv weight-norm
  tensors and asserts them, cache `emb_mert_v2.npz`) · CREPE (§2.2: tracks cached once
  `crepe_tracks_v2.npz` + a 10 ms RMS envelope `rms10_v3.npz`, octave correction, **B2** plateau
  note segmentation with re-articulation splits, sinusoid-fit vibrato / Viterbi-aware portamento /
  F0 descriptors, MEC negative-control calibration prints; helpers in the
  `# >>> HELPERS:descriptors … # <<<` block) · Essentia
  (dynamics/timbre/tonal; L1-normalised flux, silence-gated dynamic range) · §2.3c recording-level
  sanity check (report only) · timing + DTW (§2.4/§2.4b: `timing_*` reference-free family, `dtw_*`
  deviation family vs MEC). Each extracted in full, then forwarded through one shared
  UNSUPERVISED gate `unsup_forward_select` (never sees Y). madmom/MusiCNN remain skip stubs.
  **§2.6 = TEMPORARY manual descriptor audit** (see below; removable between the AUDIT markers).
- **Part 3 — EDA/validation** (incl. §3.3 CREPE sanity, §3.3b timing sanity + manipulation check).
- **Part 4 — prediction:** **§4.0 = the one CV engine** (protocols, nested model selection =
  default classifier, nested ridge **with optional block weighting**, metrics, chance floors,
  Freedman–Lane + noise-ceiling helpers) · §4.1 NestedSelect (★default) + 8-classifier sensitivity
  table · §4.2 VA regression (+ `Ridge-nested · MERT-full`) · §4.4 + §4.4b/c/d ablation study ·
  §4.5/§4.5b per-class + Dummy floors · §4.6–§4.8 on the default classifier / X_C · §4.9
  within-piece prediction (B-INC Freedman–Lane incremental test, B-NC split-half noise ceiling).
- **Part 5 — attribution:** condition-delta (BH-FDR over all 180 §5.1 tests) · dose-response
  (Page's trend) · SHAP · embedding distance · interpretable linear read-out · cost of
  explainability (both protocols) · §5.6 sub-family ablation.
- **Part 6 — tool characterization:** MERT↔feature correspondence probe (GroupKFold(5) over
  pieces, α nested) · tool complementarity · §6.2b guard.
- **Part 7 — synthesis/save.**
- **Cleaning (outside the notebook):** `questionnaire_cleaning_v3.py` (qc_pack, not in the repo)
  supersedes the participant screening: D1 within-form LOO person-total r < 0.10 · D2 MCD distance
  on form-centred ratings > simulated 99.9th pct (14.55) · D3 scale reversal · D4 mean IRV outside
  [0.25, 2.20] · D5 LongString ≥ 8 in answer order; exclude at ≥ 2 flags → `questionnaire_5_P012`
  (D1+D2), `questionnaire_5_P020` (D1+D3, reversed valence). `scripts/questionnaire_cleaning.py`
  and `scripts/excerpt_cleaning.py` are the original (older) versions. **Decision: the excerpt QC
  (ICC, r_wg, bimodality coefficient, Kruskal–Wallis, crossed additive model) stays OUTSIDE the
  notebook** (offline scripts; numbers in `docs/thesis.md` §11).

## Evaluation protocol (§4.0 — read before quoting any Part 4–6 number)
- **`PRIMARY_PROTOCOL='piece'`** (leave-one-piece-out, `LeaveOneGroupOut` on `PIECE_GROUPS`);
  `loo` is always reported next to it as the optimistic secondary; `PROTOCOLS=('piece','loo')`,
  printed piece first, with the optimism gap. `get_cv(protocol)` is the ONLY splitter builder
  (`'piece5'` = GroupKFold(5) on pieces, used only by the §6.1b probe).
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
  Essentia / DTW): the §4.4b lattice regression heads (via `_abl_reg_kw`) and `X_C_REG` in §4.4d.
  Unchanged: single-block heads, interpretable-only sets, PCA-MERT sets, every classification head,
  §4.2, §4.9, §5.4, §5.5. §4.0 guard on `X_C_REG`: B = 1 exact, Σ p_b·w_b² = P, equal share,
  BlockScaler stateless, weighted operator == weighted sklearn path.
- **B-MERTFULL:** the regression heads see MERT at full dimensionality (`MERT_REG = X_MERT_FULL`);
  the classification heads keep the PCA-forwarded `X_A`.
- `freedman_lane_targets` (B-INC) and `split_half_ceiling` (B-NC) live in the HELPERS block.
- Metrics: macro-F1; r **and** `oos_r2` (baseline = training-fold mean) **and** `rmse`.
- **Chance floors:** Dummy seeded `base + 1000·fold + label`, 200 repeats → mean / SD / P95;
  `CHANCE_FLOOR_P95` = max P95 over strategies (strict "beats chance" line). §4.0 now prints the
  class prevalences of `Y` and the analytic uniform/stratified F1 derived from them (nothing
  hardcoded).

## Key variables
- `df_long` — per-rating, **screened** (§1.1); `EXCLUDED_PARTICIPANTS` (§0.3), `N_PARTICIPANTS_RAW`
  (= 118, §1.1). `PARTICIPANT_DATA = df_long` (§1.4; the old `df_long_clean` branch is gone).
- `df_agg`/`df_all` — per-excerpt; `valence_mean/arousal_mean` = NORMALIZED (ground truth),
  `*_raw` kept in parallel. `emotion_labels` → `Y`; `top_tags` descriptive only.
- `Y` (60×5), `mlb.classes_`. Full blocks: `df_mert`, `df_crepe`, `df_essentia`, `df_dtw`.
  Forwarded: `emb_mert_fwd`, `df_crepe_fwd`, `df_ess_fwd`, `df_dtw_fwd`; lists `*_FORWARD`.
- CREPE: `CREPE_CANDIDATES` (10), `CREPE_LABELS`, `CREPE_DIAGNOSTIC = ['crepe_vibrato_n_notes']`;
  `CREPE_TRACKS`, `CREPE_DIAG`, `CREPE_CENTS`, `CREPE_VOICED`, **`CREPE_ENV_DB`** (10 ms RMS dB on
  the CREPE grid, B2), `CREPE_VITERBI_SOURCE_OK` (report only); `VIB_N_SUBST_NONOTES` /
  `VIB_N_SUBST_NOVIB`; `_med_rate` (the declared median vibrato-rate substitute, read by §2.6).
  `CREPE_DIAG[e]` keys (B2): `n_octave_fixed, n_vib_notes, n_windows, n_vib_windows, n_notes_60,
  n_notes_60_v2, n_reart_splits, n_close_splits, n_transitions, n_glides, glide_durs, glide_exts,
  glide_excess, transitions, vib_windows, notes_by_min` (the Phase B key `n_vib` is gone).
- Diagnostic-only columns: `CREPE_DIAGNOSTIC`, `ESSENTIA_DIAGNOSTIC =
  ['essentia_spectral_flux_raw_mean']`, `TIMING_DIAGNOSTIC = ['timing_octave_jump_frac']`.
- Assembly: `X_A` (MERT-fwd), `X_B` (CREPE-fwd), `X_C` (all), `X_D` (MERT+Essentia), `X_F`
  (MERT+DTW), `X_MERT_FULL`; regression twins `MERT_REG`, `X_C_REG`, `X_D_REG`, `X_F_REG` (stored
  raw — block weighting happens inside the folds). `STRICT_BLOCKS=('MERT','Essentia')`.
- Timing families: `DTW_REFFREE` (`timing_*`, condition-independent) vs `DTW_DEV` (`dtw_*`, 0/1 at
  MEC by construction); `DTW_MODEL_COLS` (→ X_C/X_F, family 1 only, `DTW_DEV_IN_MODELS=False`) vs
  `df_dtw_attr` (→ `df_interp`, both families, `DTW_DEV_IN_INTERP=True`).
- `df_interp` + `INTERP_FEAT_LABELS` (30 features: 10 CREPE + 10 Essentia-fwd + 10 DTW-fwd, §5.1).
- Results: `all_recs`, `DEFAULT_RECS`, `NB_RECS`, `DEFAULT_SET='C: Combined'`, `REG_RES`,
  `PRED_QUADS`, `INTERP_CV_R`, `INTERP_CV_F1`, `COE`, `BLOCK_F1`, `df_within_piece`,
  `df_wp_ceiling` / `WP_CEILING`, `df_delta_tests`, `df_reclevel`, `df_inner_skips`, `df_abl_bw`
  (tier 4E). Ablation rows now also carry `block_weighting` ('equal' / 'none' / 'n/a'),
  `n_alpha_edge_val`, `n_alpha_edge_aro`, `n_outer_folds`.

## Ablation study (§4.4b → §5.6, one engine)
`ablate()` in **§4.4b** is the ONLY scorer; every tier appends to `ABL_ROWS`/`ABL_PRED`;
`df_ablation` → `ablation_master.csv` at the end of §5.6.
- **Protocol:** default classifier (nested selection) → F1-macro; nested piece-grouped ridge on
  `X_reg` (the `MERT_REG`-based matrix whenever MERT is in the set) → r, R², RMSE, **with the
  deployed block weighting on multi-block MERT-full heads** (`ablate(reg_block_sizes=…,
  block_weighting=…)`, built by `_abl_reg_kw(names, weighting=None)`); scaler inside each fold;
  `abl_alpha_edge_report` prints the α grid-edge hits of the weighted heads (never widens the grid).
- **Both CV protocols, `piece` first.** Stats: bootstrap 95% CI · paired permutation p · BH-FDR
  within (tier, protocol); `abl_vs_ref(…, pcols=…)` can restrict the tested metrics (tier 4E =
  regression metrics only). `beats_chance` = F1 > `ABL_FLOOR_P95`.
- **Tiers:** `1-blocks` (15 subsets; unique vs marginal — the unique R² uses the deployed
  weighting) · `3-subfamily` (§5.6) · `4A-*` + `4B-gate` + `4D-balancing` (§4.4c) ·
  **`4E-block-weighting`** (§4.4c: the 7 multi-block MERT-full subsets fitted unweighted and
  weighted; Δ = weighted − unweighted; the deployed member must equal its Tier-1 row) ·
  `4C-neutral-radius` + `4C-ground-truth` (§4.4d; `X_C_REG` rows weighted as deployed).
- **Guards:** §4.0 engine equivalence (GaussianNB) · §4.0 NestedSelect parallel == serial · **§4.0
  block-weighting guard** · §4.4b lattice rebuilds `X_C/X_D/X_F` and `X_C_REG/X_D_REG/X_F_REG`
  (RAW matrices) · §4.4c 4D nested-on-X_A equals the lattice `MERT` row · **§4.4c 4E deployed ==
  lattice** · §4.4d label re-derivation reproduces `Y` · §4.9 closed-form operators equal
  `nested_ridge_predict` · §6.2b additive == lattice · §5.5 interpretable r equals §5.4 · §2.5 no
  pre-Phase-B cache name · **§2.6 audit recompute asserts** (temporary).
- **NOT run:** CREPE step_ms, Essentia frame sizes, per-MERT-layer (**ON HOLD** — decision
  2026-11-03), CREPE voicing threshold / the B2 descriptor constants as an ablation tier (cheap from
  `crepe_tracks_v2.npz` + `rms10_v3.npz`; §2.2 prints report-only sensitivities and the MEC
  calibration grids instead), tempogram window / tempo prior.

## §2.6 — TEMPORARY manual descriptor audit (2026-11-03; updated for B2 on 2026-11-05)
One markdown cell + two code cells between `# >>> AUDIT` / `# <<< AUDIT` markers (cells 39–41; the
Kaggle run regenerated the ids: `617ff63d`, `c214f0e2`, `704bc12e`), right after §2.5.
`RUN_MANUAL_AUDIT = True`, `AUDIT_EXCERPTS = Meditation_Thais_{MEC,EXP,EXG} + Mozart_VC4_{MEC,EXP,EXG}`;
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

## Phase B2 — descriptor repairs after the manual audit (2026-11-05; APPLIED, NOT RUN)
**Audit findings (from the §2.6 outputs of the 2026-11-03 N = 116 run, checked by eye and ear):**
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

## What to check after the next run (N = 116 + block weighting + Phase B2 + §2.6)
First, interactive run up to §2.6 (the §1 items already held in the 2026-11-03 run):
- [ ] §1.1: "2 excluded — questionnaire_5_P012, questionnaire_5_P020"; 118 → 116 participants,
      2360 → 2320 rows; per form F1 20→20 · F2 21→21 · F3 26→26 · F4 19→19 · F5 32→30; asserts pass.
- [ ] §1.1b maps ≈ V 2.94 + 1.57z, A 3.39 + 1.53z · §1.2b anchors V 0.342 → 0.338, A 0.276 → 0.253 ·
      8/60 class changes raw → norm · §1.2 classes HA_LV 20, LA_LV 17, HA_HV 11, Neutral 7, LA_HV 5
      (`Schindlers_List_EXP` HA_LV → LA_LV) · §1.4 prints the exclusions.
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
- [ ] §2.4: timing features recomputed with the B2 onsets (`dtw_features_v3.csv`); DTW forwarding
      count (3–10).
- [ ] §2.6: every recompute / detail-consistency assert passes (windows → coverage / rate / extent,
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
- [ ] §4.0: class prevalences 11/20/5/17/7 and the analytic uniform F1; block-weighting guard ✅ and
      the printed weights (expected MERT-full w ≈ 0.51, CREPE / Essentia ≈ 4.46, DTW ≈ 5.33 for
      795 = 768 + 10 + 10 + 7 columns); parallel == serial ✅; runtime.
- [ ] §4.4b: `[w]` on the 7 MERT-full multi-block rows; do MERT + X rows now differ from MERT alone
      (R²_aro was 0.45–0.46 for all of them at N = 118)? α grid-edge report (warn > 0).
- [ ] §4.4c tier 4E: Δ weighted − unweighted (r, R², q) per set, both protocols; deployed == lattice
      assert ✅; α edge hits.
- [ ] §4.4d: X_C rows weighted; NEUTRAL_RADIUS rule (did r = 1.0 again beat 0.75 on both columns?).
- [ ] Everything else at N = 116: NestedSelect vs strict P95 (piece), §4.9 B-INC q, §5.5 CoE,
      §5.6, §6.1b, the §5.1 BH count (180 tests). All guards ✅. Total runtime (≈ 97 min at N = 118
      + 4E ≈ a few minutes + §2.6).
- [ ] **§5.1b dose-response re-check for the vibrato family** — the thesis abstract claims that
      vibrato extent / coverage escalate MEC → EXP → EXG (Phase B: z +5.14 / +4.90). With the B2
      vibrato (MEC ≈ no vibrato windows by construction of the negative control) re-read the Page z
      of extent, coverage and rate before keeping the claim; also §5.6 Vibrato / Portamento rows
      (Phase B: removing Portamento *raised* F1 — consistent with the false alarms) and §6.1b for the
      vibrato / portamento descriptors (re-check `CREPE_UNIQUE2`).

## Current findings — Phase B run on N = 118 (pre-block-weighting; superseded by the next run)
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
- **§4.0 floors:** piece mean 0.2715 / strict P95 0.3274; loo 0.2720 / 0.3202. 4 CPUs, parallel ==
  serial ✅.
- **§4.1 NestedSelect (piece):** A 0.282, **B 0.361 (only set above the strict P95)**, C 0.302, D 0.281;
  loo A 0.517, B 0.476, C 0.626, D 0.540. NB (piece) A 0.112, B 0.260, C 0.162, D 0.126.
- **§4.2 (piece):** Ridge-nested · MERT-full R²v +0.016, R²a +0.446 (r_a 0.63).
- **§4.4b lattice (piece):** CREPE+Essentia F1 0.389; Essentia+DTW 0.386 with **R²v 0.29, R²a 0.67**
  (best regression); every set containing MERT-full has R²a 0.446–0.459 and R²v ≈ 0.01–0.02 — the
  MERT block dominates the single-α ridge (→ block weighting). Full set F1 0.302 (below P95). Unique
  F1: MERT −0.053, CREPE −0.045, Essentia −0.051, DTW +0.036 (none significant).
- **§4.4c (piece):** 4A-mert-pca R²a PCA −0.25 vs full +0.45 (Δ +0.70, q = 0.0015), R²v −0.17 vs
  +0.02 (q = 0.039), F1 0.28 vs 0.33 (n.s.) · Essentia whole block F1 0.26 vs curated 0.34 · DTW
  ref-free 0.37 / deviation-only 0.27 / both 0.32 · gate 0.80 → 0.385 (n.s.) · **4D:** Interp +
  SVM-Linear balanced 0.417 [0.346, 0.475] (Δ +0.063 vs nested, q = 0.13); NB below nested on all
  three matrices (q ≤ 0.017).
- **§4.4d (piece):** NEUTRAL_RADIUS 1.0 beats 0.75 on BOTH columns for X_A (0.362 / 0.419 vs 0.282 /
  0.353) and X_C (0.374 / 0.383 vs 0.302 / 0.377) — the rule triggers (pending decision). Raw GT
  F1 0.34 vs normalised 0.30.
- **§4.5–§4.8 (X_C, piece):** LA_HV and Neutral collapse (recall 0); per condition MEC 0.24 /
  EXP 0.29 / EXG 0.32; quadrant accuracy 0.25 (fixed Ridge α = 10 on X_A); Krippendorff α HA_HV
  0.31, LA_LV 0.31, others ≤ 0.03.
- **§4.9 within-piece (piece):** ceilings (split-half, centred) V 0.455 [0.088, 0.718], A 0.678
  [0.520, 0.813]. ΔValence: Condition-only R² 0.047; Interpretable 0.174 (p = 0.006); **B-INC
  incremental ΔR² Interpretable +0.129 (q = 0.037), MERT-full +0.128 (q = 0.037), Interp + MERT-PCA
  +0.117 (q = 0.037)**, MERT-PCA +0.077 (q = 0.070). ΔArousal: Condition-only 0.408; Interpretable
  0.423; incremental +0.024 (p = 0.044, q = 0.070); MERT sets do not add. R²/ceiling Interp V 0.38,
  A 0.62.
- **§5.1:** 180 tests, 10 raw p < .05 vs 9.0 expected, **0 q < .05** → not evidence.
- **§5.1b dose-response (Page, 20 pieces):** flux (normalised) z +5.22, **vibrato extent +5.14,
  coverage +4.90**, LUFS +4.74, DR +4.11, DTW deviation +4.59 (built in), rolloff −2.85, **tempo
  log2-SD −2.21 and local tempo −2.06 (both decrease)**; vibrato rate +1.90 n.s.; portamento n.s.
  Audience: arousal z +4.11, valence +1.26 n.s.
- **§5.4 read-out (piece):** V r 0.25 / R² 0.12; A r 0.77 / R² 0.62; class F1 0.335.
- **§5.5 CoE (piece; negative = interpretable wins):** interpretable F1 0.353, R²v 0.120, R²a 0.620 vs
  MERT-PCA 0.282 / −0.171 / −0.252 and MERT-full 0.328 / 0.016 / 0.446. Headline ΔR v −0.181,
  ΔR a −0.140, ΔR² v −0.104, ΔR² a −0.174, ΔF1 −0.071; vs MERT-PCA ΔR²a −0.872.
- **§5.6 (piece):** removing Portamento RAISES F1 to 0.465 (loss −0.112, q = 0.015) and removing Timbre
  to 0.424 (q = 0.009); removing Dynamics costs R²a 0.090 (q = 0.11); Dynamics alone R²a 0.44;
  Vibrato alone F1 0.28, R²a 0.04.
- **§6.1b probe:** flux 0.86, F0 mean 0.80, centroid 0.73, HPCP entropy 0.73, pulse clarity 0.71,
  note rate 0.65, vibrato coverage 0.48; LUFS 0.08; vibrato rate −0.02; α hit the grid edge in **34**
  outer folds.
- **§6.2 (piece):** + all CREPE +0.057, + DTW +0.052, + Essentia −0.001, + 2 CREPE-unique −0.019;
  §6.2b exact. Master ablation table 138 rows / 9 tiers.

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
- [x] §4.0 4 CPUs, parallel == serial ✅. NestedSelect beats the strict P95 under piece only on X_B.
- [x] §4.2 / lattice MERT-full in the regression heads (`n_features_reg` 768–795), rebuild ✅ → found the
      MERT-dominance problem (→ block weighting).
- [x] §4.9 B-INC: Δvalence incremental q = 0.037 (Interpretable, MERT-full); Δarousal q = 0.070.
- [x] §5.1b vibrato extent / coverage now escalate strongly; §6.1b α edge 34 folds.
- [x] Runtime ≈ 97 min.

## Decisions recorded (2026-11-03)
- **Excerpt QC stays OUTSIDE the notebook** (ICC, r_wg, bimodality coefficient, Kruskal–Wallis,
  crossed additive model — offline scripts).
- **Per-MERT-layer caching is ON HOLD.**
- **Timing pattern — the author's explanation:** in EXG he tended to play faster and more
  virtuosically in fast pieces and took more time in slow pieces. This is the likely reason for the
  local-tempo / duration-ratio directions (piece-dependent, no single MEC → EXG direction).
  `timing_tempo_logsd` and pulse clarity are still checked visually in §2.6.
- Block weighting of the multi-block MERT-full regression heads: deployed `'equal'`; tier 4E is
  descriptive.

## Open issues
- **PENDING DECISIONS (unchanged):**
  - `DTW_DEV_IN_INTERP=True` vs the thesis CoE definition (condition-encoding features out of both
    sides of the cost of explainability).
  - `CREPE_UNIQUE2` hard-coded `[f0_range, vibrato_rate]` in §6.2 (read off the pre-Phase-B probe;
    in the N = 118 Phase B probe F0 range is well represented, R² 0.43).
  - The "all-positive" chance floor (≈ 0.322) cited in the thesis but not computed in the notebook.
  - The §4.4d NEUTRAL_RADIUS rule triggers (under piece, r = 1.0 beats 0.75 on both columns for X_A
    and X_C).
  - Fig 5 (§4.7) omits the MEC points (`cc` holds only EXP/EXG) / figure dedup.
  - §4.2 / §4.7 use a fixed Ridge α = 10 while the rest uses nested ridge.
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
- **Not approved / deferred (unchanged):** SHAP fitted in-sample (§5.2) · §5.3 embedding-distance
  sign · §1.4 mixed model · BH in §5.1b · Phase C augmentation (minority classes; pitch ±1 semitone
  or tempo ±5 %, training folds only and only with the source piece in the fold; caveats: no new
  piece diversity, perturbs studied variables, unrated copies, phase-vocoder artefacts).
- "8 emotions" is mechanically 5 classes (report as 5) · the ablation lattice is a DESCRIPTIVE map
  — never select the winning subset on the same CV and then report its score.
- **`LA_HV` has n=5 from only 2 pieces** (La_Vita ×3, Meditation_Thais MEC+EXP; same at N = 116).
- **Session map not provided yet** (`SESSION_MAP_CSV`); gain equality across the 3 recording
  sessions is not independently verified.
- **Parameter ablations still pending:** CREPE (step_ms; voicing threshold and the B2 constants
  cheap from the cached tracks + envelope), Essentia frame sizes, tempogram window / tempo prior.
- §5.5's headline CoE is a best-of over two black boxes (max(MERT-PCA, MERT-full)) — conservative;
  the two separate gaps are saved in `coe_gaps.csv`.
- B2 detectability limits: a glide is only seen between two plateau notes (a slide that never
  settles within ±40 c is part of a transition, not a note); a re-articulation without an envelope
  dip ≥ 6 dB (e.g. a seamless legato bow change) is not split; vibrato on notes < 500 ms or with a
  half-extent < 3 c is invisible; rates outside 4–8 Hz are clamped to the band edge.

## Conventions
- Scale **0–6**, `VA_MID=3.0`, `NEUTRAL_RADIUS=0.75` (un-tuned), `MERT_LAYERS=[5,6,7]`.
- `SEED=42`, `N_DUMMY_REPEATS=200`, `WP_N_PERM=500`, `NC_N_SPLITS=200`, `SELECT_C_GRID=
  logspace(-3,2,6)` → 13 `SELECT_CONFIGS`, `MERT_REG_REPR='full'`, **`REG_BLOCK_WEIGHTING='equal'`**,
  **`EXCLUDED_PARTICIPANTS`**, `SESSION_MAP_CSV`, `OLD_CACHE_NAMES` (§0.3); `RIDGE_ALPHAS=
  logspace(-2,4,13)`, inner `GroupKFold(5)`, `NMS_N_JOBS=os.cpu_count()`, `NMS_TIE_TOL=1e-12` (§4.0);
  `TUNE_GRID_C={'C': logspace(-3,2,6)}` (§4.1); probe `ALPHAS=[1,10,100,1000,10000]` (§6.1b).
- CREPE (§2.2, B2): 10 ms step, `viterbi=True`, voiced = confidence > 0.5; octave fix ±10 frames /
  [1000,1400] c; trend 15 frames; plateaus ±40 c (range ≤ 80 c), merge < 50 c / ≤ 50 ms,
  re-articulation dip 6 dB (150 ms pre / 60 ms post; sensitivity 3/6/9), env 25 ms rectangular on
  the 10 ms grid; notes ≥ 60 ms (timing/portamento) and ≥ `VIB_MIN_NOTE_MS=500` (vibrato;
  500/650/800); vibrato windows 500 ms / hop 250 ms, LS sinusoid 4–8 Hz step 0.05, R² ≥ 0.50,
  half-extent ≥ 3 c; glides rise 80–500 ms, excess over t_min ≥ 50 ms (Viterbi 220 c/frame),
  80–2000 c, sign ≥ 80 %, rise × 1.25. Legacy (report only): boundary lag 5 / 70 c (also the
  close-split threshold), FFT presence ±0.5 Hz / [1,15] Hz ≥ 0.30, half-extent ≥ 5 c.
- Essentia: DR gate `DR_ABS_GATE_DB=-60`, `DR_REL_GATE_DB=40`; flux on L1-normalised spectra.
  §2.3c: `FLAT_MIN=0.2` (0.1/0.2/0.3) over 50 Hz–8 kHz.
- Timing/DTW: `DTW_SR=22050`, `DTW_HOP=512`, `DTW_TG_WIN=151` (≈ 3.5 s), log-normal tempo prior at
  100 BPM within 40–200 BPM, interior frames only (≥ 20), chroma-CQT alignment.
- Excerpt IDs `PieceName_ConditionCode`; participant IDs `questionnaire_{n}_P{idx:03d}`.
- `OUTPUT_DIR = '/kaggle/working/pipeline_outputs'`; caches `emb_mert_v2.npz`,
  `crepe_tracks_v2.npz`, `rms10_v3.npz`, `feat_crepe_v3.csv`, `essentia_features_v2.csv/.npz`,
  `rms_frames_v2.npz`, `dtw_features_v3.csv`; the §2.6 audit writes only to `OUTPUT_DIR/manual_audit/`.
- Forwarding gate is UNSUPERVISED (reads only features, never `Y`); StandardScaler (and the
  BlockScaler) stay inside each CV fold. No cell other than §4.0 builds a CV splitter; no cell
  rebinds a §4.0 or a HELPERS:descriptors name. The `# >>> HELPERS:cv_engine … # <<<` block (§4.0)
  and the `# >>> HELPERS:descriptors … # <<<` block (§2.2) are self-contained.
- **Editing the notebook:** the `.ipynb` (≈ 6 MB, single-line JSON) exceeds the Read tool's notebook
  limit, so NotebookEdit cannot be used on it. Edit cell by cell by splitting the JSON at the
  `{"cell_type"` boundaries, decoding / re-encoding the cell source with sed/awk under `LC_ALL=C`,
  clearing the outputs of changed code cells, and verifying the round trip byte for byte. Claude
  Code cannot run the notebook (no audio/GPU) → surgical edits + static checks only.

## Audit (Oct 2026)
Full audit of the 2026-08-16 run's outputs; corrections in three phases.
- **Phase A — pipeline corrections (applied and run 2026-10-09).** §4.0 CV engine · A1 Dummy floors ·
  A2 leave-one-piece-out PRIMARY + nested ridge + §4.9 · A3 MERT pos-conv restore (false alarm:
  embeddings unchanged) · A4 GaussianNB pre-declared default (failed under piece → B-NMS) · A5
  balancing + inner-CV C (+ tier 4D) · A6 out-of-sample R² / RMSE · A7 group-aware MERT probe · A14
  BH-FDR over all §5.1 tests · A-QC §2.3c. Phase A headline (now historical): NB below chance under
  piece on every set (C 0.168); arousal R² X_C 0.50; 4D Interp + SVM balanced 0.385.
- **Why A2:** 95% (V) / 91% (A) of the between-excerpt variance is between-piece, condition 1% / 4%.
- **Phase B — descriptor repairs + analysis upgrades (applied 2026-10-10, run on N = 118 —
  results above).** B-NMS nested model selection · B-INC Freedman–Lane · B-NC noise ceiling ·
  B-MERTFULL · B-shared CREPE tracks + octave fix + note segmentation · B8 vibrato · B9 portamento ·
  B10 F0 range · B11 normalised flux · B12 DR gate · B13 `timing_*` + 151-frame tempogram ·
  B-NF flatness floor + session map · B-cache `_v2`.
- **Phase B2 — descriptor repairs after the manual audit (applied 2026-11-05, not run)** — see the
  "Phase B2" section above (the table below is Phase A → Phase B; the B2 column changes are there).

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
(§5.1b). Remaining caveats: CREPE's 10 ms resolution, only notes ≥ 500 ms and half-extent ≥ 3 c,
rates clamped to 4–8 Hz, the declared median substitutions (counts printed in §2.2).

## Thesis document (LaTeX, NTUA template) — status 2026-10-30
Written separately from the notebook (folder `thesis_work`, `main.tex`, not in this repo).
Chapters 1–7 written (73 pages of a 120-page cap); Ch. 8 Framework, Ch. 9 Experiments and Ch. 10
Conclusion wait for the N = 116 Phase B run. Full status: `docs/thesis.md` §12 and the planning doc
"Διπλωματική: validation και πλάνο συγγραφής". When pipeline numbers change, the thesis chapters
that quote them are Ch. 7 (Dataset) and, later, Ch. 9.
