# Violin Emotion Thesis — Pipeline Overview

> **Revision status: Phase A applied — results pending the next Kaggle run.** Every Part 4/5/6
> predictive number from before Phase A is superseded (see "Current findings").

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
  weight-norm tensors from the checkpoint and asserts them**, cache `emb_mert_v2.npz`) · CREPE
  (pitch/vibrato/portamento) · Essentia (dynamics/timbre/tonal) · **§2.3c recording-level sanity
  check** (report only: LUFS robust z, within-piece LUFS, noise-floor chain test, clipping) ·
  **DTW (timing/rubato, §2.4/§2.4b)**. Each extracted in full, then **forwarded through one shared
  UNSUPERVISED gate** `unsup_forward_select` (never sees Y). madmom/MusiCNN remain skip stubs.
- **Part 3 — EDA/validation** (incl. §3.3b DTW timing sanity + manipulation check).
- **Part 4 — prediction:** **§4.0 = the one CV engine** (protocols, default classifier, nested
  ridge, metrics, chance floors) · §4.1 8-classifier **sensitivity table** (never used for
  selection) · §4.2 VA regression · §4.4 + **§4.4b/c/d ablation study** · §4.5/§4.5b per-class +
  re-seeded Dummy floors · §4.6–§4.8 on the default classifier / X_C · **§4.9 within-piece
  prediction** (piece-centred targets & features, piece CV only, within-piece permutation).
- **Part 5 — attribution:** condition-delta (**BH-FDR over all §5.1 tests**) · dose-response
  (Page's trend) · SHAP · embedding distance · interpretable linear read-out · cost of
  explainability (both protocols; headline + two separate gaps) · **§5.6 sub-family ablation**.
- **Part 6 — tool characterization:** MERT↔feature correspondence probe (**GroupKFold(5) over
  pieces, α nested**) · tool complementarity (default classifier, both protocols) · §6.2b guard.
- **Part 7 — synthesis/save** (default classifier F1 per protocol, chance mean/P95, mean F1 of the
  sensitivity classifiers).
- **Cleaning (separate scripts):** questionnaire_cleaning.py (participant screening, MCD/
  Mahalanobis etc.), excerpt_cleaning.py (stimulus QC, ICC, manipulation check). Exclusions
  NOT yet applied (pipeline runs on all 118).

## Evaluation protocol (§4.0 — read before quoting any Part 4–6 number)
- **`PRIMARY_PROTOCOL='piece'`** (leave-one-piece-out, `LeaveOneGroupOut` on `PIECE_GROUPS`);
  `loo` is always reported next to it as the optimistic secondary; `PROTOCOLS=('piece','loo')`,
  printed piece first, with the optimism gap. `get_cv(protocol)` is the ONLY splitter builder
  (`'piece5'` = GroupKFold(5) on pieces, used only by the §6.1b probe).
- **Default classifier (pre-declared, A4):** `DEFAULT_CLF_NAME='NaiveBayes'`,
  `make_default_clf() → GaussianNB()`. Downsides stated in §4.0 (independence violated across
  blocks, no shrinkage, zero-inflated features, uncalibrated probabilities — never interpret
  `predict_proba`; nothing in the notebook does).
- `cv_predict_multilabel(make_est, X, Y, protocol, tune_grid=None)` — manual outer loop, binary
  relevance, `SafeMultiOutputClassifier` semantics; with `tune_grid`, C is chosen per label
  inside the training fold by inner GroupKFold(5) on pieces (pooled OOF F1; ties → strongest
  regularisation; skipped → C=1.0 if <5 training positives or positives in <2 pieces).
  Asserted equal to the old `cross_val_predict(SafeMultiOutput(GaussianNB))` on X_A.
- `nested_ridge_predict(X, y, protocol)` — α ∈ `RIDGE_ALPHAS=logspace(-2,4,13)` chosen inside each
  outer training fold by inner GroupKFold(5) on pieces (pooled SSE; ties → largest α); replaces
  every `RidgeCV`. `NestedRidgeOperator` = its closed-form twin (used only for §4.9's permutation
  refits; equivalence asserted on the observed targets).
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
- Assembly: `X_A` (MERT-fwd), `X_B` (CREPE-fwd), `X_C` (MERT+CREPE+Essentia+DTW),
  `X_D` (MERT+Essentia), `X_F` (MERT+DTW — the complementarity contrast in §6.2),
  `X_MERT_FULL` (768-dim, kept aside for representational analyses, NOT a model input).
  `STRICT_BLOCKS=('MERT','Essentia')` → missing block raises, never silent zero-fill; DTW is
  non-strict but a missing row still warns loudly in `bfv`.
- **DTW has two families** (§2.4): (1) `DTW_REFFREE` — reference-free timing/rubato, graded for
  all 60 incl. MEC, condition-INdependent; (2) `DTW_DEV` — within-piece warp-path deviation vs
  the piece's own MEC, which is the self-alignment identity (0 deviation / 1 ratio) at MEC **by
  construction** and so partly encodes condition. §2.4b splits them: `DTW_MODEL_COLS` (→ X_C/X_F,
  family 1 only by default, `DTW_DEV_IN_MODELS=False`) vs `df_dtw_attr` (→ `df_interp`, both
  families, `DTW_DEV_IN_INTERP=True`). Flip either constant to change the regime.
- Interpretable frame + labels for Part 5: `df_interp` + `INTERP_FEAT_LABELS`
  (CREPE + Essentia + DTW, built in §5.1; §5.1b/§5.2/§5.4/§5.5/§5.6 all derive from it).
- Phase-A results containers: `all_recs` (8 clf × 4 sets × 2 protocols, `protocol` key),
  `DEFAULT_RECS[(set, protocol)]`, `DEFAULT_SET='C: Combined'`, `REG_RES[protocol]`
  (`reg_res` = primary), `PRED_QUADS`, `INTERP_CV_R[protocol][dim]`, `INTERP_CV_F1[protocol]`,
  `COE[protocol]`, `BLOCK_F1[protocol]`, `df_within_piece`, `df_delta_tests` (+ `q_ME/EX/MG`),
  `df_reclevel`, `df_inner_skips`.

## Ablation study (§4.4b → §5.6, one engine)
`ablate()` in **§4.4b** is the ONLY scorer; every tier appends to `ABL_ROWS`/`ABL_PRED` and the
consolidated `df_ablation` → `ablation_master.csv` is written at the end of §5.6.
- **Protocol** (identical to §5.5/§6.2): default classifier (GaussianNB) → F1-macro; nested
  piece-grouped ridge → r, `r2_val/r2_aro`, `rmse_val/rmse_aro`; scaler inside each fold. Reads
  `Y` only to SCORE, never to select — forwarding stays unsupervised.
- **Both CV protocols, `piece` first** (primary) — quote `piece` for generalisation claims.
- **Stats:** bootstrap 95% CI over excerpts · **paired permutation** p (predictions swapped per
  excerpt, metric recomputed — valid for F1, r and R²) · **BH-FDR within (tier, protocol)** over
  F1, r_val, r_aro, R²_val, R²_aro. Runs with different ground truth are `paired=False` → NaN p
  (§4.4d). `beats_chance` = F1 > `ABL_FLOOR_P95`.
- **Tiers:** `1-blocks` (all 15 subsets of MERT/CREPE/Essentia/DTW → unique vs marginal
  contribution, §4.4b) · `3-subfamily` (leave-one-family-out, §5.6) · `4A-*` + `4B-gate` (config
  knobs, §4.4c) · **`4D-balancing`** (§4.4c: NB empirical priors (ref) / NB balanced priors /
  SVM-Linear C=1 unbalanced / SVM-Linear balanced + inner-CV C, on X_A, X_C, interpretable;
  classification only; per-matrix reference via `abl_vs_ref(..., ref_map=)`; descriptive — the
  default does NOT change based on it) · `4C-neutral-radius` + `4C-ground-truth` (§4.4d).
- **Guards:** §4.0 engine equivalence (GaussianNB) · §4.4b lattice rebuilds `X_C/X_D/X_F` · §4.4c
  4D NB-on-X_A equals the lattice `MERT` row · §4.4d label re-derivation reproduces `Y` · §4.9
  closed-form operator equals `nested_ridge_predict` · **§6.2b** §6.2's additive numbers equal the
  §4.4b lattice rows (both protocols) · §5.5 interpretable r equals §5.4 exactly. Gate sweeps write
  audits to `OUTPUT_DIR/ablation_gate_audit/`.
- **NOT run (needs re-extraction from audio):** CREPE step_ms/voicing/vibrato band, Essentia
  frame sizes, DTW tempogram window, **per-MERT-layer** (§2.1 averages layers 5–7 — have it cache
  per-layer once and future layer ablations become free).

## Current findings (short) — Kaggle run of 2026-08-16 (DTW + full ablation executed)
**Part 4/5/6 predictive numbers below are PRE-AUDIT, superseded by the Phase A run** (MERT
re-extraction with the pos-conv fix, new default classifier, piece-wise primary protocol,
nested ridge). No Phase-A number exists yet.
- Cost of explainability (§5.5, LOOCV, pre-audit): **ΔR valence +0.112, ΔR arousal +0.051,
  ΔF1 +0.145** (the older +0.300 / +0.037 are obsolete).
- Expressive escalation carried by **dynamics & timbre, not vibrato** (dose-response §5.1b:
  flux/loudness rise strongly, vibrato n.s.; audience arousal rises, valence n.s.). Phase A does
  not touch §5.1/§5.1b inputs; Phase B (descriptor repair) will.
- Normalization (§1.2b): raw↔norm Pearson r = **0.996 (V) / 0.993 (A)**; **10/60 excerpts change
  class, 4/60 change quadrant** (the old "0/60 flips" claim was wrong). Conclusion unchanged:
  a defensibility step, no material change in results.
- **Variance structure:** 95% of valence and 91% of arousal variance across the 60 excerpts is
  between-piece; condition explains 1% (V) / 4% (A). LOOCV therefore mostly measures piece
  recognition.

## What to check after the Phase A Kaggle run
- [ ] §2.1 prints `loading info: missing=[…original0, …original1] unexpected=[…weight_g, …weight_v]`
      followed by `✅ MERT pos_conv weight-norm parameters restored from checkpoint` (no RuntimeError).
      NOTE: transformers' own "not used" / "newly initialized … pos_conv_embed" warnings are printed
      INSIDE `from_pretrained`, i.e. before the restore, so they still appear — the restore message
      and the equality assert are the evidence that the weights are now the checkpoint's.
- [ ] §2.3c: any recording-level flags (level outlier / within-piece / chain change / clipping),
      per-condition LUFS vs floor.
- [ ] §4.0: the GaussianNB equivalence assert passes; Dummy floors vary across folds.
- [ ] §4.5b: LOOCV uniform floor ≈ 0.27, stratified ≈ 0.20; which default-classifier rows beat the
      strict P95 under `piece`.
- [ ] §4.4b lattice assert, §4.4d label-reproduction assert, §6.2b guard (both protocols) all pass.
- [ ] Tier `4D-balancing`: does balancing help under `piece`? (descriptive — default stays NB).
- [ ] §4.9 within-piece: does any feature set beat Condition-only? (expectation to verify:
      arousal may, valence probably not).
- [ ] §5.5 CoE sign under `piece` (headline and the two separate gaps).
- [ ] §4.1 `df_inner_skips`: how often inner-CV tuning was skipped (A5 edge case; LA_HV expected).
- [ ] §6.1b: how often α hits the new grid edge (1e4).
- [ ] Runtime: Phase A adds roughly 20–35 min (see the Phase A report).

## Conventions
- Scale **0–6**, `VA_MID=3.0`, `NEUTRAL_RADIUS=0.75` (un-tuned), `MERT_LAYERS=[5,6,7]`.
- `SEED=42`, `N_DUMMY_REPEATS=200`, `WP_N_PERM=500` (§0.3); `RIDGE_ALPHAS=logspace(-2,4,13)`,
  inner `GroupKFold(5)` (§4.0); `TUNE_GRID_C={'C': logspace(-3,2,6)}` (§4.1); probe
  `ALPHAS=[1,10,100,1000,10000]` (§6.1b).
- DTW: `DTW_SR=22050`, `DTW_HOP=512`, `DTW_TG_WIN=384`, BPM band 30–300, chroma-CQT alignment.
- Excerpt IDs: `PieceName_ConditionCode` (e.g. `Bach_Adagio_S1_MEC`).
- Participant IDs: `questionnaire_{n}_P{idx:03d}` (alias `S{n}_P{idx:03d}`).
- `OUTPUT_DIR = '/kaggle/working/pipeline_outputs'`; MERT cache `emb_mert_v2.npz`.
- Forwarding gate is UNSUPERVISED (reads only features, never `Y`); StandardScaler stays inside
  each CV fold. No cell other than §4.0 builds a CV splitter; no cell rebinds a §4.0 name.
- Claude Code cannot run the notebook (no audio/GPU) → surgical edits + static checks + local
  synthetic tests only. The `# >>> HELPERS:cv_engine … # <<<` block in §4.0 is self-contained so
  it can be extracted and tested outside the notebook.

## Open issues
- "8 emotions" is mechanically 5 classes (report as 5) · figure dedup partially pending ·
  exclusions not applied · nested selection (the ablation lattice is a DESCRIPTIVE map — never
  select the winning subset on the same CV and then report its score) · feature hygiene (trust
  vibrato depth over rate) · DTW forwards 11 columns at N=60 (8 ref-free + 3 dev; the §2.4b
  "outside 3–10" warning fired in the 2026-08-16 run) — trim `DTW_CANDIDATES` if that matters.
- **`LA_HV` has n=5 from only 2 pieces** (La_Vita ×3, Meditation_Thais MEC+EXP) — under `piece`
  it is learned from one piece at a time; inner-CV tuning is skipped for it in many folds.
- **Phase B (deferred) — descriptors measure something else than their name:** vibrato
  descriptor, portamento detector, F0 range, spectral-flux normalisation, dynamic-range silence
  gate, timing onsets/tempogram.
- **Phase C (deferred idea) — augmentation:** supervisor's suggestion to augment the minority
  positive-valence classes (pitch shift ±1 semitone, tempo ±5%, never combined); copies used
  **only in training folds, and only when their source piece is in that training fold** (test =
  originals only; gate/PCA fitted on originals; scoring/permutations on the 60 originals).
  Caveats: cannot add piece diversity (LA_HV comes from 2 pieces); perturbs studied variables (F0
  mean, tempo, onset rate) while keeping labels; copies' labels were never rated; phase-vocoder
  artefacts affect Essentia timbre. Decision: evaluate class balancing (A5, tier 4D) first;
  augmentation later as an ablation tier.
- **Not approved yet:** SHAP fitted in-sample (§5.2) · §5.3 embedding-distance sign · §1.4 mixed
  model · BH in §5.1b.
- **Known, not fixed (out of scope):** Fig 5 VA scatter (§4.7) never draws MEC points (`cc` holds
  only EXP/EXG); README points to `docs/thesis_handoff.md`, the file is `docs/thesis.md`.
- §5.5's headline CoE is a best-of over two black boxes (max(MERT-PCA, MERT-full)) —
  conservative; the two separate gaps are saved in `coe_gaps.csv`.

## Audit (Oct 2026)
Full audit of the 2026-08-16 run's outputs; corrections run in three phases.
- **Phase A — pipeline corrections (APPLIED, results pending the next Kaggle run).** A-shared
  §4.0 CV engine · A1 DummyClassifier floors (re-seeded per fold × label, 200 repeats — were
  collapsing to F1=0) · A2 leave-one-piece-out as PRIMARY protocol everywhere + nested
  piece-grouped ridge + §4.9 within-piece prediction + CoE handles a negative gap · A3 MERT
  positional-conv weights were randomly initialised (checkpoint keys `weight_g/_v` not loaded) →
  restored + asserted, cache v2 · A4 GaussianNB as the single pre-declared default classifier, all
  best-of selection removed · A5 class balancing + inner-CV C for the sensitivity classifiers
  (+ tier `4D-balancing`) · A6 out-of-sample R² and RMSE next to r (+ R² in the ablation BH
  family) · A7 group-aware MERT probe with nested α · A14 BH-FDR over all §5.1 tests (stars from
  q; `analysis_condition_delta_tests.csv`) · A-QC recording-level sanity check (§2.3c).
- **Why A2:** 95% (V) / 91% (A) of the between-excerpt variance is between-piece, condition 1% /
  4% → LOOCV mostly measures piece recognition.
- **Phase B** and **Phase C** — see Open issues.

## Vibrato descriptor — read before quoting any vibrato result
`crepe_vibrato_rate_hz` is the argmax of a spectrum **inside the same [4,8] Hz band the signal was
already bandpassed to**, so it can never report "no vibrato"; both vibrato features run over
**concatenated non-contiguous voiced frames** (each voicing gap injects a discontinuity into that
band); depth conflates **extent** with **coverage**. A vibrato null (dose-response n.s., or a
§5.6 ablation Δ≈0) is therefore evidence about the **descriptor**, belonging to the tool taxonomy —
NOT the musical claim "vibrato does nothing". Making it a musical claim requires repairing the
descriptor first (time-contiguous segments, extent/coverage split, rate estimated on a wider band
before the bandpass).
