# Violin Emotion Thesis — Pipeline Overview

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

## Pipeline (notebook/thesis_pipeline.ipynb, Kaggle P100, 97 cells, runs top-to-bottom)
- **Part 1 — ground truth:** parse → **per-rater normalization (§1.1b, new ground truth)** →
  aggregate/derive labels → validation (§1.2b) → label matrix → rating stats → lift.
- **Part 2 — features:** MERT (768-dim, layers 5–7) · CREPE (pitch/vibrato/portamento) ·
  Essentia (dynamics/timbre/tonal) · **DTW (timing/rubato, §2.4/§2.4b — ACTIVE)**. Each extracted
  in full, then **forwarded through one shared UNSUPERVISED gate** `unsup_forward_select`
  (never sees Y → leakage-free). madmom/MusiCNN remain skip stubs at §2.4.
- **Part 3 — EDA/validation** (incl. §3.3b DTW timing sanity + manipulation check).
- **Part 4 — prediction:** LOOCV multi-label classification + VA regression over feature sets
  A/B/C/D (X_F is built in §2.5 but tested in §6.2, not §4.1); DummyClassifier floors (§4.5b)
  are the official chance baseline (MFCC baseline removed). **§4.4b/c/d = the systematic
  ablation study** (see below).
- **Part 5 — attribution:** condition-delta · dose-response (Page's trend) · SHAP · embedding
  distance · interpretable linear read-out · cost of explainability · **§5.6 sub-family
  ablation** (decomposes the CoE by expressive family).
- **Part 6 — tool characterization:** MERT↔feature correspondence probe · tool complementarity.
- **Part 7 — synthesis/save.**
- **Cleaning (separate scripts):** questionnaire_cleaning.py (participant screening, MCD/
  Mahalanobis etc.), excerpt_cleaning.py (stimulus QC, ICC, manipulation check). Exclusions
  NOT yet applied (pipeline runs on all 118).

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

## Ablation study (§4.4b → §5.6, one engine)
`ablate()` in **§4.4b** is the ONLY scorer; every tier appends to `ABL_ROWS`/`ABL_PRED` and the
consolidated `df_ablation` → `ablation_master.csv` is written at the end of §5.6.
- **Protocol** (identical to §5.5/§6.2): SVM-Linear → F1-macro, RidgeCV → r_val/r_aro, scaler
  inside the CV pipeline. Reads `Y` only to SCORE, never to select — forwarding stays unsupervised.
- **Two CV protocols, both always reported:** `loo` (Part-4 protocol, OPTIMISTIC — the same
  piece's other two renditions stay in training) and `piece` (`LeaveOneGroupOut` on `piece` =
  leave-one-piece-out, the honest unseen-material number). Quote `piece` for generalisation claims.
- **Stats:** bootstrap 95% CI over excerpts · **paired permutation** p (predictions swapped per
  excerpt, metric recomputed — valid for F1 and r) · **BH-FDR within (tier, protocol)**. Runs with
  different ground truth are `paired=False` and get NaN p by design (§4.4d).
- **Tiers:** `1-blocks` (all 15 subsets of MERT/CREPE/Essentia/DTW → unique vs marginal
  contribution, §4.4b) · `3-subfamily` (leave-one-family-out over Pitch/Vibrato/Portamento/
  Dynamics/Timbre/Tonal/Timing-reffree/Timing-dev, §5.6) · `4A-*` + `4B-gate` (config knobs,
  §4.4c) · `4C-neutral-radius` + `4C-ground-truth` (target side, §4.4d).
- **Guards:** §4.4b asserts the lattice rebuilds `X_C/X_D/X_F` exactly; §4.4d asserts the label
  re-derivation reproduces `Y` at the deployed radius; **§6.2b** asserts §6.2's additive numbers
  equal the §4.4b lattice rows. Gate sweeps write audits to `outputs/ablation_gate_audit/` so the
  §2.3b/§2.4b audit CSVs are never overwritten.
- **NOT run (needs re-extraction from audio):** CREPE step_ms/voicing/vibrato band, Essentia
  frame sizes, DTW tempogram window, **per-MERT-layer** (§2.1 averages layers 5–7 — have it cache
  per-layer once and future layer ablations become free).

## Current findings (short)
Arousal interpretable/predictable (r≈0.82 interpretable, ~0.86 MERT-full; **cost of explainability
+0.037**); valence resists (**CoE +0.300**, intrinsic to MER). Expressive escalation carried by
**dynamics & timbre, not vibrato** (dose-response: flux/loudness rise strongly, vibrato n.s.;
audience arousal rises, valence n.s.). Normalization is a defensibility step (corr raw↔norm 0.997,
0/60 quadrant flips) — confirms signal, doesn't change results.

## Conventions
- Scale **0–6**, `VA_MID=3.0`, `NEUTRAL_RADIUS=0.75` (un-tuned), `MERT_LAYERS=[5,6,7]`.
- DTW: `DTW_SR=22050`, `DTW_HOP=512`, `DTW_TG_WIN=384`, BPM band 30–300, chroma-CQT alignment.
- Excerpt IDs: `PieceName_ConditionCode` (e.g. `Bach_Adagio_S1_MEC`).
- Participant IDs: `questionnaire_{n}_P{idx:03d}` (alias `S{n}_P{idx:03d}`).
- `OUTPUT_DIR = '/kaggle/working/outputs'`.
- Forwarding gate is UNSUPERVISED (reads only features, never `Y`); StandardScaler stays inside
  each CV pipeline.
- Claude Code cannot run the notebook (no audio/GPU) → surgical edits + static checks only.

## Open issues
"8 emotions" is mechanically 5 classes (report as 5) · figure dedup partially pending ·
exclusions not applied · nested selection (the ablation lattice is a DESCRIPTIVE map — never
select the winning subset on the same CV and then report its score) · feature hygiene (trust
vibrato depth over rate) · **DTW numbers not yet run** (code is in, all values come from the next
Kaggle run) · DTW may forward many columns at N=60 — watch the §2.4b "outside 3–10" warning and
trim `DTW_CANDIDATES` if so · **all ablation numbers also pending the next Kaggle run** (code
smoke-tested on synthetic data only).
Addressed by the ablation study: Neutral radius sensitivity (§4.4d, 0.5/0.75/1.0 with macro-F1
reported **with and without** Neutral) · FDR (BH within tier×protocol) · the LOOCV piece-leakage,
now quantified as the `loo` vs `piece` optimism gap · portamento per-second normalisation (§4.4c,
needs audio for durations, skips loudly otherwise).

## Vibrato descriptor — read before quoting any vibrato result
`crepe_vibrato_rate_hz` is the argmax of a spectrum **inside the same [4,8] Hz band the signal was
already bandpassed to**, so it can never report "no vibrato"; both vibrato features run over
**concatenated non-contiguous voiced frames** (each voicing gap injects a discontinuity into that
band); depth conflates **extent** with **coverage**. A vibrato null (dose-response n.s., or a
§5.6 ablation Δ≈0) is therefore evidence about the **descriptor**, belonging to the tool taxonomy —
NOT the musical claim "vibrato does nothing". Making it a musical claim requires repairing the
descriptor first (time-contiguous segments, extent/coverage split, rate estimated on a wider band
before the bandpass).