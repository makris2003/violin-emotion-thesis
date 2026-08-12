# Violin Emotion Thesis — Pipeline Overview

## Project
Master's thesis: **expressive violin techniques → acoustic result → perceived audience emotion**,
studied with an **interpretable** ML pipeline. Two categorisation deliverables (tools for detecting
technique; modalities for measuring audience effect). Framing is **categorisation + interpretable
attribution**, NOT a leaderboard — MERT is one tool category (the deep/SSL exemplar).

## Dataset
60 excerpts = 20 pieces × 3 conditions (MEC mechanical / EXP expressive / EXG exaggerated),
recorded by the author playing violin. 118 participants across 5 **linked (ring)** forms rated
**valence 0–6, arousal 0–6** (midpoint 3.0; `0` is a real rating, not NA) + emotion tags.
Ring: 20/form, 8 anchors shared with each neighbour, 4 unique, 40 anchors total.
Ground truth = **5 classes** (4 circumplex quadrants + Neutral) from **normalized** VA means.

## Pipeline (thesis-pipeline.ipynb, Kaggle P100, ~86 cells, runs top-to-bottom)
- **Part 1 — ground truth:** parse → **per-rater normalization (§1.1b, new ground truth)** →
  aggregate/derive labels → validation (§1.2b) → label matrix → rating stats → lift.
- **Part 2 — features:** MERT (768-dim, layers 5–7) · CREPE (pitch/vibrato/portamento) ·
  Essentia (dynamics/timbre/tonal). Each extracted in full, then **forwarded through one shared
  UNSUPERVISED gate** `unsup_forward_select` (never sees Y → leakage-free). madmom/DTW/MusiCNN
  are skip stubs at §2.4 (DTW being reactivated).
- **Part 3 — EDA/validation.**
- **Part 4 — prediction:** LOOCV multi-label classification + VA regression over feature sets
  A/B/C/D; DummyClassifier floors (§4.5b) are the official chance baseline (MFCC baseline removed).
- **Part 5 — attribution:** condition-delta · dose-response (Page's trend) · SHAP · embedding
  distance · interpretable linear read-out · cost of explainability.
- **Part 6 — tool characterization:** MERT↔feature correspondence probe · tool complementarity.
- **Part 7 — synthesis/save.**
- **Cleaning (separate scripts):** questionnaire_cleaning.py (participant screening, MCD/
  Mahalanobis etc.), excerpt_cleaning.py (stimulus QC, ICC, manipulation check). Exclusions
  NOT yet applied (pipeline runs on all 118).

## Key variables
- `df_long` — per-rating (+ `valence_norm/arousal_norm` after §1.1b).
- `df_agg`/`df_all` — per-excerpt; `valence_mean/arousal_mean` = NORMALIZED (ground truth),
  `*_raw` kept in parallel. `emotion_labels` → `Y`; `top_tags` descriptive only.
- `Y` (60×N), `mlb.classes_`. Full blocks: `df_mert`, `df_crepe`, `df_essentia`.
  Forwarded: `emb_mert_fwd`, `df_crepe_fwd`, `df_ess_fwd`; lists `*_FORWARD`.
- Assembly: `X_A` (MERT-fwd), `X_B` (CREPE-fwd), `X_C` (MERT+CREPE+Essentia), `X_D` (MERT+Essentia),
  `X_MERT_FULL` (768-dim, kept aside for representational analyses, NOT a model input).
  `STRICT_BLOCKS=('MERT','Essentia')` → missing block raises, never silent zero-fill.
- Interpretable frame + labels for Part 5: `df_interp` + `INTERP_FEAT_LABELS`
  (CREPE + Essentia; **add DTW here when reactivated**).

## Current findings (short)
Arousal interpretable/predictable (r≈0.82 interpretable, ~0.86 MERT-full; **cost of explainability
+0.037**); valence resists (**CoE +0.300**, intrinsic to MER). Expressive escalation carried by
**dynamics & timbre, not vibrato** (dose-response: flux/loudness rise strongly, vibrato n.s.;
audience arousal rises, valence n.s.). Normalization is a defensibility step (corr raw↔norm 0.997,
0/60 quadrant flips) — confirms signal, doesn't change results.

## Conventions
- Scale **0–6**, `VA_MID=3.0`, `NEUTRAL_RADIUS=0.75` (un-tuned), `MERT_LAYERS=[5,6,7]`.
- Excerpt IDs: `PieceName_ConditionCode` (e.g. `Bach_Adagio_S1_MEC`).
- Participant IDs: `questionnaire_{n}_P{idx:03d}` (alias `S{n}_P{idx:03d}`).
- `OUTPUT_DIR = '/kaggle/working/outputs'`.
- Forwarding gate is UNSUPERVISED (reads only features, never `Y`); StandardScaler stays inside
  each CV pipeline.
- Claude Code cannot run the notebook (no audio/GPU) → surgical edits + static checks only.

## Open issues
"8 emotions" is mechanically 5 classes (report as 5) · Neutral radius un-tuned · figure dedup
partially pending · DTW/timing not yet added · exclusions not applied · stats hygiene (FDR / nested
selection) · feature hygiene (normalise portamento count; trust vibrato depth over rate).