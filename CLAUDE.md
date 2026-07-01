# Violin Emotion Thesis — Pipeline Overview

## Project
Master's thesis: association of violin performance techniques 
with perceived emotions using ML on audio features.

## Dataset
60 audio excerpts: 20 pieces × 3 conditions (MEC/EXP/EXG).
111 participants across 5 sessions rated valence (1-6), 
arousal (1-6), and emotion tags per excerpt.

## Pipeline
Notebook: thesis-pipeline.ipynb (Kaggle, P100 GPU)
- Feature extraction: MERT (768-dim SSL) + CREPE (8-dim pitch)
- Classification: 8 classifiers, LOOCV, 8 emotion labels (multi-label)
- Analysis: SHAP technique importance + Condition-Delta correlations
- Cleaning: questionnaire_cleaning.py (Mahalanobis Distance, MCD)

## Key variables
df_crepe: (60,8) CREPE pitch features indexed by excerpt_id
emb_mert: dict {excerpt_id: np.array(768)}
Y: (60,8) binary label matrix
df_delta: piece-level technique/VA deltas for 20 pieces
df_shap_importance: (8 emotions × 8 techniques) mean |SHAP|

## Current status
Validation mode: MERT + CREPE only.
Madmom / Essentia / MusiCNN / DTW cells are skip stubs.

## Conventions
OUTPUT_DIR = '/kaggle/working/outputs'
Excerpt IDs format: PieceName_ConditionCode (e.g. Bach_Adagio_S1_MEC)