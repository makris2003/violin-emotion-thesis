# Violin Expressiveness → Emotion Recognition

Diploma thesis project (School of Electrical and Computer Engineering, NTUA): *Expressive Techniques in Solo Violin Performance and Their Effect on Perceived Emotion*. It follows the chain **expressive technique → acoustic result → perceived emotion** with an interpretable pipeline.

## Overview

The project categorises the **tools** that detect a technique in the signal (CREPE pitch tracking, Essentia DSP, timing/DTW alignment, and the self-supervised MERT representation as one tool category — the opaque reference) and the **modalities** that measure its effect on the audience (valence–arousal ratings, emotion tags). It is an interpretable-attribution study, not a leaderboard.

It asks three questions:

1. Which techniques does the performer escalate across MEC → EXP → EXG, and which of them reach the listeners as valence or arousal?
2. Can perceived emotion be attributed interpretably to technique, and what does that explainability cost relative to the opaque representation?
3. Where do the interpretable features and the SSL representation agree or complement each other?

## Dataset

- **60 audio excerpts** — 20 pieces × 3 performance conditions:
  - `MEC` — mechanical (neutral baseline)
  - `EXP` — expressive (natural)
  - `EXG` — exaggerated (heightened expression)
- **118 responses** across 5 linked online forms (**116 after participant screening**) rated each excerpt on **valence (0–6)**, **arousal (0–6)**, and tagged 1–3 perceived emotions (Tenderness, Nostalgia, Peacefulness, Power, Joyful Activation, Tension, Sadness, Neutral).
- Ground truth = **5 classes** (the 4 valence–arousal circumplex quadrants + Neutral) from rater-normalised means. Leave-one-piece-out is the primary evaluation protocol; LOOCV is reported as the optimistic secondary.

*Audio and questionnaire data are not included in the repository.*

## Repository Structure

```
violin-emotion-thesis/
├── notebook/
│   └── thesis_pipeline.ipynb      # main analysis pipeline (feature extraction → modelling → analysis)
├── scripts/
│   ├── questionnaire_cleaning.py  # participant-level response screening
│   └── excerpt_cleaning.py        # excerpt/stimulus quality checks
├── docs/
│   └── thesis.md                  # detailed project notes (status, findings, open issues)
├── CLAUDE.md                      # project context
├── .gitignore
└── README.md
```

## Methods

**Feature extraction.** Each excerpt is encoded with MERT (embeddings averaged over transformer layers 5–7), CREPE-derived pitch descriptors (vibrato, portamento, F0), Essentia DSP descriptors (dynamics, timbre, tonal) and timing/DTW descriptors, each reduced by one shared unsupervised forwarding gate.

**Prediction.** 5-class classification with nested model selection and valence/arousal regression with nested ridge, under leave-one-piece-out (primary) and LOOCV (secondary), against re-seeded chance floors; within-piece prediction with permutation tests and a noise ceiling; a block-lattice ablation study.

**Attribution.** Condition deltas, dose-response (Page's trend), SHAP, an interpretable linear read-out and the cost of explainability against MERT.

**Participant & excerpt screening.** The participant screening that sets the exclusions (v3) and the excerpt QC run outside the notebook. The two scripts in `scripts/` are the original (older) versions.

## Setup

The pipeline runs on **Kaggle with a GPU**. torch is pinned to 2.3.1 for the P100 (sm_60); the recorded runs used a Tesla T4.

```bash
# core dependencies
pip install torch==2.3.1        # pinned for P100 compatibility
pip install transformers crepe
pip install scikit-learn shap pandas numpy scipy matplotlib
```

## Usage

**Run the main pipeline:** open `notebook/thesis_pipeline.ipynb` on Kaggle with a GPU enabled and run all cells top to bottom. It handles feature extraction, modelling, and analysis, and saves all figures and result tables.

**Run the screening scripts:**

```bash
python scripts/questionnaire_cleaning.py   # participant screening report
python scripts/excerpt_cleaning.py         # excerpt QC report
```

Each script writes a summary and CSV reports to its output directory.

## Development

The project is version-controlled on GitHub and executed on Kaggle (linked via Kaggle's GitHub integration). Code is edited locally and pushed to GitHub; Kaggle pulls the latest for GPU runs.

---

*Diploma thesis (NTUA, School of ECE) — music information retrieval / affective computing.*
