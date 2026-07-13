# Violin Expressiveness → Emotion Recognition

A thesis project investigating how expressive performance techniques in solo violin shape the emotions listeners perceive, and how well modern audio representations can predict those emotions.

## Overview

The project compares two families of audio features for perceived-emotion prediction:

- **MERT** — a self-supervised music representation model (768-dim embeddings)
- **CREPE** — interpretable low-level pitch features (vibrato, portamento, F0 statistics, voiced fraction)

It asks three questions:

1. Do self-supervised embeddings outperform hand-crafted acoustic features for predicting perceived emotion?
2. Does combining both feature families outperform either alone?
3. Which performance techniques are associated with which emotions?

## Dataset

- **60 audio excerpts** — 20 pieces × 3 performance conditions:
  - `MEC` — mechanical (neutral baseline)
  - `EXP` — expressive (natural)
  - `EXG` — exaggerated (heightened expression)
- **118 participants** across 5 listening sessions rated each excerpt on **valence (0–6)**, **arousal (0–6)**, and tagged the perceived emotions.
- Emotions are organized on the valence–arousal circumplex: Tenderness, Nostalgia, Peacefulness, Power, Joyful Activation, Tension, Sadness, and Neutral.

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
│   └── thesis_handoff.md          # detailed project notes
├── CLAUDE.md                      # project context
├── .gitignore
└── README.md
```

## Methods

**Feature extraction.** Each excerpt is encoded with MERT (embeddings averaged over mid-stack transformer layers) and with CREPE-derived pitch features, giving three feature sets: MERT, CREPE, and Combined.

**Classification.** Multi-label emotion prediction with Leave-One-Out Cross-Validation across several classifiers, evaluated per feature set.

**Regression.** Continuous valence and arousal prediction (Ridge / SVR / MLP).

**Technique–emotion analysis.** Two complementary views: SHAP feature importance (which techniques predict each emotion) and condition-delta analysis (how technique changes across conditions relate to changes in perceived emotion).

**Participant & excerpt screening.** Standalone scripts apply established quality-control methods — robust Mahalanobis distance (MCD), leave-one-out person-total correlation, response-variability and long-string checks for participants; one-way ICC reliability, agreement (rWG), and per-piece manipulation checks for excerpts.

## Setup

The pipeline is developed for a **Kaggle GPU environment (NVIDIA P100)**.

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

*Master's thesis — music information retrieval / affective computing.*
