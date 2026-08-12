# Thesis Handoff — Violin Expressiveness → Emotion Recognition

**Purpose:** complete standalone context to continue work in a fresh conversation. Covers (1) scope, (2) current pipeline architecture, (3) progress, (4) findings, (5) open issues, (6) technical report status, (7) conventions, (8) workflow.

> **Revision status (current):** This supersedes the older "validation-mode (MERT+CREPE only)" handoff. The pipeline has since gained **per-rater normalization** (new ground truth), an **active Essentia block** (dynamics/timbre/tonal), a **shared leakage-free forwarding gate** for every tool, a full **Part 5 attribution suite** and **Part 6 tool-characterization**, and the **MFCC baseline was removed**. The interpretable-feature narrative has moved **from vibrato to dynamics/timbre + pitch**. DTW (timing) is the next addition. Notebook: `thesis-pipeline.ipynb`, **86 cells**, Kaggle P100.

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
- **118 participants across 5 linked questionnaire forms.** Ratings: **valence 0–6**, **arousal 0–6**, plus multi-select emotion tags.
- **Ring (linked) form design:** 20 excerpts/form; **8 anchors shared with each neighbouring form, 4 unique**, so the five forms connect cyclically through **40 anchor excerpts**. This linking underpins the normalization validation (§3/§1.2b).
- **Scale 0–6** (7 points, midpoint **3.0**). `0` is a genuine lowest rating (verified in raw forms), **not** NA. `VA_MID = 3.0` — correct, do not change.
- **Ground truth = 5 classes** (4 circumplex quadrants + Neutral), derived from **normalized** VA means (see §1.1b flow). `NEUTRAL_RADIUS = 0.75` (un-tuned — see open issues).

**Environment:** Kaggle, P100 GPU, PyTorch 2.3.1 (last sm_60 build).

---

## 2. Pipeline Architecture (current, 86 cells)

Runs top-to-bottom. `§` = the in-notebook section banners.

| Part | § | Content |
|---|---|---|
| **0 Setup** | 0.1–0.4 | install / imports / **config** (`VA_MID=3.0`, `NEUTRAL_RADIUS=0.75`, `MERT_LAYERS=[5,6,7]`, `QUADRANT_EMOTIONS`, tag maps) / audio manifest |
| **1 Audience ground truth** | 1.1 | parse 5 CSVs → `df_long` (keeps `0`) |
| | **1.1b** | **per-rater normalization → new ground truth** (per-dim z, σ-floor, rescale preserving grand-mean + between-excerpt SD) |
| | 1.2 | aggregate → `df_agg`; derive 5-class labels from **normalized** VA; raw kept in parallel |
| | **1.2b** | **normalization validation** (anchor cross-form Δ, quadrant stability, corr, variance) |
| | 1.3 | label matrix `Y` (60×N), `mlb`, `df_expr`, `df_all` |
| | 1.4 | rating stats — Friedman / Kendall W / Wilcoxon |
| | 1.5 | lift analysis (emotion↔condition, raw tags) |
| **2 Feature extraction** | 2.1 / **2.1b** | **MERT** (768-dim, layers 5–7) / **controlled MERT forwarding** (PCA-95, unsupervised) |
| | 2.2 / **2.2b** | **CREPE** (pitch/vibrato/portamento) / **controlled CREPE forwarding** |
| | 2.3 / **2.3b** | **Essentia** (dynamics/timbre/tonal) / **controlled Essentia forwarding** |
| | 2.4 | **Planned extractors — madmom / DTW / MusiCNN (SKIP STUBS)** ← DTW being reactivated |
| | 2.5 | assemble `X_A/B/C/D`, `X_MERT_FULL`; `STRICT_BLOCKS` guard |
| **3 Validation & EDA** | 3.1–3.4 | class floors · VA/coverage · CREPE sanity · MERT structure (PCA scree) |
| **4 Prediction** | 4.1 | multi-label classification, LOOCV, feature sets A/B/C/D |
| | 4.2 | valence/arousal regression |
| | 4.4 | ablation table + Wilcoxon |
| | 4.5 / **4.5b** | per-class P/R/F1 / **DummyClassifier floors** (the official floor) |
| | 4.6 | per-condition F1 (MEC/EXP/EXG) |
| | 4.7 | quadrant accuracy from predicted VA (+ VA scatter, circumplex) |
| | 4.8 | prediction–ground-truth agreement (Krippendorff α) |
| **5 Attribution** | 5.1 | **condition-delta** — technique change vs rating change (within-piece) |
| | **5.1b** | **dose-response** — Page's trend test over ordered MEC<EXP<EXG |
| | 5.2 | **SHAP** technique importance per class |
| | 5.3 | **embedding-distance** attribution |
| | 5.4 | **interpretable linear read-out** (signed technique→emotion) |
| | 5.5 | **cost of explainability** (interpretable vs representation, same CV) |
| **6 Tool characterization** | 6.1 / 6.1b | MERT-dim ↔ feature correlation / linear probe (does MERT encode each feature?) |
| | 6.2 | **tool complementarity** — does each tool add signal on top of MERT? |
| **7 Synthesis** | 7.1 | final summary + save all outputs |

### 2.1 The shared forwarding gate (key mechanism)

Every tool is **extracted in full** (nothing lost on disk), then **forwarded** through one shared, **unsupervised** selector `unsup_forward_select(candidates, corr_max=0.90, name, eps, order)`: near-constant prune + correlation prune, in an a-priori preference order. It reads **only the feature matrix, never `Y`** — this is what makes a once-up-front selection safe to feed LOOCV (a supervised selection would leak held-out folds). `StandardScaler` stays **inside** every CV pipeline. MERT (§2.1b), CREPE (§2.2b), Essentia (§2.3b) all use this same gate → `MERT_FORWARD`, `CREPE_FORWARD`, `ESSENTIA_FORWARD`.

### 2.2 What enters the models vs what is kept aside

`X_A=MERT-fwd`, `X_B=CREPE-fwd`, `X_C=MERT+CREPE+Essentia (all fwd)`, `X_D=MERT+Essentia`. **`X_MERT_FULL`** (768-dim) is kept aside — **not** a model input — for the *representational* analyses (§3.4 structure, §5.3 distances, §6.1/§6.1b correspondence), which must see all dimensions. `STRICT_BLOCKS=('MERT','Essentia')` → a missing block raises rather than silently zero-filling (per the "hard imports, no fallbacks" policy).

---

## 3. Progress since the previous handoff

- **Per-rater normalization added (§1.1b/§1.2b) and made the new ground truth.** Per-dimension z with σ-floor 0.5, rescaled to preserve the raw grand mean and between-excerpt SD (so VA_MID and quadrant geometry are untouched — only rater scale-use bias is removed). **Validated via the ring:** the 40 anchors are each rated by two independent pools, so the cross-form discrepancy before/after is the honest test. **Functional with or without participant exclusions** (all constants derived at runtime). Raw retained in parallel.
- **Essentia re-enabled (§2.3/§2.3b)** — dynamics (integrated loudness, dynamic/loudness range), timbre (spectral flux/centroid/rolloff, dissonance, log-attack time), tonal (key strength, HPCP entropy). This closed the dynamics/timbre gap and populates the signal-processing tool category.
- **Shared leakage-free forwarding** unified across all tools (§2.1).
- **Part 5 attribution suite** built out: condition-delta, dose-response (Page), SHAP, embedding-distance, interpretable read-out, cost of explainability.
- **Part 6 tool characterization** (correspondence probe + complementarity).
- **MFCC baseline removed** — it was degenerate (F1≈0, indistinct from the Dummy floor) and carried out-of-topic leaderboard framing. The **DummyClassifier floors (§4.5b)** are the official chance baseline.
- **Cleanup pass (partial):** MFCC gone; some duplicate/out-of-topic figures still pending removal (see open issues).
- **Technical report drafted** — Abstract + Introduction + Methods (IEEEtran), following the Chowdhury/Widmer interpretable-MER backbone (see §6).
- **DTW (timing) = next addition** (see the accompanying implementation prompt).

---

## 4. Current findings

> Numbers below are the ones confirmed in the current run / this working session. Exact classification & regression F1 shift with the Essentia+normalization additions — **read those from the live §4 output** rather than any older figure.

**Headline 1 — arousal/valence asymmetry.** Arousal is interpretable and predictable (interpretable read-out r ≈ **0.82**; MERT-full ≈ **0.86**), with a near-zero **cost of explainability (+0.037)** — the interpretable features essentially match the black box. Valence resists prediction (**CoE +0.300**). This is a known MER property (valence needs harmony/context, not more low-level descriptors), so it is a **finding, not a feature deficiency**.

**Headline 2 — escalation via dynamics & timbre, not vibrato (contrarian).** Dose-response (Page's trend, 20/20 complete pieces): **spectral flux z=+5.38\*\*\***, **loudness z=+4.74\*\*\*** rise; **rolloff z=−2.85\*\*** falls; **vibrato rate z=+1.26 n.s.** Audience side: **arousal z=+4.11\*\*\*** vs **valence z=+1.26 n.s.** ⚠️ Note the honest nuance: the mean EXG−MEC for vibrato is *positive* (rate +0.234, depth +1.903) — vibrato does rise on average, but **non-monotonically/inconsistently** (only 10–30% of pieces monotone), which is why it is n.s. The correct claim is "vibrato change is weaker and less consistent **with the current mean descriptor**," not "vibrato does nothing." A richer vibrato descriptor (coverage/extent, note-level) may recover it — open question.

**Normalization validation.** Raw grand mean V=2.97 / A=3.40 (arousal legitimately skews high); between-excerpt SD V=1.065 / A=0.956 (preserved); anchor cross-form |Δ| baseline **V=0.334 / A=0.289**; corr raw↔norm ≈ **0.997**, **0/60 change quadrant**. Per-rater min SD 0.51/0.54 → the σ-floor **never fires** (insurance, not active). **Conclusion: normalization is a defensibility/robustness step — it confirms the signal is real, not a rater artifact; it does not change results.**

**Cleaning.** 3 participants flagged (≥2 methods) — **not yet removed**; decision pending. Excerpt QC: valence ICC(1,k)≈0.96, arousal ≈0.95; a few arousal inversions among the 20 pieces (manipulation-check candidates).

---

## 5. Open issues

1. **"8 emotions" is mechanically 5 classes.** `QUADRANT_EMOTIONS` maps each quadrant to a *set*, so pairs (Tenderness≡Peacefulness, Power≡Joyful, Sadness≡Nostalgia) are byte-identical columns. **Decision taken: report honestly as 5 classes** (4 quadrants + Neutral). Alternative (rebuild `Y` from raw multi-label tags) remains available. Good supervisor topic.
2. **Neutral radius un-tuned.** `NEUTRAL_RADIUS=0.75` chosen geometrically. Needs sensitivity (0.5/0.75/1.0); report macro-F1 with and without Neutral.
3. **Figure dedup still pending** (partial cleanup only): a duplicate VA scatter still sits alongside the circumplex; check for any remaining per-class-F1 heatmap/radar and ablation-figure redundancy and prune (keep circumplex, P/R/F1 bars, PCA scree).
4. **DTW not yet added** — the χρονικές παραλλαγές (timing/rubato) technique named in the brief is still uncovered until DTW lands (next step).
5. **Participant exclusions not applied** — pipeline runs on all 118; wire the 3 flagged in once decided (normalization is already robust to this).
6. **Statistics hygiene:** dose-response / correlation analyses want FDR or a pre-registered hypothesis set; phrase classifier-set comparisons as "across the classifiers tried," not population inference; nest any selected-on-CV parameters.
7. **Feature hygiene:** `crepe_portamento_count` is a raw count (normalise per second); `crepe_vibrato_rate_hz` is bandpass-bounded to [4,8] Hz (trust depth); vibrato/portamento computed on concatenated voiced frames (prefer time-contiguous segments).

---

## 6. Technical report status

First draft written for **Abstract, Introduction, Methods** (IEEEtran, English). Structural decisions (from a survey of the closest interpretable-MER / expressive-performance literature):

- **Backbone = Chowdhury/Widmer interpretable-MER skeleton:** interpretable layer → black-box baseline → an explicit **"cost of explainability"** results section.
- **The two categorisations are their own sections** (tools; modalities), *before* Methods; Methods **instantiates** their members with cross-references (no re-description).
- **Aims woven into prose, not numbered research questions**; the two taxonomies appear as content, never flagged as questions or "axes."
- **Abstract carries no counts** (only the three levels MEC/EXP/EXG) + the two headline findings.
- **MERT positioned as a frozen/probed representation** (deep-family exemplar), explicitly not a benchmark competitor.
- **Predictive validation (LOOCV) framed as supportive → appendix**, not a leaderboard.

Methods subsections drafted: study design & graded manipulation · stimuli/corpus · participants & questionnaire (ring) · emotion measurement & ground-truth · **response normalization** · feature extraction (CREPE / Essentia / MERT as parallel blocks; DTW timing noted as planned) · analysis pipeline (deltas, dose-response, read-out, cost of explainability). Unknowns (how the three conditions were produced in detail, piece list, audio specs, demographics, exact tag set, author block) are marked as visible placeholders in the `.tex`.

---

## 7. Key conventions / variables (current)

- **Scale 0–6**, `VA_MID=3.0`, `NEUTRAL_RADIUS=0.75`, `MERT_LAYERS=[5,6,7]`.
- `df_long` — long-format per-rating (parse, §1.1); gains `valence_norm/arousal_norm` in §1.1b.
- `df_agg` / `df_all` — per-excerpt aggregated. **`valence_mean/arousal_mean` are now the NORMALIZED means** (ground truth); `valence_mean_raw/arousal_mean_raw` kept in parallel.
- `emotion_labels` — VA-quadrant-derived (from normalized means); **this becomes `Y`**. `top_tags` — raw tags, descriptive only (§1.5).
- `Y` (60×N), `mlb.classes_` = class names; `df_expr` — modelled-excerpt frame.
- **Forwarding:** `unsup_forward_select(...)`; `MERT_FORWARD` / `CREPE_FORWARD` / `ESSENTIA_FORWARD`; forwarded frames `emb_mert_fwd`, `df_crepe_fwd`, `df_ess_fwd`. Full blocks on disk: `df_mert`, `df_crepe`, `df_essentia`.
- **Assembly:** `FEAT_A/B/E/C/D`, `X_A/X_B/X_C/X_D`, `X_MERT_FULL`; `STRICT_BLOCKS=('MERT','Essentia')`; `bfv(...)` builder.
- **Interpretable frame & labels** used by Part 5: `df_interp` + `INTERP_FEAT_LABELS` (aggregate CREPE + Essentia; **DTW must be added here when reactivated** — see prompt).
- Part 5 outputs: `df_dose` (§5.1b) · SHAP importances (§5.2) · read-out coefficients (§5.4) · CoE table (§5.5).
- Excerpt IDs `PieceName_ConditionCode`; participant IDs `questionnaire_{n}_P{idx:03d}` (alias `S{n}_P{idx:03d}`).
- `OUTPUT_DIR` — all figures/CSVs. Skip stubs at §2.4: madmom, DTW, MusiCNN.

---

## 8. Workflow

**Repo:** `makris2003/violin-emotion-thesis` (private). GitHub = source of truth; Kaggle = GPU execution. `CLAUDE.md` at repo root is auto-read by Claude Code. Loop: edit locally (VS Code) → commit/push → Kaggle pulls → run on GPU → commit results back. Claude Code **cannot run** the notebook (no audio/GPU) → surgical edits + static checks only. Kaggle "Pull from GitHub" can stale its OAuth token (unlink/relink to fix).

---

## 9. Quick answers

- **"Why is valence weak?"** Intrinsic to MER (needs harmony/context). Arousal is clean with the same features → it's a finding. To *explain* (not rescue) it, add 1–2 harmonic features (mode / tonal tension) so you can say "valence is governed by harmony, which the graded expressivity holds fixed."
- **"Did normalization change results?"** No — corr 0.997, 0/60 flip. That's the point: it confirms the findings aren't a rater artifact (defensibility, not correction).
- **"Why remove MFCC?"** Degenerate (F1≈0), indistinct from the Dummy floor, out-of-topic. Dummy floors are the official baseline.
- **"Is it a leaderboard (MERT vs CREPE)?"** No — that framing is de-emphasised. It's a categorisation + interpretable-attribution study; cost of explainability is the comparison, not "who wins."
- **"Why forward features up front unsupervised?"** To keep LOOCV leakage-free — a supervised selection would leak held-out folds. The gate never sees `Y`.