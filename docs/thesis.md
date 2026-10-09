# Thesis Handoff — Violin Expressiveness → Emotion Recognition

**Purpose:** complete standalone context to continue work in a fresh conversation. Covers (1) scope, (2) current pipeline architecture, (3) progress, (4) findings, (5) open issues, (6) technical report status, (7) conventions, (8) workflow.

> **Revision status (current): Phase A applied — results pending the next Kaggle run.** The Oct-2026 audit's Phase A corrections (§10 below) are in the notebook (now **101 cells**); every Part 4/5/6 predictive number quoted from earlier runs is pre-audit and superseded. No Phase-A number exists yet.
>
> **Earlier revision note:** This supersedes the older "validation-mode (MERT+CREPE only)" handoff. The pipeline has since gained **per-rater normalization** (new ground truth), an **active Essentia block** (dynamics/timbre/tonal), a **shared leakage-free forwarding gate** for every tool, a full **Part 5 attribution suite** and **Part 6 tool-characterization**, and the **MFCC baseline was removed**. The interpretable-feature narrative has moved **from vibrato to dynamics/timbre + pitch**. **DTW (timing/rubato) is implemented and wired end-to-end (§2.4/§2.4b)** and a **systematic ablation study** (§4.4b/c/d + §5.6, one shared engine, two CV protocols, FDR-corrected) was added; **both were executed in the Kaggle run of 2026-08-16**. Notebook: `notebook/thesis_pipeline.ipynb`, **95 cells**, Kaggle P100.

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

## 2. Pipeline Architecture (current, 101 cells — Phase A applied)

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
| **2 Feature extraction** | 2.1 / **2.1b** | **MERT** (768-dim, layers 5–7; **positional-conv weight-norm tensors restored from the checkpoint + asserted**, cache `emb_mert_v2.npz`) / **controlled MERT forwarding** (PCA-95, unsupervised) |
| | 2.2 / **2.2b** | **CREPE** (pitch/vibrato/portamento) / **controlled CREPE forwarding** |
| | 2.3 / **2.3b** | **Essentia** (dynamics/timbre/tonal) / **controlled Essentia forwarding** |
| | **2.3c** | **recording-level sanity check** (report only) — peak/clipping, LUFS robust z, within-piece LUFS, noise-floor "recording-chain change" test |
| | 2.4 / **2.4b** | **DTW timing / rubato (ACTIVE)** — reference-free timing family + within-piece warp-path family / **controlled DTW forwarding**. madmom / MusiCNN remain skip stubs |
| | 2.5 | assemble `X_A/B/C/D/F`, `X_MERT_FULL`; `STRICT_BLOCKS` guard |
| **3 Validation & EDA** | 3.1–3.4 | class floors · VA/coverage · CREPE sanity · **3.3b DTW timing sanity + manipulation check** · MERT structure (PCA scree) |
| **4 Prediction** | **4.0** | **evaluation protocol & CV engine** — `get_cv` (piece = PRIMARY, loo = secondary), GaussianNB default, `cv_predict_multilabel`, `nested_ridge_predict`, `oos_r2`/`rmse`, `bh_fdr`, re-seeded Dummy floors (mean/SD/P95), equivalence guard |
| | 4.1 | multi-label classification — **8-classifier sensitivity table** (not used for selection), both protocols; balanced + inner-CV C for SVM/LogReg (A5) |
| | 4.2 | valence/arousal regression (fixed models), both protocols, r · R² · RMSE |
| | 4.4 | feature-set table + Wilcoxon (four hand-picked sets), per protocol |
| | **4.4b** | **ablation engine + block lattice** — all 15 subsets of {MERT, CREPE, Essentia, DTW}; unique vs marginal contribution per tool |
| | **4.4c** | **configuration ablation** — `MERT_PCA_VAR` · Essentia curation · DTW family · portamento normalisation · shared-gate `corr_max` · **`4D-balancing`** (NB priors vs SVM balancing, descriptive) |
| | **4.4d** | **target-side ablation** — `NEUTRAL_RADIUS` 0.5/0.75/1.0 (macro-F1 with *and* without Neutral) · normalized vs raw ground truth |
| | 4.5 / **4.5b** | per-class P/R/F1 (default on X_C) / **DummyClassifier floors** (the official floor; mean, SD, P95 over 200 re-seeded repeats) |
| | 4.6 | per-condition F1 (MEC/EXP/EXG), default on X_C |
| | 4.7 | quadrant accuracy from predicted VA (+ VA scatter, circumplex), both protocols |
| | 4.8 | prediction–ground-truth agreement (Krippendorff α), default on X_C |
| | **4.9** | **within-piece prediction** — piece-centred targets & features, piece CV only, Condition-only reference, within-piece permutation p |
| **5 Attribution** | 5.1 | **condition-delta** — technique change vs rating change (within-piece); **BH-FDR over all 174 tests** |
| | **5.1b** | **dose-response** — Page's trend test over ordered MEC<EXP<EXG |
| | 5.2 | **SHAP** technique importance per class |
| | 5.3 | **embedding-distance** attribution |
| | 5.4 | **interpretable linear read-out** (signed technique→emotion) |
| | 5.5 | **cost of explainability** (interpretable vs representation, same CV, both protocols; headline max(MERT-PCA, MERT-full) + the two separate gaps; handles a negative CoE) |
| | **5.6** | **sub-family ablation** — decomposes the CoE across Pitch/Vibrato/Portamento/Dynamics/Timbre/Tonal/Timing; writes the consolidated `ablation_master.csv` |
| **6 Tool characterization** | 6.1 / 6.1b | MERT-dim ↔ feature correlation / linear probe (does MERT encode each feature?) — **GroupKFold(5) over pieces, α nested, grid to 1e4** |
| | 6.2 / 6.2b | **tool complementarity** — does each tool add signal on top of MERT? (default classifier, both protocols) / guard vs the lattice |
| **7 Synthesis** | 7.1 | final summary + save all outputs |

### 2.1 The shared forwarding gate (key mechanism)

Every tool is **extracted in full** (nothing lost on disk), then **forwarded** through one shared, **unsupervised** selector `unsup_forward_select(candidates, corr_max=0.90, name, eps, order)`: near-constant prune + correlation prune, in an a-priori preference order. It reads **only the feature matrix, never `Y`** — this is what makes a once-up-front selection safe to feed LOOCV (a supervised selection would leak held-out folds). `StandardScaler` stays **inside** every CV pipeline. MERT (§2.1b), CREPE (§2.2b), Essentia (§2.3b) and DTW (§2.4b) all use this same gate → `MERT_FORWARD`, `CREPE_FORWARD`, `ESSENTIA_FORWARD`, `DTW_FORWARD`.

### 2.2 What enters the models vs what is kept aside

`X_A=MERT-fwd`, `X_B=CREPE-fwd`, `X_C=MERT+CREPE+Essentia+DTW (all fwd)`, `X_D=MERT+Essentia`, `X_F=MERT+DTW` (the §6.2 complementarity contrast). **`X_MERT_FULL`** (768-dim) is kept aside — **not** a model input — for the *representational* analyses (§3.4 structure, §5.3 distances, §6.1/§6.1b correspondence), which must see all dimensions. `STRICT_BLOCKS=('MERT','Essentia')` → a missing block raises rather than silently zero-filling (per the "hard imports, no fallbacks" policy).

### 2.3 Evaluation protocol (§4.0, Phase A) — one engine for every cross-validated number

- **Protocols:** `PRIMARY_PROTOCOL='piece'` (leave-one-piece-out on `PIECE_GROUPS`) and `loo` (optimistic secondary), always both, piece first, with the optimism gap; `get_cv(protocol)` is the only splitter builder (`'piece5'` = GroupKFold(5) on pieces, for the §6.1b probe only).
- **Default classifier (pre-declared):** GaussianNB (`DEFAULT_CLF_NAME`, `make_default_clf`). §4.1's 8 classifiers are a sensitivity table only; per-class / per-condition / α analyses use the default on `X_C`. Its known downsides (independence violated, no shrinkage, Gaussian fit of zero-inflated features, uncalibrated probabilities) are stated in §4.0; no probability is interpreted anywhere.
- **Engine:** `cv_predict_multilabel` (manual binary relevance, `SafeMultiOutputClassifier` semantics, optional per-label inner-CV tuning) — asserted identical to the old pipeline for GaussianNB; `nested_ridge_predict` (α by piece-grouped inner GroupKFold(5), replaces every `RidgeCV`); `NestedRidgeOperator` (closed-form twin, for permutation refits); `oos_r2` (baseline = training-fold mean), `rmse`, `bh_fdr`, `within_piece_permutation`.
- **Chance floors:** Dummy seeded per fold × label, 200 repeats → mean / SD / P95; "beats chance" = above the max P95 across strategies.

---

## 3. Progress since the previous handoff

- **Phase A of the Oct-2026 audit applied (code only; numbers pending the next Kaggle run).** §4.0 CV engine · re-seeded Dummy floors · leave-one-piece-out primary everywhere · nested ridge + R²/RMSE · §4.9 within-piece prediction · MERT pos-conv weight restore (cache v2) · GaussianNB default + best-of selection removed · class balancing + inner-CV C (+ tier `4D-balancing`) · group-aware MERT probe · BH-FDR in §5.1 · §2.3c recording-level check. See §10.

- **Per-rater normalization added (§1.1b/§1.2b) and made the new ground truth.** Per-dimension z with σ-floor 0.5, rescaled to preserve the raw grand mean and between-excerpt SD (so VA_MID and quadrant geometry are untouched — only rater scale-use bias is removed). **Validated via the ring:** the 40 anchors are each rated by two independent pools, so the cross-form discrepancy before/after is the honest test. **Functional with or without participant exclusions** (all constants derived at runtime). Raw retained in parallel.
- **Essentia re-enabled (§2.3/§2.3b)** — dynamics (integrated loudness, dynamic/loudness range), timbre (spectral flux/centroid/rolloff, dissonance, log-attack time), tonal (key strength, HPCP entropy). This closed the dynamics/timbre gap and populates the signal-processing tool category.
- **Shared leakage-free forwarding** unified across all tools (§2.1).
- **Part 5 attribution suite** built out: condition-delta, dose-response (Page), SHAP, embedding-distance, interpretable read-out, cost of explainability.
- **Part 6 tool characterization** (correspondence probe + complementarity).
- **MFCC baseline removed** — it was degenerate (F1≈0, indistinct from the Dummy floor) and carried out-of-topic leaderboard framing. The **DummyClassifier floors (§4.5b)** are the official chance baseline.
- **Cleanup pass (partial):** MFCC gone; some duplicate/out-of-topic figures still pending removal (see open issues).
- **Technical report drafted** — Abstract + Introduction + Methods (IEEEtran), following the Chowdhury/Widmer interpretable-MER backbone (see §6).
- **DTW (timing/rubato) implemented and integrated (§2.4/§2.4b).** Two families: **(1) reference-free** — local-tempo variability, IOI CV, onset rate, pulse clarity; computed per excerpt with no reference, so all 60 (MEC included) are graded and the family is condition-INdependent. **(2) within-piece DTW deviation** — chroma sequences of EXP/EXG aligned to the same piece's MEC with `librosa.sequence.dtw`; warp-path deviation, local-slope variability, tempo/path ratios. MEC is the alignment reference, so its family-(2) values are the **self-alignment identity (0 deviation / 1 ratio) by construction** — meaningful for attribution but it means family (2) partly encodes the condition. §2.4b splits them: family (1) alone feeds the predictive matrices (`DTW_MODEL_COLS`, `DTW_DEV_IN_MODELS=False`), both feed the attribution frame (`df_dtw_attr`, `DTW_DEV_IN_INTERP=True`). DTW is wired into §2.5 (X_C, new X_F), §3.3b, §5.1, §5.1b, §5.2, §5.4, §5.5, §6.1, §6.1b, §6.2. **Executed in the 2026-08-16 run** (13-col block; 11 forwarded = 8 ref-free + 3 dev; the models receive the 8 ref-free).
- **Systematic ablation study added (§4.4b/c/d + §5.6 + the §6.2b guard).** One engine `ablate()` scores every tier under one protocol (SVM-Linear → F1-macro, RidgeCV → V/A, scaler inside the CV pipeline — the same protocol as §5.5/§6.2, so all numbers are comparable by construction), caches per-excerpt predictions, and consolidates into `ablation_master.csv`. It closes the asymmetry that §4.4 compared only four hand-picked sets while §6.2 could only ever *add* to MERT: the **full 15-subset lattice** now yields each tool's **unique** (full − full∖tool) and **marginal** (alone − chance floor) contribution, MERT's included. Every row is scored under **both** LOOCV and leave-one-piece-out, with bootstrap CIs, paired permutation p-values and BH-FDR. §5.6 decomposes the single cost-of-explainability number by expressive family, giving Part 5 a third and *predictive* line of evidence alongside dose-response and SHAP. Three asserts keep it honest (the lattice must rebuild `X_C/X_D/X_F` exactly, the §4.4d label re-derivation must reproduce `Y`, and §6.2's additive numbers must equal the lattice rows). **Executed in the 2026-08-16 run** (112 scored rows, 9 tiers; all three asserts passed) — those numbers are pre-audit (SVM-Linear + RidgeCV-GCV protocol) and superseded by Phase A.

---

## 4. Current findings

> Numbers below are the ones confirmed in the current run / this working session. Exact classification & regression F1 shift with the Essentia+normalization additions — **read those from the live §4 output** rather than any older figure.

> **Pre-audit numbers.** Everything in this section that comes from Part 4/5/6 *prediction* (CoE, r, F1) is from the Kaggle run of **2026-08-16** (DTW + full ablation executed) and is **superseded by the Phase A run** (MERT re-extraction with the positional-conv fix, GaussianNB default, leave-one-piece-out primary protocol, nested piece-grouped ridge). §5.1 / §5.1b numbers are untouched by Phase A (same features, same ratings); Phase B will change them.

**Headline 1 — arousal/valence asymmetry (pre-audit).** Cost of explainability (§5.5, LOOCV): **ΔR valence +0.112, ΔR arousal +0.051, ΔF1 +0.145** (interpretable r_val 0.768 / r_aro 0.837 vs MERT-PCA 0.811 / 0.796 and MERT-full 0.880 / 0.888). The older +0.300 / +0.037 are obsolete. Arousal is the better-recovered dimension, consistent with the known MER property that valence needs harmony/context — but see the audit: under LOOCV these numbers mostly measure *piece recognition*.

**Variance structure (audit).** 95% of valence and 91% of arousal variance across the 60 excerpts is **between-piece**; the condition explains **1% (V) / 4% (A)**. LOOCV (which keeps a piece's other two renditions in training) therefore mostly measures piece recognition; leave-one-piece-out and the within-piece analysis (§4.9) are the honest tests.

**Headline 2 — escalation via dynamics & timbre, not vibrato (contrarian).** Dose-response (Page's trend, 20/20 complete pieces): **spectral flux z=+5.38\*\*\***, **loudness z=+4.74\*\*\*** rise; **rolloff z=−2.85\*\*** falls; **vibrato rate z=+1.26 n.s.** Audience side: **arousal z=+4.11\*\*\*** vs **valence z=+1.26 n.s.** ⚠️ Note the honest nuance: the mean EXG−MEC for vibrato is *positive* (rate +0.234, depth +1.903) — vibrato does rise on average, but **non-monotonically/inconsistently** (only 10–30% of pieces monotone), which is why it is n.s. The correct claim is "vibrato change is weaker and less consistent **with the current mean descriptor**," not "vibrato does nothing." A richer vibrato descriptor (coverage/extent, note-level) may recover it — open question.

**Normalization validation.** Raw grand mean V=2.97 / A=3.40 (arousal legitimately skews high); between-excerpt SD V=1.065 / A=0.956 (preserved); anchor cross-form |Δ| baseline **V=0.334 / A=0.289**; raw↔norm Pearson r = **0.996 (V) / 0.993 (A)**; **10/60 excerpts change class, 4/60 change quadrant** (the earlier "0/60 flips" claim was wrong). Per-rater min SD 0.51/0.54 → the σ-floor **never fires** (insurance, not active). **Conclusion (unchanged): normalization is a defensibility/robustness step — it confirms the signal is real, not a rater artifact; no material change in results.**

**Stimulus level.** The stimuli played to participants were **not loudness-normalized** (confirmed by the author): level differences between excerpts are part of the stimulus. §2.3c (Phase A) only checks that no recording was made at a clearly different gain/microphone distance.

**Cleaning.** 3 participants flagged (≥2 methods) — **not yet removed**; decision pending. Excerpt QC: valence ICC(1,k)≈0.96, arousal ≈0.95; a few arousal inversions among the 20 pieces (manipulation-check candidates).

---

## 5. Open issues

1. **"8 emotions" is mechanically 5 classes.** `QUADRANT_EMOTIONS` maps each quadrant to a *set*, so pairs (Tenderness≡Peacefulness, Power≡Joyful, Sadness≡Nostalgia) are byte-identical columns. **Decision taken: report honestly as 5 classes** (4 quadrants + Neutral). Alternative (rebuild `Y` from raw multi-label tags) remains available. Good supervisor topic.
2. ~~**Neutral radius un-tuned.**~~ **Now measured in §4.4d** — the 0.5/0.75/1.0 sweep runs on both X_A and X_C, reports macro-F1 **with and without** Neutral, and prints class support plus how many excerpts each radius relabels. Remaining decision: read the table once it has run and keep 0.75 unless another radius wins on *both* columns.
3. **Figure dedup still pending** (partial cleanup only): a duplicate VA scatter still sits alongside the circumplex; check for any remaining per-class-F1 heatmap/radar and ablation-figure redundancy and prune (keep circumplex, P/R/F1 bars, PCA scree).
4. **DTW block size / DTW-dev in `df_interp`.** In the 2026-08-16 run the §2.4b gate forwarded 11 columns (8 ref-free + 3 dev) and the "outside 3–10" warning fired — trim `DTW_CANDIDATES` if that matters. Open: whether letting the DTW-deviation family into `df_interp` moves the §5.5 cost-of-explainability numbers, since those columns are 0/1 at MEC by construction (`DTW_DEV_IN_INTERP=False` gives a strictly condition-independent CoE).
5. **Participant exclusions not applied** — pipeline runs on all 118; wire the 3 flagged in once decided (normalization is already robust to this).
6. **Statistics hygiene:** the ablation study (§4.4b) carries bootstrap CIs, paired permutation tests and **BH-FDR within each tier × protocol** (now over F1, r and R²); **§5.1 now applies BH-FDR over all its tests** (Phase A); §5.1b (dose-response) still wants the same treatment or a pre-registered hypothesis set — not yet approved. Keep phrasing classifier-set comparisons as "across the classifiers tried," not population inference. **The lattice must stay descriptive** — selecting the winning subset on the same CV and then reporting its score is exactly the nested-selection error.
7. **Feature hygiene:** `crepe_portamento_count` is a raw count — §4.4c measures the per-second fix (needs audio durations, skips loudly otherwise); `crepe_vibrato_rate_hz` is bandpass-bounded to [4,8] Hz *and argmax-ed inside that same band*, so it can never report "no vibrato" (trust depth); vibrato/portamento computed on concatenated voiced frames (prefer time-contiguous segments). **Consequence for the write-up:** any vibrato null — the n.s. dose-response or a §5.6 ablation Δ≈0 — is evidence about the **descriptor** and belongs to the tool taxonomy, not to the musical claim "vibrato does nothing". Repair the descriptor before promoting it to a musical finding.
8. ~~**LOOCV leaks piece identity.**~~ **Addressed in Phase A:** leave-one-piece-out is now the PRIMARY protocol everywhere (§4.0), LOOCV is the optimistic secondary, and §4.9 removes the piece entirely (within-piece prediction). Quote the `piece` number for any claim about unseen material.
9. **`LA_HV` has n=5 from only 2 pieces** (La_Vita ×3, Meditation_Thais MEC+EXP): under `piece` it is learned from one piece at a time, and inner-CV tuning is skipped for it in many folds.
10. **Phase B (deferred) — descriptors measure something else than their name:** vibrato descriptor, portamento detector, F0 range, spectral-flux normalisation, dynamic-range silence gate, timing onsets/tempogram.
11. **Phase C (deferred idea) — augmentation.** The supervisor's suggestion: augment the minority positive-valence classes (pitch shift ±1 semitone, tempo ±5%, never combined), with augmented copies used **only in training folds, and only when their source piece is itself in that training fold** (test = originals only, gate/PCA fitted on originals, scoring/permutations on the 60 originals). Caveats: it cannot add piece diversity (LA_HV comes from only 2 pieces); it perturbs studied variables (F0 mean, tempo, onset rate) while keeping labels; the copies' labels were never rated; phase-vocoder artefacts affect Essentia timbre features. **Decision:** evaluate class balancing (A5, tier `4D-balancing`) first; augmentation later as an ablation tier.
12. **Not approved yet:** SHAP fitted in-sample (§5.2) · §5.3 embedding-distance sign · §1.4 mixed model · BH in §5.1b.
13. **Known, not fixed (out of scope):** Fig 5 VA scatter (§4.7) never draws the MEC points (`cc` holds only EXP/EXG); README points to `docs/thesis_handoff.md`, the file is `docs/thesis.md`. §5.5's headline CoE is a best-of over two black boxes (max(MERT-PCA, MERT-full)) — conservative; the two separate gaps are in `coe_gaps.csv`.

---

## 6. Technical report status

First draft written for **Abstract, Introduction, Methods** (IEEEtran, English). Structural decisions (from a survey of the closest interpretable-MER / expressive-performance literature):

- **Backbone = Chowdhury/Widmer interpretable-MER skeleton:** interpretable layer → black-box baseline → an explicit **"cost of explainability"** results section.
- **The two categorisations are their own sections** (tools; modalities), *before* Methods; Methods **instantiates** their members with cross-references (no re-description).
- **Aims woven into prose, not numbered research questions**; the two taxonomies appear as content, never flagged as questions or "axes."
- **Abstract carries no counts** (only the three levels MEC/EXP/EXG) + the two headline findings.
- **MERT positioned as a frozen/probed representation** (deep-family exemplar), explicitly not a benchmark competitor.
- **Predictive validation (LOOCV) framed as supportive → appendix**, not a leaderboard.

Methods subsections drafted: study design & graded manipulation · stimuli/corpus · participants & questionnaire (ring) · emotion measurement & ground-truth · **response normalization** · feature extraction (CREPE / Essentia / MERT as parallel blocks; **the DTW timing block now needs writing up as a fourth parallel block, not as "planned"** — including the honest note that its within-piece family is referenced to the MEC rendition) · analysis pipeline (deltas, dose-response, read-out, cost of explainability). Unknowns (how the three conditions were produced in detail, piece list, audio specs, demographics, exact tag set, author block) are marked as visible placeholders in the `.tex`.

---

## 7. Key conventions / variables (current)

- **Scale 0–6**, `VA_MID=3.0`, `NEUTRAL_RADIUS=0.75`, `MERT_LAYERS=[5,6,7]`.
- `df_long` — long-format per-rating (parse, §1.1); gains `valence_norm/arousal_norm` in §1.1b.
- `df_agg` / `df_all` — per-excerpt aggregated. **`valence_mean/arousal_mean` are now the NORMALIZED means** (ground truth); `valence_mean_raw/arousal_mean_raw` kept in parallel.
- `emotion_labels` — VA-quadrant-derived (from normalized means); **this becomes `Y`**. `top_tags` — raw tags, descriptive only (§1.5).
- `Y` (60×N), `mlb.classes_` = class names; `df_expr` — modelled-excerpt frame.
- **Forwarding:** `unsup_forward_select(...)`; `MERT_FORWARD` / `CREPE_FORWARD` / `ESSENTIA_FORWARD`; forwarded frames `emb_mert_fwd`, `df_crepe_fwd`, `df_ess_fwd`. Full blocks on disk: `df_mert`, `df_crepe`, `df_essentia`.
- **Assembly:** `FEAT_A/B/E/DTW/C/D/F`, `X_A/X_B/X_C/X_D/X_F`, `X_MERT_FULL`; `STRICT_BLOCKS=('MERT','Essentia')` (DTW non-strict but a missing row warns loudly); `bfv(...)` builder.
- **Interpretable frame & labels** used by Part 5: `df_interp` + `INTERP_FEAT_LABELS` (CREPE + forwarded Essentia + forwarded DTW, built in §5.1; §5.1b / §5.2 / §5.4 / §5.5 all derive their technique set from `INTERP_FEAT_LABELS ∩ df_interp.columns`).
- Part 5 outputs: `df_dose` (§5.1b) · SHAP importances (§5.2) · read-out coefficients (§5.4) · CoE table (§5.5).
- Excerpt IDs `PieceName_ConditionCode`; participant IDs `questionnaire_{n}_P{idx:03d}` (alias `S{n}_P{idx:03d}`).
- `OUTPUT_DIR` — all figures/CSVs. Skip stubs at §2.4: madmom, MusiCNN (DTW is active).
- **DTW constants:** `DTW_SR=22050`, `DTW_HOP=512`, `DTW_TG_WIN=384`, BPM band 30–300, chroma-CQT alignment; `DTW_REFFREE` / `DTW_DEV` name the two families.
- **Phase A constants:** `SEED=42`, `N_DUMMY_REPEATS=200`, `WP_N_PERM=500` (§0.3); `PRIMARY_PROTOCOL='piece'`, `PROTOCOLS=('piece','loo')`, `RIDGE_ALPHAS=logspace(-2,4,13)`, inner `GroupKFold(5)`, `DEFAULT_CLF_NAME='NaiveBayes'` (§4.0); `TUNE_GRID_C={'C': logspace(-3,2,6)}` (§4.1); probe `ALPHAS=[1,10,100,1000,10000]` (§6.1b); MERT cache `emb_mert_v2.npz`; `OUTPUT_DIR='/kaggle/working/pipeline_outputs'`.
- **Phase A result containers:** `all_recs` (with `protocol`), `DEFAULT_RECS[(set, protocol)]`, `REG_RES[protocol]` (`reg_res` = primary), `PRED_QUADS`, `INTERP_CV_R[protocol][dim]`, `INTERP_CV_F1[protocol]`, `COE[protocol]`, `BLOCK_F1[protocol]`, `df_within_piece`, `df_delta_tests`, `df_reclevel`, `df_inner_skips`; ablation rows gain `r2_val/r2_aro/rmse_val/rmse_aro/beats_chance/classifier`.

---

## 8. Workflow

**Repo:** `makris2003/violin-emotion-thesis` (private). GitHub = source of truth; Kaggle = GPU execution. `CLAUDE.md` at repo root is auto-read by Claude Code. Loop: edit locally (VS Code) → commit/push → Kaggle pulls → run on GPU → commit results back. Claude Code **cannot run** the notebook (no audio/GPU) → surgical edits + static checks + local synthetic tests only (the §4.0 `# >>> HELPERS:cv_engine` block is self-contained for that purpose). The notebook is stored as single-line JSON, so review changes cell by cell (e.g. nbdime) rather than with a plain `git diff`. Kaggle "Pull from GitHub" can stale its OAuth token (unlink/relink to fix).

---

## 9. Quick answers

- **"Why is valence weak?"** Intrinsic to MER (needs harmony/context). Arousal is clean with the same features → it's a finding. To *explain* (not rescue) it, add 1–2 harmonic features (mode / tonal tension) so you can say "valence is governed by harmony, which the graded expressivity holds fixed."
- **"Did normalization change results?"** Not materially — raw↔norm r = 0.996 (V) / 0.993 (A); 10/60 excerpts change class (mostly across the Neutral radius), 4/60 change quadrant. That's the point: it confirms the findings aren't a rater artifact (defensibility, not correction).
- **"Why remove MFCC?"** Degenerate (F1≈0), indistinct from the Dummy floor, out-of-topic. Dummy floors are the official baseline.
- **"Is it a leaderboard (MERT vs CREPE)?"** No — that framing is de-emphasised. It's a categorisation + interpretable-attribution study; cost of explainability is the comparison, not "who wins."
- **"Why forward features up front unsupervised?"** To keep the CV leakage-free — a supervised selection would leak held-out folds. The gate never sees `Y`.
- **"Why leave-one-piece-out as the primary protocol?"** 95% (V) / 91% (A) of the excerpt variance is between pieces; LOOCV keeps a held-out excerpt's two sibling renditions in training, so it mostly measures piece recognition.
- **"Why GaussianNB as the default?"** Declared before the first piece-wise run so it cannot be a post-hoc pick: no hyper-parameters to select at N=60, deterministic, cheap enough for every ablation row. Its downsides are stated in §4.0; §4.1 shows the sensitivity to the learner.

---

## 10. Audit (Oct 2026)

A full audit of the 2026-08-16 run's outputs produced three phases of corrections.

- **Phase A — pipeline corrections (APPLIED; results pending the next Kaggle run).** A-shared §4.0 CV engine · **A1** DummyClassifier floors were re-created with one seed in every fold, so `stratified`/`uniform` predicted the same constant everywhere (F1-macro = 0); now seeded per fold × label, 200 repeats, mean/SD/P95 · **A2** leave-one-piece-out PRIMARY everywhere, nested piece-grouped ridge instead of RidgeCV, §5.5 handles a negative CoE, new §4.9 within-piece prediction · **A3** MERT's positional-convolution weight-norm tensors were never loaded (checkpoint `weight_g/_v` vs model `parametrizations.weight.original0/1`) — the conv ran with random weights; now restored + asserted, cache `emb_mert_v2.npz`; **every MERT-based number from before is superseded** · **A4** GaussianNB as the single pre-declared default; every best-of selection removed (§4.5 headline, §4.6 best overall, §4.8, §7.1) · **A5** class balancing + per-label inner-CV C for SVM/LogReg, balanced RF/ET, XGBoost `scale_pos_weight`; tier `4D-balancing` · **A6** out-of-sample R² and RMSE next to every r, R² permutation p in the ablation BH family · **A7** MERT probe with GroupKFold(5) over pieces and nested α (grid to 1e4) · **A14** BH-FDR over all 174 §5.1 tests, stars from q · **A-QC** §2.3c recording-level sanity check (report only).
- **Key fact behind A2:** 95% of valence variance and 91% of arousal variance across the 60 excerpts is between-piece; the condition explains 1% (V) / 4% (A).
- **Phase B (deferred):** descriptors that measure something other than their name (see §5 item 10).
- **Phase C (deferred idea):** augmentation (see §5 item 11).

### What to check after the Phase A Kaggle run

- [ ] §2.1: `loading info` shows exactly the two pos-conv keys missing/unexpected, then `✅ MERT pos_conv weight-norm parameters restored from checkpoint` (no RuntimeError). Transformers' own "newly initialized … pos_conv_embed" warning is printed *inside* `from_pretrained`, before the restore, so it still appears — the restore message + equality assert are the evidence.
- [ ] §2.3c: any recording-level flags; per-condition LUFS vs noise floor.
- [ ] §4.0: the GaussianNB equivalence assert passes; Dummy predictions vary across folds.
- [ ] §4.5b: LOOCV uniform floor ≈ 0.27, stratified ≈ 0.20; which default-classifier rows beat the strict P95 under `piece`.
- [ ] §4.4b lattice assert, §4.4d label-reproduction assert, §6.2b guard (both protocols) all pass.
- [ ] Tier `4D-balancing`: does balancing help under `piece`? (descriptive — the default stays GaussianNB).
- [ ] §4.9: does any feature set beat Condition-only? (expectation to verify: arousal may, valence probably not).
- [ ] §5.5: CoE sign under `piece` (headline and both separate gaps).
- [ ] §4.1 `df_inner_skips`: how often inner-CV tuning was skipped (A5 edge case).
- [ ] §6.1b: how often α hits the new grid edge (1e4).