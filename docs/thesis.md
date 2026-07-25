# Thesis Handoff — Violin Expressiveness → Emotion Recognition

**Purpose:** complete standalone context to continue work in a fresh conversation. Covers (1) the original project scope, (2) pipeline architecture, (3) all progress to date, (4) current results, (5) workflow, (6) open issues.

> **Revision status (current):** Includes the **0–6 scale correction**, the **Neutral/MEC de-confounding fix**, and the re-run results that followed. Numbers in §5 are from the current notebook run and supersede all earlier figures.

---

## 1. Project Overview

### 1.1 Original thesis brief (the authoritative scope)

> Σκοπός της εργασίας είναι να μελετηθούν **τεχνικές που προσδίδουν εκφραστικότητα σε μουσική ερμηνεία** (πχ βιμπράτο, χρονικές/τονικές παραλλαγές, διανθίσματα κλπ), και η **επίδρασή τους πάνω στο ακουστικό αποτέλεσμα** και κατ' επέκταση **στο κοινό**.
>
> Θα πρέπει να γίνει μια **κατηγοριοποίηση των εργαλείων** για τον εντοπισμό των τεχνικών (signal processing, DNNs, αλγόριθμοι, κ.ά.) αλλά και μια **κατηγοριοποίηση των modalities** τα οποία μπορούν να δείξουν την επίδρασή τους στο κοινό (valence-arousal metrics, annotations, κ.ά).

This brief defines **three deliverable axes**. See §6.1 for a scope-alignment audit — this matters, because the implementation has drifted narrower than the brief.

### 1.2 Operational research questions

1. Do SSL embeddings (**MERT**) outperform hand-crafted low-level features (**CREPE**) for predicting *perceived* emotion from expressive solo violin?
2. Does **combining** both outperform either alone?
3. **Which performance techniques are associated with which emotions?**

### 1.3 Dataset

- **60 excerpts** = **20 pieces × 3 performance conditions**
  - `MEC` = mechanical (no expression, baseline)
  - `EXP` = expressive (natural)
  - `EXG` = exaggerated (over-the-top expressive)
- **118 participants** across 5 listening sessions rated each excerpt on **valence (0–6)**, **arousal (0–6)**, and tagged perceived emotions.
- **Scale is 0–6 (7 points, integers 0…6, midpoint 3.0).** `0` is a **genuine lowest rating**, verified against raw forms (a `0` co-occurs with normal arousal values and filled-in emotion tags, e.g. valence=0 on sad excerpts). It is **not** a skip/NA marker. `VA_MID = 3.0` is therefore **correct** — do not "fix" it to 3.5.
- 8 emotion labels derived from valence/arousal circumplex quadrants: Tenderness, Nostalgia, Peacefulness, Power, Joyful Activation, Tension, Sadness, Neutral. **(But see §6.2 — these are mechanically only 5 classes.)**

**Environment:** Kaggle notebook, P100 GPU. PyTorch pinned to 2.3.1 (last version supporting P100's sm_60).

---

## 2. Pipeline Architecture

`notebook/thesis_pipeline.ipynb` runs top to bottom (86 cells, clean execution 1→N, no errors):

| § | Content |
|---|---|
| 1–3 | Install / imports / config (`VA_MID=3.0`, `NEUTRAL_RADIUS=0.75`, `MERT_LAYERS=[5,6,7]`, `EMOTION_LABELS`, `QUADRANT_EMOTIONS`) |
| 4 | Parse 5 questionnaire CSVs → long-format responses (keeps `0` ratings) |
| 5 | Aggregate per excerpt; VA means/SDs; quadrant assignment; `derive_labels` |
| 6 | Label matrix `Y` (60×8) |
| **6A** | **Statistical emotion analysis** (participants only) — Note 5 |
| **6B** | **Lift analysis** (emotion↔condition binding, raw tags) — Note 6 |
| 7 | **MERT** embeddings — 768-dim, averaged over hidden layers **[5,6,7]** (mid-stack) |
| 8 | **CREPE** — 8 interpretable pitch features |
| **8A** | **Dominant-frequency demo** — Note 1 |
| 9–12 | Skip stubs: Madmom, DTW, Essentia DSP, MusiCNN |
| 13 | Feature assembly: **A**=MERT (60×768), **B**=CREPE (60×8), **C**=Combined (60×776) |
| **13A** | **EDA** (4 groups) — Note 7 |
| 14 | Classification: 8 classifiers × 3 feature sets, LOOCV, multi-label |
| 15 | VA regression (Ridge/SVR/MLP) — *MERT only* |
| 16 | MFCC baseline |
| 17 | Ablation table + Wilcoxon |
| **17A** | **Expanded per-label metrics + Dummy baselines** — Note 3 |
| 18–21 | Analyses A–D: per-condition F1, quadrant accuracy, Krippendorff's α, embedding-distance attribution |
| **E1** | **Condition-Delta (Path B)** — technique change vs emotion change |
| **E2** | **SHAP technique importance (Path A)** |
| **E3** | **MERT ↔ CREPE correspondence** (correlation + linear probe) — Note 2 |
| **E4** | **Feature selection** (leakage-free k-sweep + stability) — Note 4 |
| 22–23 | Figures + CSV export |

### 2.1 How §E3 works (frequently misunderstood)

MERT does **not** "look for" vibrato or F0 range — it has no concept of them. §E3 trains a **linear probe**: a regression that tries to predict each *known* CREPE feature **from** the 768 MERT dimensions. CREPE supplies the ground-truth answer; the probe asks "is this information recoverable inside the embedding?" High R² → MERT encodes it. R²≈0 → it doesn't.

**Consequence:** CREPE is not kept for *performance* (it loses badly). It is the **interpretability instrument** — the translator turning 768 opaque numbers into musical concepts. Remove CREPE and §E1, §E2, §E3, §E4 all cease to exist, along with research question 3.

---

## 3. Progress History

### 3.1 Removed all librosa fallbacks
Replaced silent `try/except → librosa` degradation with **hard imports that fail loudly**. Silent degradation produced wrong-dimensional features and made debugging impossible.

### 3.2 Diagnosed essentia TF-models warning
`TensorflowPredictMusiCNN` lives only in `essentia-tensorflow`, not DSP-only `essentia`. Fix: `pip install essentia-tensorflow --no-deps` (the `--no-deps` avoids pulling a TensorFlow that conflicts with pinned PyTorch).

### 3.3 Reduced to validation mode (MERT + CREPE only)
Cut to one SSL + one low-level extractor to validate end-to-end. **Madmom** (timing/rhythm), **DTW** (alignment), **Essentia DSP** (timbre/loudness), **MusiCNN** (2nd DNN embedding) replaced with skip stubs — **not deleted**; re-enabling means restoring those cells and expanding the feature-assembly lists.

### 3.4 Removed the Neutral Gate
Originally the 20 MEC excerpts were excluded from classification and auto-assigned Neutral. Gate removed → all 60 excerpts classified, N raised 40→60 for LOOCV.

### 3.5 Fixed classification crash (SafeMultiOutputClassifier)
`ValueError: number of classes has to be greater than one` in LOOCV folds where a rare label had 0 positives. **`SafeMultiOutputClassifier`** fits one binary classifier per label and predicts majority class (0) for degenerate single-class columns. Scientifically correct: "no positive evidence → predict absent."

### 3.6 Split cleaning into two standalone scripts

**`questionnaire_cleaning.py` — participant screening.** 5 methods: **Robust Mahalanobis (MCD)** (primary, supervisor-recommended; 40-dim space = 20 valence + 20 arousal; threshold √χ²(40,0.001)=8.57), classical MD, **leave-one-out person-total correlation**, **IRV**, **LongString**. Excluded only if flagged by **≥2 independent methods** (classical MD excluded from the vote as redundant).

**`excerpt_cleaning.py` — stimulus QC.** Replaced v1's mathematically-invalid per-column "ICC" with correct diagnostics: rater coverage, dispersion, rWG agreement, tag/valence entropy, bimodality, **one-way panel ICC(1)/ICC(1,k)** (correct model — raters are not crossed with targets), and a per-piece **manipulation check** (does mean arousal follow MEC ≤ EXP ≤ EXG?). **Flags, never auto-removes.**

### 3.6b Fixed the 0–6 scale bug in the cleaning scripts
Both scripts assumed **1–6** (`VA_SCALE_MIN=1`) and silently converted every `0` to `NaN` — deleting ~3–4% of real data concentrated on low-valence excerpts and bottom-of-scale participants. The excerpt script also used `N_SCALE_POINTS=6` in the rWG null variance (should be 7 → `(7²−1)/12 = 4.0`, not 2.92), which understated the null and pushed all rWG values down.

**Fixed:** `VA_SCALE_MIN=0`, `N_SCALE_POINTS=7`, histogram bins from 0. **Effects:** flagged-participant list changed (only `S5_P012` stable), contested excerpts **54 → 46**, inversions **4 → 3**. **The main pipeline was never affected** — its parse always kept `0` and `VA_MID=3.0` was already correct.

### 3.7 Technique–Emotion Association (the core thesis question)
**Path A — SHAP (§E2):** per emotion, RandomForest (400 trees, balanced) on the 8 CREPE features + SHAP TreeExplainer → 8×8 emotion×technique importance matrix.
**Path B — Condition-Delta (§E1):** per piece, technique change MEC→EXP and EXP→EXG correlated with listener ΔValence/ΔArousal (within-subject evidence).
**Why both:** Path B = performer's side ("when I added vibrato, did listeners feel it?"); Path A = model's side ("which features predict each emotion?"). Convergence = evidence.

### 3.8 Statistical analyses (§6A, §6B)
**§6A:** per-condition VA means/SDs, emotion frequency, **Friedman** on piece-level condition means (repeated-measures) with Kendall's W, post-hoc Wilcoxon + Bonferroni.
**§6B:** `lift = P(emotion|condition) / P(emotion overall)` using **raw participant tags** (not VA-derived labels — avoids circularity), threshold-based incidence (≥30%), sensitivity sweep [0.20, 0.30, 0.40] + threshold-free mean-share check.

### 3.9 Implemented remaining supervisor notes
§8A (Note 1), §13A (Note 7), §17A (Note 3), §E3 (Note 2), §E4 (Note 4) — all present and executed.

### 3.10 ★ Removed the Neutral/MEC confound (major correction)

**The problem:** `derive_labels` force-assigned `Neutral` to every MEC excerpt by rule:
```python
return ['Neutral'] if row['condition']=='MEC' else QUADRANT_EMOTIONS.get(row['quadrant'],['Neutral'])
```
So `Neutral` ≡ MEC condition. Since MEC is acoustically very distinct (flat dynamics, no vibrato), a large share of the headline score was **mechanical-condition detection**, not emotion recognition.

**Why the fallback couldn't save it:** `QUADRANT_EMOTIONS` contains all 4 quadrants, so `.get(quadrant, ['Neutral'])` **never fires**. Removing the MEC rule alone would drop Neutral from 20 → **0** excerpts, leaving an all-zero column that drags F1-macro down artificially.

**The fix — a genuine, condition-independent Neutral rule:**
```python
NEUTRAL_RADIUS = 0.75   # in config
def derive_labels(row):
    d = np.hypot(row['valence_mean'] - VA_MID, row['arousal_mean'] - VA_MID)
    if d <= NEUTRAL_RADIUS:
        return ['Neutral']
    return QUADRANT_EMOTIONS.get(row['quadrant'], ['Neutral'])
```
An excerpt is Neutral if its mean VA sits near the centre (3.0, 3.0) — **any condition can be Neutral**.

**Also removed:** dead line `df_mec = df_agg[df_agg['condition']=='MEC']...` (commented "kept for Analysis D" — false; Analysis D builds its own dict from `df_all`. Fossil from the old Neutral Gate era).

**⚠️ `NEUTRAL_RADIUS = 0.75` is an un-tuned free parameter.** It was chosen geometrically (max possible VA distance from centre is √(3²+3²)≈4.24; 0.75 ≈ half a scale-point in each dimension), **not** derived from the data. Needs a sensitivity analysis — see §6.3.

---

## 4. Current Pipeline Status

| Component | Status |
|---|---|
| MERT embeddings (SSL, 768-dim, layers [5,6,7]) | ✅ Active, all 60 excerpts |
| CREPE features (8-dim, interpretable) | ✅ Active |
| Label matrix Y (60×8) | ✅ All 60 excerpts, Neutral de-confounded |
| LOOCV classification (8 classifiers × 3 sets) | ✅ Working |
| VA regression | ⚠️ Active but **MERT only** — no CREPE/Combined/baseline |
| MFCC baseline | ⚠️ Degenerate (F1-macro 0.0) |
| Participant cleaning | ✅ 0–6-corrected; 3 flagged; **decision pending** |
| Excerpt QC | ✅ 0–6-corrected; 46 contested, 3 inversions |
| §6A, §6B, §8A, §13A, §17A, §E1, §E2, §E3, §E4 | ✅ All done |
| Madmom / DTW / Essentia DSP / MusiCNN | ⏭ Skip stubs |
| **Participant exclusions** | ❌ **Not applied** — pipeline runs on all 118 |

---

## 5. Current Results (post-fix, all 118 participants, no exclusions)

### 5.1 Headline

| Feature set | Best F1-macro | Best classifier |
|---|---|---|
| **A: MERT** | **0.605** | SVM-Linear |
| B: CREPE | 0.344 | — |
| C: Combined | 0.589 | — |

- **Wilcoxon:** MERT > CREPE **p=0.008**; Combined > CREPE **p=0.008**; MERT vs Combined **n.s. (p=0.109)** → **MERT ≈ Combined ≫ CREPE**
- **VA regression** (Ridge, MERT): valence **r=0.863**, arousal **r=0.863** (unchanged by the fix — label-independent)
- **Quadrant accuracy** (Ridge VA → circumplex): **0.70**

### 5.2 ★ The de-confounding evidence (most important result)

| | Before fix | After fix |
|---|---|---|
| Neutral excerpts | 20 (all MEC) | **7** (MEC 3 / EXP 2 / EXG 2) |
| MERT F1-macro | 0.736 | **0.605** |
| Neutral per-label F1 | **0.90 (easiest)** | **0.154 (hardest)** |
| Neutral Krippendorff α | 0.846 (highest) | **0.051** |
| Analysis A: MEC per-condition F1 | 0.11 (artifact) | **0.50** |

**Interpretation:** the F1 drop **is the confound leaving**, not the model worsening. The Neutral inversion (easiest → hardest) proves the old model was detecting *mechanical performance*, not *emotion*. 0.605 is the honest emotion-recognition number. Analysis A is now meaningful: EXG 0.70 > EXP 0.61 > MEC 0.50 (stronger expressive signal = easier).

### 5.3 Per-label F1 (§17A)

Tenderness **0.800** ≡ Peacefulness 0.800 · Power **0.692** ≡ Joyful Activation 0.692 · Tension **0.59** · Nostalgia **0.560** ≡ Sadness 0.560 · Neutral **0.154**

*(The `≡` pairs are byte-identical — see §6.2.)*

### 5.4 EDA (§13A)

- **Label counts:** Tension 20, Power 13, Joyful 13, Nostalgia 12, Sadness 12, Tenderness 8, Peacefulness 8, Neutral 7. No class <6 → no LOOCV-degenerate risk.
- **Trivial baseline:** predict-all-absent Hamming loss **0.1938**.
- **Quadrant occupancy:** LV/HA (Tension) 23, LV/LA 15, HV/HA 14, HV/LA 8.
- **Responses per condition:** MEC 944, EXP 944, **EXG 472** (half — by design).
- **CREPE sanity:** 0/60 NaN, 0/60 unreliable. vibrato_rate mean 4.92 Hz but **0/60 read zero** (bandpass-bounded — see §6.5). portamento 21/60 zero (plausible).
- **MERT structure:** 22 PCA components ≥90% variance; **silhouette = −0.030** → MERT does **not** cluster by condition unsupervised. Information exists but needs supervision.

### 5.5 §6A / §6B (participant-only, most defensible)

- **§6A:** valence Friedman **n.s. (p=0.39)**; arousal **significant (p=0.001)**, driven by MEC↔EXG. → *the manipulation moves arousal, not valence.*
- **§6B lift:** only **Neutral↔MEC** is threshold-stable. Its value is **3.00 / 0.00 / 0.00** — this is the **ceiling** (= n_conditions = 60/20), reached whenever an emotion appears exclusively in one condition. It is a **genuine perceptual finding from raw tags** (listeners call only mechanical performances "neutral"), distinct from the labelling confound that was fixed.

### 5.6 §E1–§E4

- **§E1 (Path B):** mostly null. 48 correlations tested; 3 nominal hits (p 0.016–0.037, uncorrected). **None survive FDR.**
- **§E2 (Path A, SHAP):** now dominated by **pitch** features (F0 Mean / Range / CV).
- **§E3 (probe):** MERT strongly encodes **F0 Mean R²=0.97**, **F0 CV R²=0.80**; weakly vibrato depth (0.11); **does not** recover F0 Range or vibrato rate (R²≈0).
- **§E4:** **k=50** matches full performance; adding CREPE changes F1 by **−0.002**. ⚠️ **CREPE selection-stability collapsed 98% → 0%** after the fix.

### 5.7 ★ Narrative change: vibrato → pitch

Before the fix, vibrato_depth survived 98% of folds and looked like the key interpretable feature. **This was an artifact:** old Neutral = MEC = "no vibrato", so "detect vibrato" ≈ "detect Neutral". After de-confounding, **all CREPE features drop to 0% survival** and SHAP shifts to pitch features.

**The honest, cleaner story:** pitch-based features carry what interpretable signal exists — and §E3 shows MERT *already encodes exactly those* (F0 Mean 0.97, F0 CV 0.80). **This directly explains Combined ≈ MERT.** Build the feature chapter on this. **Retract any vibrato-depth claim.**

### 5.8 Cleaning results (0–6-corrected)

- **Participants (118):** per method — robust MD 30, classical MD 1, person-r 4, IRV 1, LongString 4. **3 flagged (≥2 methods):** `S5_P012` (MD-rob, MD-std, person-r), `S5_P021` (MD-rob, LongString), `S1_P006` (IRV, LongString). **Not auto-removed.**
- **Excerpts:** valence ICC(1,k)=**0.963**, arousal **0.952** (single-rater 0.396/0.335, avg k≈39). **46/60** flagged low-agreement (diagnostic only). **3 arousal inversions** — the manipulation-check candidates:
  - **Bach_Adagio −0.53** · **Mendelssohn_VC_Mvt2 −0.17** · **La_Vita_E_Bella −0.13**
  - **12 of 20 pieces** show no statistically distinguishable arousal across conditions (Kruskal–Wallis p>0.05).

### 5.9 Reproducibility note
Two independent re-runs produced **byte-identical numeric results** (only tqdm timings and log timestamps differed). Mild positive evidence that the MERT `pos_conv` init warning does not introduce run-to-run variance — but still pin versions (§6.6).

---

## 6. Open Issues

### 6.1 ★ Scope alignment audit (~7.5/10)

Assessed against the three axes of the original brief (§1.1). **Not off-topic — but narrowed to one deep slice at the expense of breadth.**

| Axis | Rating | Assessment |
|---|---|---|
| **A. Techniques → acoustic → audience** | **8/10** ✓ | Strong core. CREPE covers vibrato, portamento (διανθίσματα), F0 range (τονικές παραλλαγές); MEC/EXP/EXG covers acoustic effect; 118 participants + VA + tags covers audience. **But:** χρονικές παραλλαγές (rubato/tempo) — *named explicitly in the brief* — are **entirely absent** (that's madmom/DTW). Dynamics missing (Essentia). Coverage has narrowed to the **pitch domain only**. |
| **B. Tool taxonomy (SP / DNN / algorithms)** | **5/10** ⚠️ | Weakest. §E3 (SSL vs hand-crafted) *is* a category comparison — but with only **2 of 6** extractors running, a "taxonomy" rests on one tool per category. **Check whether the written thesis has an explicit taxonomy chapter.** |
| **C. Modality taxonomy** | **6/10** | The chosen modality (VA + annotations) is exactly what the brief names. But the brief asks for a *categorisation* of modalities (dimensional / categorical / self-report / physiological / behavioral). Implementing two is fine; the **survey** is a writing deliverable. **Check the text.** |

**The two real scope errors:** (1) focus drifted from descriptive/taxonomic toward **benchmarking** ("does MERT beat CREPE?"); (2) techniques and tools narrowed to pitch.

**Key insight:** re-enabling the extractors is **not** a performance chase — it is **scope realignment**. Essentia → dynamics + timbre (Axis A) + SP category (Axis B). madmom/DTW → *χρονικές παραλλαγές*, the named-but-missing category (Axis A) + algorithms category (Axis B). MusiCNN → strengthens DNN category (Axis B).

### 6.2 ★ "8 emotions" is mechanically 5 classes

**Where it happens:**
1. **Config** — `QUADRANT_EMOTIONS` maps each quadrant to a *fixed set*, never to individual emotions:
   `HA_HV:[Power, Joyful]` · `HA_LV:[Tension]` · `LA_HV:[Tenderness, Peacefulness]` · `LA_LV:[Sadness, Nostalgia]`
2. **`derive_labels`** — therefore returns only **5 possible outputs** (4 quadrant packages + Neutral).
3. **`MultiLabelBinarizer`** — builds 8 columns, but pairs always co-occur → **3 pairs are byte-identical columns**.

**Result:** Tenderness ≡ Peacefulness, Power ≡ Joyful Activation, Sadness ≡ Nostalgia. Identical per-label F1, SHAP rows, and Krippendorff α. No model can ever separate them.

**Options:** (a) report honestly as **5 classes**; (b) rebuild `Y` **directly from raw participant tags** (multi-label) so the 8 become genuinely distinct. **Unresolved — good supervisor question.**

### 6.3 Neutral is now a near-dead class
F1 0.154, α 0.051, 7 scattered excerpts. `NEUTRAL_RADIUS=0.75` was chosen geometrically, not empirically.
**Actions:** sensitivity analysis (0.5 / 0.75 / 1.0); report macro-F1 **with and without** Neutral; consider principled alternatives — set radius so Neutral matches the smallest class prevalence (~8), or define Neutral **perceptually** from rater disagreement / tag entropy (already computed in excerpt QC) instead of a radius. Or drop Neutral → clean 7-emotion design.

### 6.4 Statistics
- **§E1 needs FDR** (Benjamini-Hochberg). Better: restrict to a pre-registered hypothesis set from Path A, and/or use **mixed-effects** (piece as random effect) for power. Note honestly that the weak valence manipulation means part of the null is real.
- **§17 Wilcoxon over 8 classifiers is not population inference.** Phrase as "across the eight classifiers tried…". Also a cosmetic bug: labels print as `T vs E` (uses `s1[-1]`) — print full set names.
- **§E4 "best k" and §E3 probe alpha are selected on the same CV they report** (mild optimism). Nest, or phrase as "k=50 was best in this sweep."

### 6.5 Modelling / feature hygiene
- **SVM-RBF underperforms everywhere** (0.275 on MERT, worst) — tune C/gamma or drop.
- **All Dummy baselines = 0.0** (structural — every label is a minority class) → uninformative. Add a **prevalence-aware baseline** and a **condition-detection floor**. MFCC baseline still degenerate (use linear kernel).
- **Regression is MERT-only** → add CREPE / Combined / mean-predictor for parallelism with the classification ablation.
- **`crepe_portamento_count`** is a raw count → confounds with duration; normalise per second.
- **`crepe_vibrato_rate_hz`** is bandpass-bounded to [4,8] Hz → fires even on no-vibrato excerpts (0/60 read zero; probe R²≈0). Gate on vibrato depth; depth is the trustworthy one.
- **Vibrato/portamento computed on concatenated voiced frames** → `filtfilt` sees discontinuities, inflating portamento counts. Analyse within time-contiguous voiced segments.

### 6.6 Reproducibility / environment
- MERT load warns `pos_conv_embed.weight_g/weight_v` "not used" / `parametrizations.weight.original0/1` "newly initialized" (torch weight-norm reparametrization mismatch). Empirically stable across two runs, but **pin `transformers`/`torch`**, add `torch.manual_seed`, verify the load. `nnAudio` missing → MERT CQT front-end disabled.
- **`warnings.filterwarnings('ignore')` globally** hides convergence/numerical warnings — filter by category.

### 6.7 Existing fallback mechanisms (audit)

**Deliberate & correct — keep:**
- `SafeMultiOutputClassifier` majority-class fallback for single-class folds (§14).
- Matching degenerate-fold fallback for Dummy baselines (§17A).
- SHAP version-compatibility fallback (list / 3-D array / plain) (§E2).

**Silent — be careful:**
- ⚠️ **Feature assembly (`bfv`): missing MERT embedding or missing CREPE row → `np.zeros(...)`.** Never fires now (all 60 present), but **when re-enabling extractors a failed feature becomes a silent zero-vector**. Make it warn loudly — consistent with the stated "hard imports, no fallbacks" policy.
- Parse: unparseable rating → `NaN` (no range check, so `0` correctly survives).
- `QUADRANT_EMOTIONS.get(q, ['Neutral'])` — default never fires (all 4 quadrants present).

**Graceful error handling:** MERT extraction failure → skip + warn; Wilcoxon failure → print + continue; §8A demo wrapped in try/except; EDA t-SNE label fallback.

---

## 7. Next Steps (priority order)

1. **§E1 FDR correction** — free, immediate, fixes a real statistical claim.
2. **Write up the 8→5 collapse and the Neutral with/without macro-F1** — pure writing, no code.
3. **Switch the interpretable-feature narrative from vibrato to pitch** (§5.7).
4. **★ Re-enable Essentia** (loudness + timbre + tonal) — see §8.
5. **Regression baselines** (CREPE/Combined/mean) + drop/tune SVM-RBF.
6. **Review the 3 flagged participants** with supervisor; decide exclusions; wire into the §4 parse.
7. **Then madmom (onsets) + DTW** for χρονικές παραλλαγές, then MusiCNN as external baseline.
8. Verify MERT pos_conv load; pin versions; normalise portamento count.

---

## 8. Standing Recommendation: Re-enable Essentia

**The gap:** CREPE is **pitch only**. No **dynamics/loudness** and no **timbre** — both central to expressive performance.

**Three arguments:**
1. **Covers a missing expressive dimension.** Dynamics (crescendo, accents) is a primary expressive parameter tied directly to arousal; CREPE captures none of it.
2. **Targets the weakest dimension.** In the MER literature, Essentia's **tonal** and **timbre** features specifically improve **valence** prediction — exactly the weak dimension here (§6A: valence n.s.). Expressive vs non-expressive performances also differ in attack slope, onset rate, spectral flux.
3. **An objective acceptance test already exists.** Run any new block through the **§E3 probe** (does MERT already encode it?) **and** the **§E4 within-fold block test** (does it improve F1?). A block passing **both** is genuinely complementary *and* emotion-relevant — and would **break the Combined ≈ MERT tie**, the most interesting open question.

**Extraction sketch:** MonoLoader @16 kHz → loudness (EBU R128 / dynamic range), spectral flux/centroid/rolloff, attack time, dissonance, chroma/key. Mean + std per excerpt, matching the CREPE convention.

**Also note:** re-enabling extractors serves **scope realignment** (§6.1), not just performance — this is the stronger argument for the supervisor.

---

## 9. Workflow Setup

**Repository:** `makris2003/violin-emotion-thesis` (private, GitHub). GitHub = single source of truth; Kaggle = execution environment (GPU + data).

```
violin-emotion-thesis/
├── CLAUDE.md              ← project context, auto-read by Claude Code
├── README.md
├── .gitignore             ← excludes audio, .npz, outputs/
├── notebook/
│   └── thesis_pipeline.ipynb
├── scripts/
│   ├── questionnaire_cleaning.py
│   └── excerpt_cleaning.py
├── data/
│   ├── raw/               ← audio (gitignored)
│   └── forms/             ← questionnaire CSVs (gitignored)
├── outputs/               ← figures/tables (gitignored)
└── docs/
    └── thesis_handoff.md
```

**Tools:** Windows + VS Code at `C:\VScodeProjects\violin-emotion-thesis`; GitHub Desktop; **Claude Code** (CLI + VS Code extension, reads `CLAUDE.md` at session start); Kaggle ↔ GitHub linked.

**Loop:** edit locally → commit/push → Kaggle pulls → run on GPU → commit results back.

**Gotcha:** Kaggle's "Pull from GitHub" overwrites the editor and its OAuth token goes stale ("Failed to fetch file content"). Fix: unlink/relink GitHub in Kaggle settings, or File → Import Notebook → GitHub tab. No SSH on Kaggle.

---

## 10. Key Conventions / Variables

- **Scale: 0–6** (7 points, midpoint 3.0). `VA_MID = 3.0` — **correct, do not change**.
- `NEUTRAL_RADIUS = 0.75` — VA distance from centre defining Neutral (un-tuned, see §6.3)
- `MERT_LAYERS = [5,6,7]` — mid-stack hidden layers, averaged
- Excerpt IDs: `PieceName_ConditionCode` (e.g. `Bach_Adagio_S1_MEC`)
- Participant IDs: `questionnaire_{n}_P{idx:03d}` (alias `S{session}_P{idx:03d}`)
- `df_all` / `df_agg` — per-excerpt aggregated (condition, piece, valence_mean, arousal_mean, emotion_labels, top_tags)
- `top_tags` — raw participant tags; **descriptive only, never becomes `Y`** (used by §6B)
- `emotion_labels` — VA-quadrant-derived; **this is what becomes `Y`**
- `df_crepe` — (60,8) CREPE features · `emb_mert` — {excerpt_id: np.array(768)}
- `Y` — (60,8) binary label matrix; `mlb.classes_` = the 8 emotion names
- `X_A / X_B / X_C` — MERT / CREPE / Combined
- `df_delta` (§E1) · `df_shap_importance` (§E2) · `df_probe` (§E3) · `df_ks` (§E4)
- `SafeMultiOutputClassifier` — LOOCV-safe multi-label wrapper
- `MLSelectKBest` — leakage-safe within-fold multilabel selector (§E4)
- `OUTPUT_DIR` — all figures/CSVs
- Skip stubs: §9 madmom, §10 DTW, §11 Essentia DSP, §12 MusiCNN

---

## 11. Quick Answers (common questions)

- **"Why did F1 drop 0.736 → 0.605?"** A confound left. Neutral was MEC — the easiest label. The number is lower but real.
- **"Why LOOCV?"** Only 60 excerpts; maximises training data, standard for small sets.
- **"Why MERT layers 5–7?"** Mid-stack layers give the best musical representations; consistent with recent MER work.
- **"Isn't VA_MID = 3.0 wrong?"** No — the scale is 0–6, so the centre is 3.0 (verified in raw forms).
- **"What is CREPE exactly?"** 8 interpretable pitch features: vibrato rate/depth, portamento count/extent, F0 range/CV/mean, voiced fraction.
- **"Why keep CREPE if it loses?"** It's the interpretability instrument, not a performance competitor — §E1–§E4 and research question 3 all depend on it.
- **"Why is Neutral lift exactly 3.00/0/0?"** 3.00 is the ceiling (= 60/20 = n_conditions), reached when an emotion appears in exactly one condition. It's a real perceptual finding from raw tags.