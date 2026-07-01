# 🎻 Thesis Progress Handoff — Violin Expressiveness & Emotion Recognition
**Last updated: May 2026 — includes all session progress**

---

## Thesis Overview

**Title:** *Expressive Techniques in Violin Performance and Their Emotional Effect on the Audience*

**Research question:** Do large-model SSL embeddings outperform hand-crafted low-level acoustic features for predicting perceived emotion from expressive solo violin? Does combining both outperform either alone?

### Dataset
- **60 audio excerpts**: 20 musical pieces × 3 expressive conditions
  - `_MEC` = Mechanical (metronome-strict, no expressive intent)
  - `_EXP` = Expressive (natural musical interpretation)
  - `_EXG` = Exaggerated (deliberately over-expressive)
- All recordings are unaccompanied solo violin by the researcher
- 20 pieces cover all 4 quadrants of the **Circumplex Model of Affect**:
  - `HA_HV` → Power, Joyful Activation
  - `HA_LV` → Tension
  - `LA_HV` → Tenderness, Peacefulness
  - `LA_LV` → Sadness, Nostalgia

### Participant Study
- 5 questionnaire forms (Google Forms, exported as wide-format CSV in Greek)
- Each form: 20 excerpts rated on:
  - **Q1** Valence: "How pleasant/positive?" (0–6)
  - **Q2** Arousal: "How intense/energetic?" (0–6)
  - **Q3** Emotion tags: select 1–3 from 8 GEMS-aligned labels
- MEC/EXP excerpts appear in **2 forms each**; EXG excerpts appear in **1 form**
- Pooling is automatic: `groupby('excerpt_id')` in §5 aggregates across forms naturally
- Demographic columns (music background, age group, gender) are **stripped** from pipeline — not used

### Emotion Label Set (GEMS-aligned)
`Tenderness`, `Nostalgia`, `Peacefulness`, `Power`, `Joyful Activation`, `Tension`, `Sadness`, `Neutral`

---

## Key Design Decisions — DO NOT REVERT

| Decision | Rationale |
|---|---|
| Ground-truth labels derived from **participant mean VA** (not Q3 tags) | More theoretically grounded; anchors to Circumplex Model |
| **MEC excerpts → Neutral by rule**, excluded from classifier training | No expressive intent by design; including them inflates accuracy trivially |
| **Multi-label classification** (1–3 emotions per excerpt) | Reflects real perceptual complexity |
| **Chord/key branch from Music2Emo omitted** | Chord extractors produce noise on monophonic audio |
| **LOOCV** (Leave-One-Out Cross-Validation) | N=60 too small for standard k-fold |
| **No fine-tuning of MERT** | N=60 far too small for 95M parameters |

---

## Platform & Environment

| Item | Value |
|---|---|
| Platform | **Kaggle Notebooks** (migrated from Google Colab) |
| GPU | **Tesla P100-PCIE-16GB** (compute capability sm_60 / Pascal) |
| Python | 3.12 |
| PyTorch | **2.3.1** (pinned — last version supporting sm_60; PyTorch ≥ 2.4 dropped sm_60) |
| CUDA | 12.1 |

### Critical: PyTorch must be pinned to 2.3.1
PyTorch ≥ 2.4 dropped sm_60 (Pascal) kernel support. Kaggle pre-installs a newer version. The install cell pins 2.3.1 **before** all other installs to prevent any package from pulling a newer torch.

### Kaggle dataset structure
```
/kaggle/input/datasets/george031218732003/violin-thesis-data/ThesisDataset/
├── Bach_Adagio_S1/
│   ├── Bach_Adagio_S1_EXG_1.wav
│   ├── Bach_Adagio_S1_EXP_1.wav
│   └── Bach_Adagio_S1_MEC_1.wav
├── Bruch_VC1/
│   └── ...
└── ... (20 piece folders total)

/kaggle/input/datasets/george031218732003/violin-thesis-forms/Forms/
├── questionnaire_1.csv
├── questionnaire_2.csv
├── questionnaire_3.csv
├── questionnaire_4.csv
└── questionnaire_5.csv
```

**Important naming notes:**
- Audio files have a `_1` suffix before `.wav` — the pipeline strips this with `re.sub(r'_\d+$', '', stem)`
- 4 piece names differ between old pipeline and Kaggle folder names — the pipeline now uses Kaggle names:
  - `CinemaParadiso` → `Cinema_Paradiso`
  - `DonJuan` → `Don_Juan`
  - `SchindlersList` → `Schindlers_List`
  - `SecretGarden` → `Secret_Garden`

### Cache persistence on Kaggle
`/kaggle/working/` is wiped between sessions. After a full extraction run, save outputs as a new Kaggle dataset (e.g. `violin-thesis-cache`) and re-mount it at `/kaggle/input/violin-thesis-cache/` for future sessions.

---

## Questionnaire Form Structure

Each form contains 20 excerpts in this order (ratio: 8 MEC, 8 EXP, 4 EXG per form):

| Form | Excerpt order (position 1→20) |
|---|---|
| **Form 1** | Bach_Adagio_S1_MEC, Bruch_VC1_EXG, Cinema_Paradiso_EXP, Don_Juan_EXG, Godfather_MEC, La_Vita_E_Bella_MEC, Meditation_Thais_EXP, Mendelssohn_VC_Mvt2_MEC, Mozart_VC3_EXG, Mozart_VC4_MEC, Schindlers_List_MEC, Secret_Garden_EXP, Shostakovich_SQ8_exrpt1_EXP, Shostakovich_SQ8_exrpt2_MEC, Shostakovich_SQ8_exrpt3_EXG, Shostakovich_Sym5_Mvt1_MEC, Shostakovich_Sym5_Mvt3_EXP, Shostakovich_VC1_EXP, Tanzil_Serenade_EXP, Vitali_Chaconne_EXP |
| **Form 2** | Bach_Adagio_S1_EXG, Bruch_VC1_EXP, Cinema_Paradiso_EXP, Don_Juan_EXP, Godfather_EXG, La_Vita_E_Bella_EXG, Meditation_Thais_MEC, Mendelssohn_VC_Mvt2_EXG, Mozart_VC3_EXP, Mozart_VC4_MEC, Schindlers_List_MEC, Secret_Garden_EXP, Shostakovich_SQ8_exrpt1_EXP, Shostakovich_SQ8_exrpt2_MEC, Shostakovich_SQ8_exrpt3_EXP, Shostakovich_Sym5_Mvt1_MEC, Shostakovich_Sym5_Mvt3_MEC, Shostakovich_VC1_MEC, Tanzil_Serenade_EXP, Vitali_Chaconne_MEC |
| **Form 3** | Bach_Adagio_S1_EXP, Bruch_VC1_EXP, Cinema_Paradiso_MEC, Don_Juan_EXP, Godfather_EXP, La_Vita_E_Bella_EXP, Meditation_Thais_MEC, Mendelssohn_VC_Mvt2_EXP, Mozart_VC3_EXP, Mozart_VC4_EXG, Schindlers_List_EXG, Secret_Garden_MEC, Shostakovich_SQ8_exrpt1_MEC, Shostakovich_SQ8_exrpt2_EXG, Shostakovich_SQ8_exrpt3_EXP, Shostakovich_Sym5_Mvt1_EXG, Shostakovich_Sym5_Mvt3_MEC, Shostakovich_VC1_MEC, Tanzil_Serenade_MEC, Vitali_Chaconne_MEC |
| **Form 4** | Bach_Adagio_S1_EXP, Bruch_VC1_MEC, Cinema_Paradiso_MEC, Don_Juan_MEC, Godfather_EXP, La_Vita_E_Bella_EXP, Meditation_Thais_EXG, Mendelssohn_VC_Mvt2_EXP, Mozart_VC3_MEC, Mozart_VC4_EXP, Schindlers_List_EXP, Secret_Garden_MEC, Shostakovich_SQ8_exrpt1_MEC, Shostakovich_SQ8_exrpt2_EXP, Shostakovich_SQ8_exrpt3_MEC, Shostakovich_Sym5_Mvt1_EXP, Shostakovich_Sym5_Mvt3_EXG, Shostakovich_VC1_EXG, Tanzil_Serenade_MEC, Vitali_Chaconne_EXG |
| **Form 5** | Bach_Adagio_S1_MEC, Bruch_VC1_MEC, Cinema_Paradiso_EXG, Don_Juan_MEC, Godfather_MEC, La_Vita_E_Bella_MEC, Meditation_Thais_EXP, Mendelssohn_VC_Mvt2_MEC, Mozart_VC3_MEC, Mozart_VC4_EXP, Schindlers_List_EXP, Secret_Garden_EXG, Shostakovich_SQ8_exrpt1_EXG, Shostakovich_SQ8_exrpt2_EXP, Shostakovich_SQ8_exrpt3_MEC, Shostakovich_Sym5_Mvt1_EXP, Shostakovich_Sym5_Mvt3_EXP, Shostakovich_VC1_EXP, Tanzil_Serenade_EXG, Vitali_Chaconne_EXP |

**CSV column layout:** col 0 = timestamp, cols 1–2 = demographics (stripped), then 20×3 blocks starting at col 3: `[emotion_tags, valence, arousal]` per excerpt, cols −2/−1 = demographics (stripped).

---

## Pipeline Architecture

### Feature Extraction

| Section | Tool | Output dim | Status |
|---|---|---|---|
| §7 | MERT-v1-95M (frozen, layers 5–7, mean-pooled) | 768 | ✅ Working |
| §8 | CREPE (pitch/vibrato/portamento) | 8 | ✅ Working |
| §9 | madmom RNNOnsetProcessor (timing/IOI) | 8 | ✅ Working |
| §10 | librosa DTW (MEC↔EXP/EXG warp path stats) | 6 | ✅ Working |
| §11 | Essentia DSP (loudness/spectral/MFCC 2–13) | 20 | ✅ Working (fixed) |
| §12 | Essentia MusiCNN (DNN embedding) | 200 | ⚠️ See notes |
| §13 | VioMusic CBA (BiGRU encoder) | 256 | ✅ Working (random-init) |
| §14 | Music2Emo (fused embedding) | 768 | ⚠️ See notes |

### Three Feature Sets

| Set | Contents | Research Question |
|---|---|---|
| **A** | MERT + Music2Emo + MusiCNN + CBA | Can SSL embeddings alone predict emotion? |
| **B** | CREPE + madmom + Essentia DSP + DTW | Can low-level acoustic features alone predict emotion? |
| **C** | A + B (combined) | Does fusion outperform either alone? |

### Classification (§16)
- **8 classifiers**: SVM-Linear, SVM-RBF, LogReg, RandomForest, ExtraTrees, KNN, NaiveBayes, XGBoost
- **Strategy**: Binary Relevance via `MultiOutputClassifier`
- **Validation**: LOOCV
- **Primary metric**: Hamming Loss; secondary: F1-macro, F1-micro, Subset Accuracy
- **Statistical test**: Wilcoxon signed-rank across classifiers per feature set

### VA Regression (§17)
Ridge, SVR, MLP on Set A (MERT embeddings) with LOOCV. Pearson r and RMSE against participant means.

### Heuristic Baseline (§18)
MFCC + spectral features (18-dim) + SVM. Comparison point for multi-label classification.

### Analyses
| Analysis | What it tests |
|---|---|
| **A** | F1-macro per condition (EXP vs EXG) — does more expressiveness help? |
| **B** | Circumplex quadrant accuracy (predicted VA quadrant vs participant quadrant) |
| **C** | Krippendorff's α per emotion label (model vs participant-derived GT) |
| **D** | Pearson r between MERT embedding distance (MEC→EXP→EXG) and ΔArousal — **novel contribution** |

---

## Known Issues & Applied Fixes

### ✅ Fixed: §11 Essentia DSP — `Flux.compute requires 1 argument(s), 2 given`
Newer Essentia changed `es.Flux()` API — it no longer accepts a previous spectrum as second argument. Fixed by computing inter-frame flux manually:
```python
# OLD (broken):
if prev is not None: flf.append(fl_a(fs, prev))
# NEW (fixed):
if prev is not None:
    flf.append(float(np.sqrt(np.sum((fs - prev) ** 2))))
# fl_a = es.Flux() line removed entirely
```

### ✅ Fixed: §13 VioMusic CBA — `NameError: VIOMUSIC_WEIGHTS not defined`
Config cell (§2) defined `VIOMUSICWEIGHTS` (no underscore), but §13 referenced `VIOMUSIC_WEIGHTS` (with underscore). Fixed by renaming in §2 to `VIOMUSIC_WEIGHTS = None`.

### ✅ Fixed: MERT — `CUDA error: no kernel image available`
P100 (sm_60) not supported by PyTorch ≥ 2.4. Fixed by pinning `torch==2.3.1` at the top of the install cell using `--index-url https://download.pytorch.org/whl/cu121`.

### ✅ Non-issue: CREPE — `Delay kernel timed out` messages
TensorFlow XLA profiling warnings, not errors. CREPE ran successfully on all 60 excerpts.

### ✅ Non-issue: MERT — weight initialization warnings
Expected messages from parametrized weight normalization (pos_conv_embed). Model functions correctly.

### ⚠️ Pending: §12 MusiCNN — dim=20 instead of 200
`essentia-tensorflow` not installed (conflicts with Kaggle's pre-installed TF). Current output is 20-dim mel fallback instead of 200-dim DNN embedding. **Fix to try next session:**

Add to install cell (Cell 2) after the `essentia` line:
```python
pip_install(['essentia-tensorflow', '--no-deps'], 'essentia-tensorflow (MusiCNN)')
```

If `--no-deps` still fails, use TF-direct loading — replace `extract_musicnn` in Cell 26 with:
```python
def extract_musicnn(fp):
    if ESSENTIA_TF_OK and os.path.exists(MUSICNN_PATH):
        try:
            audio = es.MonoLoader(filename=fp, sampleRate=16000)()
            model = es.TensorflowPredictMusiCNN(graphFilename=MUSICNN_PATH,
                                                output='model/dense/BiasAdd')
            return np.mean(model(audio), axis=0)
        except Exception:
            pass

    if os.path.exists(MUSICNN_PATH):
        try:
            import tensorflow as tf
            y, _ = librosa.load(fp, sr=16000, mono=True)
            mel = librosa.feature.melspectrogram(
                y=y, sr=16000, n_fft=512, hop_length=256,
                n_mels=96, fmin=0.0, fmax=8000.0, power=1.0
            )
            mel_db = librosa.amplitude_to_db(mel + 1e-10).T.astype(np.float32)
            patch_size = 187
            patches = [mel_db[i:i+patch_size]
                       for i in range(0, mel_db.shape[0] - patch_size + 1, patch_size)]
            if not patches:
                pad = np.zeros((patch_size, 96), dtype=np.float32)
                pad[:mel_db.shape[0]] = mel_db[:patch_size]
                patches = [pad]
            patches = np.array(patches, dtype=np.float32)
            with tf.io.gfile.GFile(MUSICNN_PATH, 'rb') as f:
                graph_def = tf.compat.v1.GraphDef()
                graph_def.ParseFromString(f.read())
            with tf.Graph().as_default() as graph:
                tf.import_graph_def(graph_def, name='')
                with tf.compat.v1.Session(graph=graph) as sess:
                    inp = graph.get_tensor_by_name('model/Placeholder:0')
                    out = graph.get_tensor_by_name('model/dense/BiasAdd:0')
                    result = sess.run(out, feed_dict={inp: patches})
                    return np.mean(result, axis=0)  # (200,)
        except Exception as e:
            print(f'  ⚠️  MusiCNN TF-direct failed: {e}')

    print('  ⚠️  MusiCNN: using low-quality mel fallback — Set A results unreliable')
    y, sr = librosa.load(fp, sr=22050, mono=True)
    mel = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=512))
    bsz = 128 // 10
    return np.array([f for b in range(10)
                     for f in [mel[b*bsz:(b+1)*bsz].mean(), mel[b*bsz:(b+1)*bsz].std()]],
                    dtype=np.float32)
```

**Delete the current 20-dim cache** (`emb_musicnn.npz`) before re-running §12 so it re-extracts.

### ℹ️ By-design: §14 Music2Emo — uses MERT as proxy
`amaai-lab/music2emo` HuggingFace repo has no `config.json` — model cannot be loaded. The MERT proxy is **scientifically defensible**: Music2Emo fuses a MERT backbone + chord/key branch; the chord/key branch was already excluded by design for monophonic violin. For solo violin, Music2Emo ≈ MERT backbone anyway. Document in methods section.

### ℹ️ By-design: §13 VioMusic CBA — random-init weights
`VIOMUSIC_WEIGHTS = None`. The CBA BiGRU encoder runs with random initialization. Document in methods section.

---

## Methods Section Notes (to include in thesis write-up)

- **MEC excerpts**: Assigned `Neutral` label by rule; excluded from classifier training; re-enter only in Analysis D (embedding distance)
- **Chord/key branch (Music2Emo)**: Omitted — chord extractors produce noise on monophonic audio; Music2Emo represented by MERT proxy
- **VioMusic CBA weights**: Unavailable; encoder uses random initialization — results for this component should be interpreted as an architectural baseline only
- **MusiCNN**: If 20-dim mel fallback is still active at write-up time, note the degradation and its cause
- **EXP/MEC excerpts receive ~2× more participant ratings than EXG** (appear in 2 forms vs 1) — mean VA is still a valid ground truth; report `n_responses` per excerpt
- **LOOCV**: Chosen because N=60 is too small for standard k-fold; maximises training data per fold
- **MERT not fine-tuned**: N=60 insufficient for 95M parameters; frozen layers 5–7 used as fixed feature extractors

---

## Session Progress Summary

| Task | Status |
|---|---|
| Migrated from Google Colab to Kaggle | ✅ Done |
| Fixed Kaggle dataset paths & audio loader (subfolder structure, `_1` suffix) | ✅ Done |
| Fixed 4 piece name mismatches (CamelCase → underscore) | ✅ Done |
| Implemented 5-form questionnaire structure with pooling | ✅ Done |
| Stripped demographics from questionnaire parsing | ✅ Done |
| Fixed PyTorch sm_60 / P100 CUDA error (pinned to 2.3.1) | ✅ Done |
| Fixed Essentia Flux API change (§11) | ✅ Done |
| Fixed VIOMUSIC_WEIGHTS variable name mismatch (§13) | ✅ Done |
| Removed CREPE pYIN fallback (CREPE confirmed working) | ✅ Done |
| Improved MusiCNN with TF-direct fallback strategy (§12) | ✅ Coded, pending test |
| Improved Music2Emo error messaging (§14) | ✅ Done |
| Completed full extraction run §7–§14 | ✅ MERT(768), CREPE(8), madmom(8), DTW(6), DSP(20*), MusiCNN(20*→200 pending), CBA(256), M2E(768 proxy) |

*20-dim fallback active — fix pending as described above*

---

## Pending Items

- [ ] **Fix MusiCNN to 200-dim** (try `essentia-tensorflow --no-deps`, then TF-direct)
- [ ] **Add remaining questionnaire CSVs** (Forms 2–5) when data collection completes
- [ ] Run full classification pipeline (§16) once all questionnaire data is in
- [ ] Run VA regression (§17)
- [ ] Run all 4 analyses (§19–§23)
- [ ] Generate all 8 figures (§24)
- [ ] Save final outputs as Kaggle dataset for cache persistence
- [ ] Thesis write-up / chapter structure
- [ ] Visualisation / presentation polish

---

## Tools & Models Referenced

| Tool | Reference |
|---|---|
| MERT-v1-95M | Li et al., ICLR 2024 — `m-a-p/MERT-v1-95M` |
| Music2Emo | Kang & Herremans, arXiv 2502.03979 (2025) — `amaai-lab/music2emo` |
| VioMusic CBA | MDPI Information 15(4):224 (2024) — `github.com/mm9947/VioMusicv` |
| CREPE | Deep CNN pitch tracker — `pip install crepe` |
| madmom | RNN onset detection — `github.com/CPJKU/madmom` |
| Essentia | DSP + TF models (MusiCNN) — Essentia UPF |

---

## Output File Reference

| File | Contents |
|---|---|
| `emb_mert.npz` | MERT 768-dim embeddings (60 excerpts) |
| `emb_musicnn.npz` | MusiCNN embeddings (200-dim target / 20-dim current fallback) |
| `emb_cba.npz` | VioMusic CBA 256-dim embeddings |
| `emb_music2emo.npz` | Music2Emo embeddings (MERT proxy, 768-dim) |
| `feat_crepe.csv` | CREPE pitch/vibrato features (8-dim) |
| `feat_madmom.csv` | madmom timing/onset features (8-dim) |
| `feat_ess_dsp.csv` | Essentia DSP features (20-dim) |
| `feat_dtw.csv` | DTW warp features (6-dim) |
| `ablation_results.csv` | Full 24-experiment results |
| `final_summary.csv` | Best/avg per feature set |
| `all_predictions.csv` | Per-excerpt GT + predictions |
| `analysis_d_shifts.csv` | Piece-level embedding distances + VA deltas |
| `fig_ablation_bar.png` | Grouped bar chart |
| `fig_ablation_heatmap.png` | F1 / Hamming heatmaps |
| `fig_per_label_f1.png` | Per-label F1 heatmap |
| `fig_radar.png` | Radar — per-label F1 by feature set |
| `fig_va_scatter.png` | Predicted vs participant VA |
| `fig_circumplex.png` | Circumplex map all 60 excerpts |
| `fig_lowlevel_conditions.png` | Low-level features by condition |
| `fig_embedding_shift.png` | MERT distance vs arousal shift **(key thesis plot)** |

---

## Reporting Checklist

- [ ] Primary: A vs B vs C — Wilcoxon significant?
- [ ] **Hamming Loss** as primary metric; F1-macro as secondary
- [ ] Per-label F1 — which emotions are hardest to predict and why?
- [ ] Per-condition F1 (Analysis A) — Expressive vs Exaggerated gap?
- [ ] Quadrant accuracy (Analysis B) — VA regression quality
- [ ] Krippendorff's α per label (Analysis C)
- [ ] Analysis D correlations — **novel contribution**
- [ ] Flag in methods: CBA uses random-init weights
- [ ] Flag in methods: chord branch omitted for monophonic violin
- [ ] Flag in methods: MEC excerpts assigned Neutral by rule
- [ ] Flag in methods: Music2Emo represented by MERT proxy (justified)
- [ ] Flag in methods: MusiCNN 200-dim or document fallback if unresolved
