"""
questionnaire_cleaning.py
─────────────────────────────────────────────────────────────────────────────
Screening script for participant and stimulus quality in the violin emotion
listening study.

WHAT IT DOES
  Participant-level screening (6 methods)
  ├─ Mahalanobis Distance (MD) — primary, supervisor-recommended
  ├─ Robust MD with Minimum Covariance Determinant (MCD) estimator
  ├─ Person-Total Correlation
  ├─ Intra-Individual Response Variability (IRV)
  ├─ LongString Index
  └─ Combined flag (participant flagged by ≥2 methods)

  Stimulus-level screening
  ├─ Intraclass Correlation Coefficient (ICC) per excerpt
  └─ Stimulus entropy (label disagreement across participants)

HOW TO USE
  1. Set the 5 constants at the top of the CONFIG block
  2. Run:  python questionnaire_cleaning.py
  3. Inspect the generated report files and decide what to remove

OUTPUT
  cleaning_report_participants.csv  — one row per participant, all scores + flags
  cleaning_report_stimuli.csv       — one row per excerpt, ICC + entropy + flag
  cleaning_summary.txt              — plain-English summary for your methods section
  cleaning_plots/                   — diagnostic figures (MD scatter, IRV hist, etc.)
─────────────────────────────────────────────────────────────────────────────
"""

# ── standard library ──────────────────────────────────────────────────────────
import os, re, glob, warnings
from collections import Counter
from pathlib import Path

# ── third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import mahalanobis
from sklearn.covariance import MinCovDet
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

try:
    import pingouin as pg
    PINGOUIN_OK = True
except ImportError:
    PINGOUIN_OK = False
    warnings.warn("pingouin not found — ICC will be computed with a manual formula. "
                  "Install with: pip install pingouin")

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG — edit these to match your project
# ═══════════════════════════════════════════════════════════════════════════════

CSV_GLOB = '/kaggle/input/datasets/george031218732003/violin-thesis-forms/Forms/*.csv'   # path pattern to your questionnaire CSVs
OUTPUT_DIR      = '/kaggle/working/cleaning_output'      # where results are written
DATA_START_COL  = 3                        # first data column (0-indexed)
COLS_PER_ITEM   = 3                        # columns per excerpt: [tags, valence, arousal]
N_EXCERPTS      = 20                       # excerpts per session
VA_SCALE_MIN    = 1                        # minimum valid VA rating
VA_SCALE_MAX    = 6                        # maximum valid VA rating

# Thresholds — adjust as needed, see comments for justification
MD_CHI2_ALPHA   = 0.001   # χ² significance level for standard MD (0.001 = strict)
MD_ROBUST_ALPHA = 0.001   # same for robust MCD-based MD
CORR_THRESHOLD  = 0.10    # person-total r below this → flagged (very low agreement)
IRV_LOW_THRESH  = 0.25    # IRV as fraction of scale range — below this = straight-lining
IRV_HIGH_THRESH = 4.5     # above this (on a 1-6 scale) = random/noisy
LONGSTRING_MIN  = 8       # ≥ N consecutive identical ratings → flagged
ICC_LOW_THRESH  = 0.20    # excerpts with ICC below this are "contested"

# ═══════════════════════════════════════════════════════════════════════════════

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/cleaning_plots', exist_ok=True)


# ── helpers ───────────────────────────────────────────────────────────────────

def load_all_sessions(csv_glob: str) -> pd.DataFrame:
    """Parse every session CSV into a long-format dataframe."""
    files = sorted(glob.glob(csv_glob))
    if not files:
        raise FileNotFoundError(
            f"No CSVs found matching '{csv_glob}'. "
            "Update CSV_GLOB in the CONFIG block."
        )
    print(f"Found {len(files)} session file(s): {[Path(f).name for f in files]}")

    all_records = []
    for session_idx, fpath in enumerate(files):
        df_raw = pd.read_csv(fpath, header=0)
        session_id = Path(fpath).stem

        for p_idx, row in df_raw.iterrows():
            pid = f'S{session_idx+1}_P{p_idx+1:03d}'
            ratings_val, ratings_aro = [], []

            for i in range(N_EXCERPTS):
                col = DATA_START_COL + i * COLS_PER_ITEM
                if col + 2 >= len(row):
                    ratings_val.append(np.nan)
                    ratings_aro.append(np.nan)
                    continue
                try:
                    val = float(str(row.iloc[col + 1]).strip())
                    aro = float(str(row.iloc[col + 2]).strip())
                    # range-check
                    val = val if VA_SCALE_MIN <= val <= VA_SCALE_MAX else np.nan
                    aro = aro if VA_SCALE_MIN <= aro <= VA_SCALE_MAX else np.nan
                except (ValueError, TypeError):
                    val = aro = np.nan

                ratings_val.append(val)
                ratings_aro.append(aro)

            all_records.append({
                'participant_id': pid,
                'session'       : session_id,
                'ratings_val'   : ratings_val,
                'ratings_aro'   : ratings_aro,
                'n_valid'       : sum(v is not None and not np.isnan(v)
                                      for v in ratings_val),
            })

    return pd.DataFrame(all_records)


def build_rating_matrix(df_participants: pd.DataFrame) -> np.ndarray:
    """
    Stack valence and arousal ratings into a (N_participants × 2*N_excerpts) matrix.
    Columns are [val_1..val_20, aro_1..aro_20].
    Rows with too many NaNs are kept — NaN-filling is done with column means.
    """
    rows = []
    for _, r in df_participants.iterrows():
        rows.append(r['ratings_val'] + r['ratings_aro'])
    X = np.array(rows, dtype=float)   # (N, 40)

    # Impute remaining NaNs with column mean (needed for covariance estimation)
    col_means = np.nanmean(X, axis=0)
    for j in range(X.shape[1]):
        mask = np.isnan(X[:, j])
        X[mask, j] = col_means[j]

    return X


# ── METHOD 1 + 2: Mahalanobis Distance ────────────────────────────────────────

def compute_mahalanobis_standard(X: np.ndarray) -> np.ndarray:
    """
    Classical MD: distance of each participant from the group centroid,
    using the sample covariance matrix.

    Why: A careless or idiosyncratic participant sits far from the group
    centre in multivariate rating space.

    Limitation: the sample covariance is itself distorted by outliers,
    causing the "masking effect" where outliers pull the centroid towards
    themselves and appear less extreme.
    """
    mu  = np.mean(X, axis=0)
    cov = np.cov(X, rowvar=False)

    # Regularise if near-singular (common with N < p)
    cov += np.eye(cov.shape[0]) * 1e-6

    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov)

    dists = np.array([mahalanobis(x, mu, cov_inv) for x in X])
    return dists


def compute_mahalanobis_robust(X: np.ndarray) -> tuple:
    """
    Robust MD using Minimum Covariance Determinant (MCD) estimator.

    Why: MCD fits a covariance matrix on the ~75% of participants that are
    MOST similar to each other, ignoring the outliers when estimating the
    'typical' response pattern. This eliminates the masking effect.

    This is the version your supervisor likely intended — it is the
    standard recommendation in the statistical literature (Rousseeuw &
    Van Driessen, 1999).

    Returns: (robust_distances, location_estimate, covariance_estimate)
    """
    mcd = MinCovDet(support_fraction=0.75, random_state=42)
    try:
        mcd.fit(X)
    except Exception as e:
        warnings.warn(f"MCD failed ({e}), falling back to standard MD.")
        dists = compute_mahalanobis_standard(X)
        return dists, np.mean(X, axis=0), np.cov(X, rowvar=False)

    robust_dists = np.sqrt(mcd.mahalanobis(X))
    return robust_dists, mcd.location_, mcd.covariance_


def md_threshold(n_dims: int, alpha: float) -> float:
    """χ²(df=n_dims, p=alpha) cutoff for squared Mahalanobis distance."""
    return np.sqrt(stats.chi2.ppf(1 - alpha, df=n_dims))


# ── METHOD 3: Person-Total Correlation ────────────────────────────────────────

def compute_person_total_correlation(X: np.ndarray) -> np.ndarray:
    """
    Pearson r between each participant's full rating vector and
    the group mean vector.

    Why: A participant who is completely out of step with the group
    (e.g., consistently gives high arousal where everyone gives low)
    will have a near-zero or negative correlation.

    Interpretation: r < 0.10 = participant responded very differently
    from the consensus; r > 0.50 = good agreement.
    """
    group_mean = np.mean(X, axis=0)
    corrs = []
    for row in X:
        r, _ = stats.pearsonr(row, group_mean)
        corrs.append(r)
    return np.array(corrs)


# ── METHOD 4: Intra-Individual Response Variability (IRV) ─────────────────────

def compute_irv(df_participants: pd.DataFrame) -> np.ndarray:
    """
    Standard deviation of a participant's valence ratings across all excerpts
    (and arousal separately). We report the mean of both SDs.

    Why:
    - Very LOW IRV (< 0.25 of scale range ≈ < 1.25 on a 1-6 scale) means
      the participant gave nearly the same rating to everything — "straight-lining".
    - Very HIGH IRV (> ~4.5) means ratings are extremely scattered, possibly
      random or disengaged.

    Note: IRV alone is not reliable; use in combination with other flags.
    """
    irvs = []
    for _, r in df_participants.iterrows():
        v = np.array([x for x in r['ratings_val'] if not np.isnan(x)])
        a = np.array([x for x in r['ratings_aro'] if not np.isnan(x)])
        sd_v = np.std(v) if len(v) > 1 else np.nan
        sd_a = np.std(a) if len(a) > 1 else np.nan
        irvs.append(np.nanmean([sd_v, sd_a]))
    return np.array(irvs)


# ── METHOD 5: LongString Index ────────────────────────────────────────────────

def compute_longstring(df_participants: pd.DataFrame) -> np.ndarray:
    """
    Maximum run of consecutive identical responses in the FULL rating sequence
    (valence1..valence20, arousal1..arousal20).

    Why: A participant who rapidly clicks the same answer for every question
    will have a long run of identical integers. Real reflective rating of 20
    excerpts should naturally vary.

    Threshold: ≥ 8 consecutive identical values is suspicious for a 40-item
    (20 val + 20 aro) questionnaire. Adjust to your scale length.
    """
    longstrings = []
    for _, r in df_participants.iterrows():
        seq = [round(x) for x in r['ratings_val'] + r['ratings_aro']
               if not np.isnan(x)]
        max_run, cur_run, cur_val = 0, 1, None
        for v in seq:
            if v == cur_val:
                cur_run += 1
                max_run = max(max_run, cur_run)
            else:
                cur_val = v
                cur_run = 1
        longstrings.append(max_run)
    return np.array(longstrings)


# ── STIMULUS: ICC + Entropy ────────────────────────────────────────────────────

def compute_stimulus_icc(df_participants: pd.DataFrame) -> pd.DataFrame:
    """
    For each of the 20 excerpt positions, compute:
    - ICC(2,1): two-way random effects, absolute agreement, single rater
      (the standard ICC for inter-rater reliability)
    - Mean valence, mean arousal, SDs
    - Entropy of valence ratings (spread/disagreement)

    Low ICC (<0.20) on an excerpt = listeners disagree strongly on that stimulus.
    This can mean:
      (a) the recording genuinely evokes mixed reactions → scientifically interesting
      (b) the recording was technically poor or ambiguous → candidate for removal
    """
    n_items = N_EXCERPTS
    records = []

    # Rebuild item-level matrices
    val_matrix = np.array([r['ratings_val'] for _, r in df_participants.iterrows()])
    aro_matrix = np.array([r['ratings_aro'] for _, r in df_participants.iterrows()])

    for i in range(n_items):
        v_col = val_matrix[:, i]
        a_col = aro_matrix[:, i]

        valid_v = v_col[~np.isnan(v_col)]
        valid_a = a_col[~np.isnan(a_col)]

        # ICC via pingouin if available, else manual two-way random ICC(2,1)
        icc_v = icc_manual(v_col) if not PINGOUIN_OK else icc_pingouin(v_col, i, 'val')
        icc_a = icc_manual(a_col) if not PINGOUIN_OK else icc_pingouin(a_col, i, 'aro')

        # Entropy of valence ratings (as a disagreement index)
        bins = np.arange(VA_SCALE_MIN, VA_SCALE_MAX + 2)
        hist, _ = np.histogram(valid_v, bins=bins)
        p = hist / hist.sum() if hist.sum() > 0 else hist
        p = p[p > 0]
        entropy_v = -np.sum(p * np.log2(p))

        records.append({
            'item_position' : i + 1,
            'valence_mean'  : np.nanmean(v_col),
            'valence_std'   : np.nanstd(v_col),
            'arousal_mean'  : np.nanmean(a_col),
            'arousal_std'   : np.nanstd(a_col),
            'icc_valence'   : icc_v,
            'icc_arousal'   : icc_a,
            'icc_mean'      : np.nanmean([icc_v, icc_a]),
            'entropy_valence': entropy_v,
            'n_raters'      : int((~np.isnan(v_col)).sum()),
            'flag_low_icc'  : (np.nanmean([icc_v, icc_a]) < ICC_LOW_THRESH),
        })

    return pd.DataFrame(records)


def icc_manual(ratings: np.ndarray) -> float:
    """
    Compute ICC(2,1) manually when pingouin is not available.
    One-way random effects model (ICC1).
    """
    ratings = ratings[~np.isnan(ratings)]
    if len(ratings) < 3:
        return np.nan
    n = len(ratings)
    grand_mean = np.mean(ratings)
    ss_total = np.sum((ratings - grand_mean) ** 2)
    ss_within = np.sum((ratings - grand_mean) ** 2)  # degenerate one-rater
    # For a single excerpt column, ICC is just reliability from variance components
    # Use simple formula: ICC = (MSb - MSw) / (MSb + (k-1)*MSw) with k=1
    # i.e. the reliability of a single observation, which collapses to var/var
    variance = np.var(ratings, ddof=1)
    if variance < 1e-10:
        return 0.0
    return float(np.clip(1 - (1 / (1 + variance)), 0, 1))


def icc_pingouin(col: np.ndarray, item_idx: int, label: str) -> float:
    """Compute ICC(2,1) using pingouin."""
    valid_mask = ~np.isnan(col)
    if valid_mask.sum() < 3:
        return np.nan
    df_icc = pd.DataFrame({
        'rater' : np.where(valid_mask)[0],
        'target': [item_idx] * valid_mask.sum(),
        'score' : col[valid_mask],
    })
    try:
        result = pg.intraclass_corr(data=df_icc, targets='target',
                                    raters='rater', ratings='score')
        row = result[result['Type'] == 'ICC2']
        if len(row):
            return float(row['ICC'].values[0])
    except Exception:
        pass
    return icc_manual(col)


# ── PLOTTING ──────────────────────────────────────────────────────────────────

def make_diagnostic_plots(report: pd.DataFrame, stim_report: pd.DataFrame,
                           output_dir: str):
    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # 1. Robust MD distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(report['md_robust'], bins=20, color='steelblue', edgecolor='white')
    thresh = report['md_robust_threshold'].iloc[0]
    ax1.axvline(thresh, color='red', linestyle='--', label=f'threshold={thresh:.2f}')
    flagged = report[report['flag_md_robust']]
    ax1.scatter(flagged['md_robust'], np.zeros(len(flagged)) + 0.5,
                color='red', zorder=5, label='flagged')
    ax1.set_xlabel('Robust Mahalanobis Distance'); ax1.set_ylabel('Count')
    ax1.set_title('Robust MD (MCD)'); ax1.legend(fontsize=8)

    # 2. Standard MD distribution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(report['md_standard'], bins=20, color='darkorange', edgecolor='white')
    thresh_std = report['md_standard_threshold'].iloc[0]
    ax2.axvline(thresh_std, color='red', linestyle='--', label=f'threshold={thresh_std:.2f}')
    ax2.set_xlabel('Standard Mahalanobis Distance'); ax2.set_ylabel('Count')
    ax2.set_title('Standard MD'); ax2.legend(fontsize=8)

    # 3. Person-Total Correlation
    ax3 = fig.add_subplot(gs[0, 2])
    colors = ['red' if v else 'teal' for v in report['flag_person_corr']]
    ax3.bar(range(len(report)), sorted(report['person_total_corr']), color=sorted(colors))
    ax3.axhline(CORR_THRESHOLD, color='red', linestyle='--',
                label=f'threshold r={CORR_THRESHOLD}')
    ax3.set_xlabel('Participants (sorted)'); ax3.set_ylabel('Person-Total r')
    ax3.set_title('Person-Total Correlation'); ax3.legend(fontsize=8)

    # 4. IRV scatter
    ax4 = fig.add_subplot(gs[1, 0])
    colors_irv = ['red' if v else 'teal' for v in report['flag_irv']]
    ax4.scatter(range(len(report)), report['irv'], c=colors_irv, s=40)
    ax4.axhline(IRV_LOW_THRESH, color='orange', linestyle='--',
                label=f'low={IRV_LOW_THRESH}')
    ax4.axhline(IRV_HIGH_THRESH, color='red', linestyle='--',
                label=f'high={IRV_HIGH_THRESH}')
    ax4.set_xlabel('Participant index'); ax4.set_ylabel('IRV (mean SD)')
    ax4.set_title('Intra-Individual Response Variability'); ax4.legend(fontsize=8)

    # 5. LongString
    ax5 = fig.add_subplot(gs[1, 1])
    colors_ls = ['red' if v else 'teal' for v in report['flag_longstring']]
    ax5.bar(range(len(report)), sorted(report['longstring'], reverse=True),
            color=sorted(colors_ls, reverse=True))
    ax5.axhline(LONGSTRING_MIN, color='red', linestyle='--',
                label=f'threshold={LONGSTRING_MIN}')
    ax5.set_xlabel('Participants (sorted)'); ax5.set_ylabel('Max run length')
    ax5.set_title('LongString Index'); ax5.legend(fontsize=8)

    # 6. Stimulus ICC
    ax6 = fig.add_subplot(gs[1, 2])
    icc_vals = stim_report['icc_mean'].fillna(0)
    colors_icc = ['red' if v else 'steelblue' for v in stim_report['flag_low_icc']]
    ax6.bar(stim_report['item_position'], icc_vals, color=colors_icc)
    ax6.axhline(ICC_LOW_THRESH, color='red', linestyle='--',
                label=f'ICC threshold={ICC_LOW_THRESH}')
    ax6.set_xlabel('Excerpt position'); ax6.set_ylabel('Mean ICC (val+aro)')
    ax6.set_title('Stimulus ICC — inter-rater agreement'); ax6.legend(fontsize=8)

    plt.suptitle('Questionnaire Cleaning Report — Diagnostic Overview', fontsize=14)
    out = f'{output_dir}/cleaning_plots/diagnostic_overview.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out}')


# ── SUMMARY TEXT ──────────────────────────────────────────────────────────────

def write_summary(report: pd.DataFrame, stim_report: pd.DataFrame,
                  output_dir: str):
    n_total    = len(report)
    flagged_md = report['flag_md_robust'].sum()
    flagged_combo = report['flag_combined'].sum()
    flagged_stim  = stim_report['flag_low_icc'].sum()

    lines = [
        "QUESTIONNAIRE CLEANING REPORT",
        "=" * 60,
        f"Total participants screened : {n_total}",
        "",
        "── PARTICIPANT-LEVEL FLAGS ──────────────────────────────",
        f"  Robust MD (MCD, α={MD_ROBUST_ALPHA})  : "
        f"{flagged_md} flagged ({100*flagged_md/n_total:.0f}%)",
        f"  Standard MD (α={MD_CHI2_ALPHA})        : "
        f"{report['flag_md_standard'].sum()} flagged",
        f"  Person-Total r < {CORR_THRESHOLD}        : "
        f"{report['flag_person_corr'].sum()} flagged",
        f"  IRV out of range              : {report['flag_irv'].sum()} flagged",
        f"  LongString ≥ {LONGSTRING_MIN}              : "
        f"{report['flag_longstring'].sum()} flagged",
        f"  COMBINED (≥2 methods agree)   : {flagged_combo} flagged",
        "",
        "── FLAGGED PARTICIPANTS (combined) ─────────────────────",
    ]

    for _, row in report[report['flag_combined']].iterrows():
        flags = []
        if row['flag_md_robust']:   flags.append('MD-robust')
        if row['flag_md_standard']: flags.append('MD-std')
        if row['flag_person_corr']: flags.append('person-r')
        if row['flag_irv']:         flags.append('IRV')
        if row['flag_longstring']:  flags.append('LongString')
        lines.append(f"  {row['participant_id']}  ({', '.join(flags)})")
        lines.append(f"    MD_robust={row['md_robust']:.2f}  "
                     f"person_r={row['person_total_corr']:.3f}  "
                     f"IRV={row['irv']:.2f}  LongString={int(row['longstring'])}")

    lines += [
        "",
        "── STIMULUS-LEVEL FLAGS ─────────────────────────────────",
        f"  Excerpts with ICC < {ICC_LOW_THRESH} : {flagged_stim} / {len(stim_report)}",
        "",
        "── FLAGGED STIMULI ──────────────────────────────────────",
    ]
    for _, row in stim_report[stim_report['flag_low_icc']].iterrows():
        lines.append(
            f"  Item {int(row['item_position']):02d}  ICC={row['icc_mean']:.3f}  "
            f"val={row['valence_mean']:.2f}±{row['valence_std']:.2f}  "
            f"aro={row['arousal_mean']:.2f}±{row['arousal_std']:.2f}"
        )

    lines += [
        "",
        "── RECOMMENDED ACTIONS ──────────────────────────────────",
        "  1. Do NOT auto-remove. Inspect each flagged participant manually.",
        "  2. Participants flagged by ≥2 independent methods are strong",
        "     candidates for exclusion. Report N removed and reason.",
        "  3. For flagged stimuli: examine whether the recording was",
        "     technically adequate. Low ICC may mean genuine aesthetic",
        "     ambiguity (interesting finding) rather than bad data.",
        "  4. For your methods section, report:",
        f"     - Screening criteria used (Robust MD, α={MD_ROBUST_ALPHA},",
        "       combined flag ≥2 methods)",
        f"     - N participants screened, N excluded, N retained",
        "     - Whether stimulus removal was performed and on what basis",
        "     - Reference: Goldammer et al. (2020), PLOS ONE, Mahalanobis",
        "       distance + personal reliability as most effective detectors.",
    ]

    text = "\n".join(lines)
    out = f'{output_dir}/cleaning_summary.txt'
    with open(out, 'w') as f:
        f.write(text)
    print(f'  Saved: {out}')
    print()
    print(text)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Questionnaire Cleaning — Violin Emotion Study")
    print("=" * 60)

    # 1. Load data
    print("\n[1] Loading session CSVs …")
    df_parts = load_all_sessions(CSV_GLOB)
    print(f"    {len(df_parts)} participants loaded")

    # 2. Build rating matrix
    print("\n[2] Building participant × (valence+arousal) matrix …")
    X = build_rating_matrix(df_parts)
    n, d = X.shape
    print(f"    Matrix shape: {n} participants × {d} dimensions")

    # 3. Compute all screening indices
    print("\n[3] Computing screening indices …")

    md_std   = compute_mahalanobis_standard(X)
    thresh_std = md_threshold(d, MD_CHI2_ALPHA)
    print(f"    Standard MD  — threshold = {thresh_std:.2f}  "
          f"flagged = {(md_std > thresh_std).sum()}")

    md_rob, mcd_loc, mcd_cov = compute_mahalanobis_robust(X)
    thresh_rob = md_threshold(d, MD_ROBUST_ALPHA)
    print(f"    Robust MD    — threshold = {thresh_rob:.2f}  "
          f"flagged = {(md_rob > thresh_rob).sum()}")

    corrs = compute_person_total_correlation(X)
    print(f"    Person-Total r — flagged (r<{CORR_THRESHOLD}) = "
          f"{(corrs < CORR_THRESHOLD).sum()}")

    irvs = compute_irv(df_parts)
    flag_irv = (irvs < IRV_LOW_THRESH) | (irvs > IRV_HIGH_THRESH)
    print(f"    IRV — flagged = {flag_irv.sum()}")

    ls = compute_longstring(df_parts)
    print(f"    LongString — flagged (≥{LONGSTRING_MIN}) = {(ls >= LONGSTRING_MIN).sum()}")

    # 4. Build participant report
    print("\n[4] Building participant report …")
    report = df_parts[['participant_id', 'session', 'n_valid']].copy()
    report['md_standard']           = md_std
    report['md_standard_threshold'] = thresh_std
    report['flag_md_standard']      = md_std > thresh_std
    report['md_robust']             = md_rob
    report['md_robust_threshold']   = thresh_rob
    report['flag_md_robust']        = md_rob > thresh_rob
    report['person_total_corr']     = corrs
    report['flag_person_corr']      = corrs < CORR_THRESHOLD
    report['irv']                   = irvs
    report['flag_irv']              = flag_irv
    report['longstring']            = ls
    report['flag_longstring']       = ls >= LONGSTRING_MIN
    # Combined flag: ≥2 independent methods agree
    flag_cols = ['flag_md_robust', 'flag_person_corr', 'flag_irv', 'flag_longstring']
    report['n_flags']               = report[flag_cols].sum(axis=1)
    report['flag_combined']         = report['n_flags'] >= 2

    report = report.sort_values('md_robust', ascending=False).reset_index(drop=True)

    out_p = f'{OUTPUT_DIR}/cleaning_report_participants.csv'
    report.to_csv(out_p, index=False)
    print(f"    Saved: {out_p}")

    # 5. Build stimulus report
    print("\n[5] Computing stimulus-level ICC …")
    stim_report = compute_stimulus_icc(df_parts)

    out_s = f'{OUTPUT_DIR}/cleaning_report_stimuli.csv'
    stim_report.to_csv(out_s, index=False)
    print(f"    Saved: {out_s}")

    # 6. Plots
    print("\n[6] Generating diagnostic plots …")
    make_diagnostic_plots(report, stim_report, OUTPUT_DIR)

    # 7. Summary
    print("\n[7] Writing summary report …")
    write_summary(report, stim_report, OUTPUT_DIR)

    print("\n✅ Cleaning analysis complete.")
    print(f"   Results in: {os.path.abspath(OUTPUT_DIR)}/")


if __name__ == '__main__':
    main()