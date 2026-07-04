"""
questionnaire_cleaning.py  (v2 — participant screening only)
─────────────────────────────────────────────────────────────────────────────
Participant-level quality screening for the violin emotion listening study.

WHAT CHANGED FROM v1
  • Stimulus/ICC screening REMOVED from this file — it now lives in the separate
    excerpt_cleaning.py (the old per-excerpt "ICC" here was mathematically invalid).
  • Participant IDs now match the NOTEBOOK convention ('{csv_stem}_P{nnn}', e.g.
    'questionnaire_5_P012') so a flagged list can be pasted straight into the
    pipeline's exclusion step. A short alias ('S5_P012') is also emitted for
    continuity with earlier reports.
  • Person-total correlation is now LEAVE-ONE-OUT (each participant is compared
    against the group mean computed WITHOUT them) — removes self-inclusion bias.
  • Adds an incomplete-response flag (too few valid ratings).
  • Fixes a plotting bug in v1 (bar colours were sorted independently of bar
    heights, mis-colouring the flagged bars).
  • Writes 'excluded_participant_ids.csv' — the ≥2-method exclusion list, ready
    to use in the notebook.

SCREENING METHODS (participant level)
  1. Mahalanobis Distance (classical)      — distance from group centroid
  2. Robust MD via MCD (Rousseeuw)         — masking-resistant; PRIMARY
  3. Person-Total correlation (LOO)        — agreement with the consensus
  4. Intra-individual Response Variability — straight-lining / random noise
  5. LongString index                      — runs of identical clicks
  → COMBINED flag = flagged by ≥2 independent methods (exclusion candidates)

DESIGN NOTE / CAVEAT
  MD assumes multivariate-normal ratings; 1–6 Likert data violate this, so the
  χ² cutoff is a convention, not ground truth. With p=40 rating dimensions the
  covariance is estimated from n≈118 participants — adequate but noisy. Treat MD
  as ONE detector among several; only the ≥2-method combined flag drives removal.

USAGE
  1. Edit the CONFIG block (CSV_GLOB, OUTPUT_DIR).
  2. python questionnaire_cleaning.py
  3. Inspect the reports; apply excluded_participant_ids.csv in the notebook.

Reference: Goldammer et al. (2020), PLOS ONE — Mahalanobis distance + personal
reliability are among the most effective careless-responder detectors.
─────────────────────────────────────────────────────────────────────────────
"""

import os, glob, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import mahalanobis
from sklearn.covariance import MinCovDet
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
CSV_GLOB = '/kaggle/input/datasets/george031218732003/violin-thesis-forms/Forms/questionnaire_*.csv'
OUTPUT_DIR     = '/kaggle/working/cleaning_output'
DATA_START_COL = 3     # first data column (0-indexed): layout is [tags, val, aro] × 20
COLS_PER_ITEM  = 3
N_EXCERPTS     = 20
VA_SCALE_MIN   = 1
VA_SCALE_MAX   = 6

# Thresholds
MD_CHI2_ALPHA   = 0.001   # χ² level for classical MD
MD_ROBUST_ALPHA = 0.001   # χ² level for robust (MCD) MD
CORR_THRESHOLD  = 0.10    # person-total r below this → flagged
IRV_LOW_THRESH  = 0.25    # SD below this (straight-lining) — see note in compute_irv
IRV_HIGH_THRESH = 2.20    # SD above this on a 1–6 scale (near the theoretical max) → random
LONGSTRING_MIN  = 8       # ≥ N consecutive identical ratings → flagged
MIN_VALID_FRAC  = 0.50    # participants with < this fraction of valid ratings → flagged
MCD_SUPPORT_FRAC = 0.75   # fraction of "clean" participants MCD fits its covariance on
RANDOM_STATE     = 42

# ═══════════════════════════════════════════════════════════════════════════════

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/cleaning_plots', exist_ok=True)


# ── LOADING ───────────────────────────────────────────────────────────────────
def load_all_sessions(csv_glob):
    """Parse every session CSV into one participant-level dataframe.

    participant_id uses the notebook convention: '{csv_stem}_P{idx:03d}'.
    alias_id is the short 'S{session}_P{idx:03d}' form used in earlier reports.
    """
    files = sorted(glob.glob(csv_glob))
    if not files:
        raise FileNotFoundError(
            f"No CSVs match '{csv_glob}'. Update CSV_GLOB in CONFIG.")
    print(f"Found {len(files)} session file(s): {[Path(f).name for f in files]}")

    records = []
    for s_idx, fpath in enumerate(files):
        df_raw = pd.read_csv(fpath, header=0)
        stem = Path(fpath).stem
        for p_idx, row in df_raw.iterrows():
            pid   = f'{stem}_P{p_idx + 1:03d}'
            alias = f'S{s_idx + 1}_P{p_idx + 1:03d}'
            rv, ra = [], []
            for i in range(N_EXCERPTS):
                col = DATA_START_COL + i * COLS_PER_ITEM
                if col + 2 >= len(row):
                    rv.append(np.nan); ra.append(np.nan); continue
                try:
                    v = float(str(row.iloc[col + 1]).strip())
                    a = float(str(row.iloc[col + 2]).strip())
                    v = v if VA_SCALE_MIN <= v <= VA_SCALE_MAX else np.nan
                    a = a if VA_SCALE_MIN <= a <= VA_SCALE_MAX else np.nan
                except (ValueError, TypeError):
                    v = a = np.nan
                rv.append(v); ra.append(a)
            n_valid = int(np.sum(~np.isnan(rv)) + np.sum(~np.isnan(ra)))
            records.append({'participant_id': pid, 'alias_id': alias,
                            'session': stem, 'ratings_val': rv, 'ratings_aro': ra,
                            'n_valid': n_valid})
    return pd.DataFrame(records)


def build_rating_matrix(df):
    """(N × 40) matrix [val_1..val_20, aro_1..aro_20]; NaNs → column mean."""
    X = np.array([r['ratings_val'] + r['ratings_aro'] for _, r in df.iterrows()],
                 dtype=float)
    col_means = np.nanmean(X, axis=0)
    idx = np.where(np.isnan(X))
    X[idx] = np.take(col_means, idx[1])
    return X


# ── METHOD 1 & 2: Mahalanobis ─────────────────────────────────────────────────
def compute_md_standard(X):
    mu, cov = np.mean(X, axis=0), np.cov(X, rowvar=False)
    cov += np.eye(cov.shape[0]) * 1e-6            # regularise near-singular
    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov)
    return np.array([mahalanobis(x, mu, cov_inv) for x in X])


def compute_md_robust(X):
    """Robust MD via Minimum Covariance Determinant (masking-resistant)."""
    try:
        mcd = MinCovDet(support_fraction=MCD_SUPPORT_FRAC,
                        random_state=RANDOM_STATE).fit(X)
        return np.sqrt(mcd.mahalanobis(X))
    except Exception as e:
        warnings.warn(f"MCD failed ({e}); falling back to classical MD.")
        return compute_md_standard(X)


def md_threshold(n_dims, alpha):
    return np.sqrt(stats.chi2.ppf(1 - alpha, df=n_dims))


# ── METHOD 3: Person-Total correlation (leave-one-out) ────────────────────────
def compute_person_total_corr_loo(X):
    """Pearson r between each participant and the group mean computed WITHOUT them."""
    n = X.shape[0]
    total = X.sum(axis=0)
    out = []
    for i in range(n):
        others_mean = (total - X[i]) / (n - 1)
        if np.std(X[i]) < 1e-12 or np.std(others_mean) < 1e-12:
            out.append(0.0)                       # constant vector → no agreement info
        else:
            out.append(stats.pearsonr(X[i], others_mean)[0])
    return np.array(out)


# ── METHOD 4: IRV ─────────────────────────────────────────────────────────────
def compute_irv(df):
    """Mean of (SD of valence, SD of arousal) per participant.

    Low  → straight-lining (same answer to everything).
    High → scattered/random. Max SD on a 1–6 scale is ~2.5 (all mass at 1 & 6),
    so IRV_HIGH_THRESH is set near that ceiling rather than the >4.5 of v1
    (which was unreachable and never fired).
    """
    out = []
    for _, r in df.iterrows():
        v = np.array([x for x in r['ratings_val'] if not np.isnan(x)])
        a = np.array([x for x in r['ratings_aro'] if not np.isnan(x)])
        sd_v = np.std(v) if len(v) > 1 else np.nan
        sd_a = np.std(a) if len(a) > 1 else np.nan
        out.append(np.nanmean([sd_v, sd_a]))
    return np.array(out)


# ── METHOD 5: LongString ──────────────────────────────────────────────────────
def compute_longstring(df):
    """Longest run of identical consecutive ratings across the 40-item sequence."""
    out = []
    for _, r in df.iterrows():
        seq = [round(x) for x in (r['ratings_val'] + r['ratings_aro'])
               if not np.isnan(x)]
        max_run = cur = 1 if seq else 0
        for k in range(1, len(seq)):
            cur = cur + 1 if seq[k] == seq[k - 1] else 1
            max_run = max(max_run, cur)
        out.append(max_run)
    return np.array(out)


# ── PLOTS ─────────────────────────────────────────────────────────────────────
def make_plots(report, output_dir):
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    ax.hist(report['md_robust'], bins=25, color='steelblue', edgecolor='white')
    thr = report['md_robust_threshold'].iloc[0]
    ax.axvline(thr, color='red', ls='--', label=f'threshold={thr:.1f}')
    ax.set_title('Robust MD (MCD)'); ax.set_xlabel('robust MD'); ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[0, 1])
    ax.hist(report['md_standard'], bins=25, color='darkorange', edgecolor='white')
    thr = report['md_standard_threshold'].iloc[0]
    ax.axvline(thr, color='red', ls='--', label=f'threshold={thr:.1f}')
    ax.set_title('Classical MD'); ax.set_xlabel('MD'); ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[0, 2])
    order = report['person_total_corr'].sort_values().index
    vals  = report.loc[order, 'person_total_corr'].values
    flg   = report.loc[order, 'flag_person_corr'].values
    ax.bar(range(len(vals)), vals,
           color=['red' if f else 'teal' for f in flg])   # colour tracks value
    ax.axhline(CORR_THRESHOLD, color='red', ls='--', label=f'r={CORR_THRESHOLD}')
    ax.set_title('Person-Total r (LOO)'); ax.set_xlabel('participants (sorted)')
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 0])
    ax.scatter(range(len(report)), report['irv'],
               c=['red' if f else 'teal' for f in report['flag_irv']], s=25)
    ax.axhline(IRV_LOW_THRESH, color='orange', ls='--', label=f'low={IRV_LOW_THRESH}')
    ax.axhline(IRV_HIGH_THRESH, color='red', ls='--', label=f'high={IRV_HIGH_THRESH}')
    ax.set_title('IRV (mean SD)'); ax.set_xlabel('participant'); ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 1])
    order = report['longstring'].sort_values(ascending=False).index
    vals  = report.loc[order, 'longstring'].values
    flg   = report.loc[order, 'flag_longstring'].values
    ax.bar(range(len(vals)), vals,
           color=['red' if f else 'teal' for f in flg])
    ax.axhline(LONGSTRING_MIN, color='red', ls='--', label=f'≥{LONGSTRING_MIN}')
    ax.set_title('LongString'); ax.set_xlabel('participants (sorted)'); ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 2])
    counts = report['n_flags'].value_counts().sort_index()
    ax.bar(counts.index.astype(int), counts.values, color='slateblue')
    ax.axvline(1.5, color='red', ls='--', label='exclusion cut (≥2)')
    ax.set_title('# methods flagging each participant')
    ax.set_xlabel('n flags'); ax.set_ylabel('participants'); ax.legend(fontsize=8)

    plt.suptitle('Participant Screening — Diagnostic Overview', fontsize=14)
    out = f'{output_dir}/cleaning_plots/participant_overview.png'
    fig.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f'  Saved: {out}')


# ── SUMMARY ───────────────────────────────────────────────────────────────────
def write_summary(report, output_dir):
    n = len(report)
    lines = [
        "PARTICIPANT SCREENING REPORT", "=" * 60,
        f"Total participants screened : {n}", "",
        "── FLAGS PER METHOD ─────────────────────────────────────",
        f"  Robust MD (MCD, α={MD_ROBUST_ALPHA})   : {int(report['flag_md_robust'].sum())}",
        f"  Classical MD (α={MD_CHI2_ALPHA})       : {int(report['flag_md_standard'].sum())}",
        f"  Person-Total r < {CORR_THRESHOLD} (LOO)  : {int(report['flag_person_corr'].sum())}",
        f"  IRV out of [{IRV_LOW_THRESH}, {IRV_HIGH_THRESH}] : {int(report['flag_irv'].sum())}",
        f"  LongString ≥ {LONGSTRING_MIN}               : {int(report['flag_longstring'].sum())}",
        f"  Incomplete (< {int(MIN_VALID_FRAC*100)}% valid)     : {int(report['flag_incomplete'].sum())}",
        f"  COMBINED (≥2 methods)          : {int(report['flag_combined'].sum())}", "",
        "── EXCLUSION CANDIDATES (≥2 methods) ────────────────────",
    ]
    for _, r in report[report['flag_combined']].iterrows():
        methods = [m for m, c in [('MD-robust', 'flag_md_robust'),
                                  ('MD-std', 'flag_md_standard'),
                                  ('person-r', 'flag_person_corr'),
                                  ('IRV', 'flag_irv'),
                                  ('LongString', 'flag_longstring')] if r[c]]
        lines.append(f"  {r['participant_id']}  ({r['alias_id']})  [{', '.join(methods)}]")
        lines.append(f"    MD_rob={r['md_robust']:.2f}  person_r={r['person_total_corr']:.3f}  "
                     f"IRV={r['irv']:.2f}  LongString={int(r['longstring'])}  n_valid={int(r['n_valid'])}")
    lines += [
        "", "── METHODS-SECTION TEXT ─────────────────────────────────",
        "  Screening: classical + robust (MCD) Mahalanobis distance,",
        "  leave-one-out person-total correlation, intra-individual response",
        "  variability, and LongString index. Participants flagged by ≥2 of",
        "  these independent detectors were excluded. Report N screened /",
        "  N excluded / N retained and cite Goldammer et al. (2020).",
        "  NOTE: MD χ² cutoff assumes multivariate normality (a convention for",
        "  Likert data); the ≥2-method rule is what governs exclusion.",
    ]
    text = "\n".join(lines)
    with open(f'{output_dir}/cleaning_summary.txt', 'w') as f:
        f.write(text)
    print(f'  Saved: {output_dir}/cleaning_summary.txt\n'); print(text)


# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  Participant Screening — Violin Emotion Study")
    print("=" * 60)

    print("\n[1] Loading session CSVs …")
    df = load_all_sessions(CSV_GLOB)
    print(f"    {len(df)} participants loaded")

    print("\n[2] Building participant × (valence+arousal) matrix …")
    X = build_rating_matrix(df)
    n, d = X.shape
    print(f"    Matrix shape: {n} × {d}")

    print("\n[3] Computing screening indices …")
    md_std = compute_md_standard(X); thr_std = md_threshold(d, MD_CHI2_ALPHA)
    print(f"    Classical MD — threshold={thr_std:.2f}  flagged={(md_std > thr_std).sum()}")
    md_rob = compute_md_robust(X);   thr_rob = md_threshold(d, MD_ROBUST_ALPHA)
    print(f"    Robust MD    — threshold={thr_rob:.2f}  flagged={(md_rob > thr_rob).sum()}")
    corrs = compute_person_total_corr_loo(X)
    print(f"    Person-Total r (LOO) — flagged={(corrs < CORR_THRESHOLD).sum()}")
    irvs = compute_irv(df)
    flag_irv = (irvs < IRV_LOW_THRESH) | (irvs > IRV_HIGH_THRESH)
    print(f"    IRV — flagged={flag_irv.sum()}")
    ls = compute_longstring(df)
    print(f"    LongString — flagged={(ls >= LONGSTRING_MIN).sum()}")

    print("\n[4] Building report …")
    rep = df[['participant_id', 'alias_id', 'session', 'n_valid']].copy()
    rep['md_standard'] = md_std; rep['md_standard_threshold'] = thr_std
    rep['flag_md_standard'] = md_std > thr_std
    rep['md_robust'] = md_rob;   rep['md_robust_threshold'] = thr_rob
    rep['flag_md_robust'] = md_rob > thr_rob
    rep['person_total_corr'] = corrs; rep['flag_person_corr'] = corrs < CORR_THRESHOLD
    rep['irv'] = irvs; rep['flag_irv'] = flag_irv
    rep['longstring'] = ls; rep['flag_longstring'] = ls >= LONGSTRING_MIN
    max_valid = 2 * N_EXCERPTS
    rep['flag_incomplete'] = rep['n_valid'] < (MIN_VALID_FRAC * max_valid)

    # Combined flag: ≥2 of the independent behavioural/statistical detectors.
    # (Classical MD is excluded — redundant with robust MD. Incomplete is a
    #  separate data-quality flag, not a carelessness detector.)
    flag_cols = ['flag_md_robust', 'flag_person_corr', 'flag_irv', 'flag_longstring']
    rep['n_flags'] = rep[flag_cols].sum(axis=1)
    rep['flag_combined'] = rep['n_flags'] >= 2
    rep = rep.sort_values('md_robust', ascending=False).reset_index(drop=True)

    rep.to_csv(f'{OUTPUT_DIR}/cleaning_report_participants.csv', index=False)
    print(f"    Saved: {OUTPUT_DIR}/cleaning_report_participants.csv")

    excl = rep.loc[rep['flag_combined'], ['participant_id', 'alias_id', 'n_flags']]
    excl.to_csv(f'{OUTPUT_DIR}/excluded_participant_ids.csv', index=False)
    print(f"    Saved: {OUTPUT_DIR}/excluded_participant_ids.csv  "
          f"({len(excl)} exclusion candidates)")

    print("\n[5] Plots …");   make_plots(rep, OUTPUT_DIR)
    print("\n[6] Summary …"); write_summary(rep, OUTPUT_DIR)
    print(f"\n✅ Done. Results in {os.path.abspath(OUTPUT_DIR)}/")


if __name__ == '__main__':
    main()