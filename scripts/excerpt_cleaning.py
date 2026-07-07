"""
excerpt_cleaning.py  —  stimulus-level quality control for the violin study
─────────────────────────────────────────────────────────────────────────────
Standalone companion to questionnaire_cleaning.py. Where that script screens
PARTICIPANTS, this one screens EXCERPTS (stimuli). It does NOT read the notebook
or any of its outputs — it re-parses the same questionnaire CSVs and re-derives
the excerpt/condition/piece mapping internally (embedded below), so it can be run
on its own.

WHY THIS EXISTS
  v1's questionnaire_cleaning.py tried to flag excerpts with a per-column "ICC"
  that was mathematically invalid (a monotonic function of variance, computed on
  a single column — the opposite of what ICC means), so it never flagged anything.
  This script replaces that with correct, citable stimulus diagnostics.

WHAT IT COMPUTES  (one row per excerpt unless noted)
  1. Coverage        — n_raters per excerpt (design: MEC/EXP appear in 2 sessions,
                       EXG in 1, so EXG excerpts legitimately have ~half the raters).
  2. Dispersion      — valence/arousal SD and IQR (how spread the ratings are).
  3. rWG agreement   — James/Demaree/Wolf within-group agreement vs a uniform null
                       on a 7-point scale (1 = perfect agreement, ~0 = none).
  4. Entropy         — Shannon entropy of the emotion-tag distribution (label
                       disagreement) and of the valence histogram.
  5. Bimodality      — Hartigan's dip test (if `diptest` installed) else Sarle's
                       bimodality coefficient. Separates "genuinely split" (two
                       camps → interesting) from "uniformly uncertain" (flat).
  6. Panel ICC(1,k)  — ONE overall reliability figure for valence and for arousal,
                       via one-way ANOVA (correct model here: each excerpt is rated
                       by a DIFFERENT random set of listeners, so ICC(1), not ICC(2)).
  7. Manipulation    — per PIECE, whether mean arousal follows the expected
     check             MEC ≤ EXP ≤ EXG ordering (expressive intensity → arousal).
                       A piece whose EXG is not more arousing than its MEC is a
                       manipulation-failure candidate. Kruskal–Wallis tests whether
                       the three conditions differ at all for that piece.

PHILOSOPHY (read before removing anything)
  Removing an excerpt removes it for every participant and shrinks an already tiny
  60-excerpt design, so this script FLAGS, it does not auto-remove. Disagreement
  alone is NOT grounds for removal — it is often the finding. Remove an excerpt
  only for an INDEPENDENT reason: a technical fault in the recording, or a clear
  manipulation failure. Report every flag and your decision transparently.

USAGE
  1. Edit CONFIG (CSV_GLOB, OUTPUT_DIR). The embedded EXCERPT_MAP is keyed by CSV
     *basename*, so only the folder in CSV_GLOB needs to match your machine.
  2. python excerpt_cleaning.py
─────────────────────────────────────────────────────────────────────────────
"""

import os, glob, re, warnings
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

try:
    from diptest import diptest as _diptest
    DIP_OK = True
except Exception:
    DIP_OK = False

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
CSV_GLOB = '/kaggle/input/datasets/george031218732003/violin-thesis-forms/Forms/questionnaire_*.csv'
OUTPUT_DIR     = '/kaggle/working/excerpt_cleaning_output'
DATA_START_COL = 3
COLS_PER_ITEM  = 3
N_EXCERPTS     = 20
VA_SCALE_MIN, VA_SCALE_MAX = 0, 6
N_SCALE_POINTS = 7           # 7 scale points (0..6); for rWG expected-variance null

# Flag thresholds (diagnostic, not automatic removal)
RWG_LOW        = 0.60        # rWG below this on val OR aro → low-agreement excerpt
SD_HIGH        = 1.40        # rating SD above this → high-dispersion excerpt
DIP_P          = 0.05        # dip-test p below this → significantly bimodal
BC_BIMODAL     = 0.555       # Sarle's coefficient above this → bimodal (fallback)
KW_P           = 0.05        # Kruskal–Wallis p above this → conditions indistinguishable

# ─── EMBEDDED MAPPING (keyed by CSV basename; mirrors the notebook CONFIG) ─────
CONDITIONS = {'MEC': 'Mechanical', 'EXP': 'Expressive', 'EXG': 'Exaggerated'}
EMOTION_LABELS = ['Tenderness', 'Nostalgia', 'Peacefulness', 'Power',
                  'Joyful Activation', 'Tension', 'Sadness', 'Neutral']
GREEK_TO_ENGLISH = {
    'Χαρά / Ενέργεια (Joyful Activation)': 'Joyful Activation',
    'Γαλήνη (Peacefulness)': 'Peacefulness', 'Νοσταλγία (Nostalgia)': 'Nostalgia',
    'Θλίψη (Sadness)': 'Sadness', 'Τρυφερότητα (Tenderness)': 'Tenderness',
    'Δύναμη (Power)': 'Power', 'Ένταση (Tension)': 'Tension',
    'Ουδέτερο (Neutral)': 'Neutral',
}
EXCERPT_MAP = {
    'questionnaire_1.csv': ['Bach_Adagio_S1_MEC', 'Bruch_VC1_EXG', 'Cinema_Paradiso_EXP',
        'Don_Juan_EXG', 'Godfather_MEC', 'La_Vita_E_Bella_MEC', 'Meditation_Thais_EXP',
        'Mendelssohn_VC_Mvt2_MEC', 'Mozart_VC3_EXG', 'Mozart_VC4_MEC', 'Schindlers_List_MEC',
        'Secret_Garden_EXP', 'Shostakovich_SQ8_exrpt1_EXP', 'Shostakovich_SQ8_exrpt2_MEC',
        'Shostakovich_SQ8_exrpt3_EXG', 'Shostakovich_Sym5_Mvt1_MEC', 'Shostakovich_Sym5_Mvt3_EXP',
        'Shostakovich_VC1_EXP', 'Tanzil_Serenade_EXP', 'Vitali_Chaconne_EXP'],
    'questionnaire_2.csv': ['Bach_Adagio_S1_EXG', 'Bruch_VC1_EXP', 'Cinema_Paradiso_EXP',
        'Don_Juan_EXP', 'Godfather_EXG', 'La_Vita_E_Bella_EXG', 'Meditation_Thais_MEC',
        'Mendelssohn_VC_Mvt2_EXG', 'Mozart_VC3_EXP', 'Mozart_VC4_MEC', 'Schindlers_List_MEC',
        'Secret_Garden_EXP', 'Shostakovich_SQ8_exrpt1_EXP', 'Shostakovich_SQ8_exrpt2_MEC',
        'Shostakovich_SQ8_exrpt3_EXP', 'Shostakovich_Sym5_Mvt1_MEC', 'Shostakovich_Sym5_Mvt3_MEC',
        'Shostakovich_VC1_MEC', 'Tanzil_Serenade_EXP', 'Vitali_Chaconne_MEC'],
    'questionnaire_3.csv': ['Bach_Adagio_S1_EXP', 'Bruch_VC1_EXP', 'Cinema_Paradiso_MEC',
        'Don_Juan_EXP', 'Godfather_EXP', 'La_Vita_E_Bella_EXP', 'Meditation_Thais_MEC',
        'Mendelssohn_VC_Mvt2_EXP', 'Mozart_VC3_EXP', 'Mozart_VC4_EXG', 'Schindlers_List_EXG',
        'Secret_Garden_MEC', 'Shostakovich_SQ8_exrpt1_MEC', 'Shostakovich_SQ8_exrpt2_EXG',
        'Shostakovich_SQ8_exrpt3_EXP', 'Shostakovich_Sym5_Mvt1_EXG', 'Shostakovich_Sym5_Mvt3_MEC',
        'Shostakovich_VC1_MEC', 'Tanzil_Serenade_MEC', 'Vitali_Chaconne_MEC'],
    'questionnaire_4.csv': ['Bach_Adagio_S1_EXP', 'Bruch_VC1_MEC', 'Cinema_Paradiso_MEC',
        'Don_Juan_MEC', 'Godfather_EXP', 'La_Vita_E_Bella_EXP', 'Meditation_Thais_EXG',
        'Mendelssohn_VC_Mvt2_EXP', 'Mozart_VC3_MEC', 'Mozart_VC4_EXP', 'Schindlers_List_EXP',
        'Secret_Garden_MEC', 'Shostakovich_SQ8_exrpt1_MEC', 'Shostakovich_SQ8_exrpt2_EXP',
        'Shostakovich_SQ8_exrpt3_MEC', 'Shostakovich_Sym5_Mvt1_EXP', 'Shostakovich_Sym5_Mvt3_EXG',
        'Shostakovich_VC1_EXG', 'Tanzil_Serenade_MEC', 'Vitali_Chaconne_EXG'],
    'questionnaire_5.csv': ['Bach_Adagio_S1_MEC', 'Bruch_VC1_MEC', 'Cinema_Paradiso_EXG',
        'Don_Juan_MEC', 'Godfather_MEC', 'La_Vita_E_Bella_MEC', 'Meditation_Thais_EXP',
        'Mendelssohn_VC_Mvt2_MEC', 'Mozart_VC3_MEC', 'Mozart_VC4_EXP', 'Schindlers_List_EXP',
        'Secret_Garden_EXG', 'Shostakovich_SQ8_exrpt1_EXG', 'Shostakovich_SQ8_exrpt2_EXP',
        'Shostakovich_SQ8_exrpt3_MEC', 'Shostakovich_Sym5_Mvt1_EXP', 'Shostakovich_Sym5_Mvt3_EXP',
        'Shostakovich_VC1_EXP', 'Tanzil_Serenade_EXG', 'Vitali_Chaconne_EXP'],
}
# ═══════════════════════════════════════════════════════════════════════════════

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/plots', exist_ok=True)


def condition_of(eid):
    for c in CONDITIONS:
        if eid.endswith('_' + c):
            return c
    return 'UNKNOWN'


def piece_of(eid):
    for c in CONDITIONS:
        if eid.endswith('_' + c):
            return eid[:-(len(c) + 1)]
    return eid


def parse_tags(cell):
    if pd.isna(cell) or str(cell).strip() == '':
        return []
    out = []
    for part in [p.strip() for p in str(cell).split(',')]:
        if part in GREEK_TO_ENGLISH:
            out.append(GREEK_TO_ENGLISH[part])
        else:
            m = re.search(r'\(([^)]+)\)', part)
            if m and m.group(1).strip() in EMOTION_LABELS:
                out.append(m.group(1).strip())
    return out


# ── LOAD → long format ────────────────────────────────────────────────────────
def load_long(csv_glob):
    files = sorted(glob.glob(csv_glob))
    if not files:
        raise FileNotFoundError(f"No CSVs match '{csv_glob}'. Update CSV_GLOB.")
    print(f"Found {len(files)} session file(s): {[Path(f).name for f in files]}")
    rows = []
    for fpath in files:
        base = Path(fpath).name
        if base not in EXCERPT_MAP:
            warnings.warn(f"No excerpt order for {base} — skipping."); continue
        order = EXCERPT_MAP[base]
        df = pd.read_csv(fpath, header=0)
        for p_idx, row in df.iterrows():
            pid = f'{Path(fpath).stem}_P{p_idx + 1:03d}'
            for i in range(N_EXCERPTS):
                col = DATA_START_COL + i * COLS_PER_ITEM
                if col + 2 >= len(row):
                    break
                try:
                    v = float(str(row.iloc[col + 1]).strip())
                    a = float(str(row.iloc[col + 2]).strip())
                    v = v if VA_SCALE_MIN <= v <= VA_SCALE_MAX else np.nan
                    a = a if VA_SCALE_MIN <= a <= VA_SCALE_MAX else np.nan
                except (ValueError, TypeError):
                    v = a = np.nan
                eid = order[i]
                rows.append({'participant_id': pid, 'excerpt_id': eid,
                             'piece': piece_of(eid), 'condition': condition_of(eid),
                             'valence': v, 'arousal': a,
                             'tags': parse_tags(row.iloc[col])})
    return pd.DataFrame(rows)


# ── metrics ───────────────────────────────────────────────────────────────────
def rwg(values):
    """James/Demaree/Wolf single-item rWG vs a uniform null on an N-point scale."""
    v = values[~np.isnan(values)]
    if len(v) < 2:
        return np.nan
    s2 = np.var(v, ddof=1)
    sigma2_eu = (N_SCALE_POINTS ** 2 - 1) / 12.0     # uniform-null variance
    return 1.0 - s2 / sigma2_eu                       # may be negative; report raw


def shannon_entropy(counts):
    c = np.array([x for x in counts if x > 0], dtype=float)
    if c.sum() == 0:
        return 0.0
    p = c / c.sum()
    return float(-np.sum(p * np.log2(p)))


def bimodality(values):
    """Return (is_bimodal, statistic, pvalue_or_nan, method)."""
    v = values[~np.isnan(values)]
    if len(v) < 4 or np.var(v) < 1e-9:
        return False, np.nan, np.nan, 'n/a'
    if DIP_OK:
        d, p = _diptest(v.astype(float))
        return (p < DIP_P), float(d), float(p), 'dip'
    # Sarle's bimodality coefficient fallback
    n = len(v)
    g = stats.skew(v, bias=False)
    k = stats.kurtosis(v, fisher=True, bias=False)
    denom = k + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)) if n > 3 else np.nan
    bc = (g ** 2 + 1) / denom if denom and denom > 0 else np.nan
    return (bc is not np.nan and bc > BC_BIMODAL), float(bc), np.nan, 'BC'


def panel_icc1(long_df, value_col):
    """Overall ICC(1) and ICC(1,k) via one-way ANOVA (targets = excerpts).

    Correct model here because each excerpt is rated by a different random set
    of listeners (raters are NOT crossed with targets).
    """
    d = long_df[['excerpt_id', value_col]].dropna()
    groups = [g[value_col].values for _, g in d.groupby('excerpt_id')]
    groups = [g for g in groups if len(g) > 0]
    m = len(groups)
    N = sum(len(g) for g in groups)
    if m < 2 or N <= m:
        return np.nan, np.nan, np.nan
    grand = np.concatenate(groups).mean()
    ss_b = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_w = sum(((g - g.mean()) ** 2).sum() for g in groups)
    ms_b = ss_b / (m - 1)
    ms_w = ss_w / (N - m)
    # unbalanced k0 correction
    ni = np.array([len(g) for g in groups])
    k0 = (N - (ni ** 2).sum() / N) / (m - 1)
    icc1 = (ms_b - ms_w) / (ms_b + (k0 - 1) * ms_w) if (ms_b + (k0 - 1) * ms_w) else np.nan
    icc1k = (ms_b - ms_w) / ms_b if ms_b else np.nan
    return float(icc1), float(icc1k), float(k0)


# ── EXCERPT-LEVEL REPORT ──────────────────────────────────────────────────────
def build_excerpt_report(long_df):
    recs = []
    for eid, g in long_df.groupby('excerpt_id'):
        v = g['valence'].values
        a = g['arousal'].values
        tag_counts = Counter(t for tags in g['tags'] for t in tags)
        val_hist = np.histogram(v[~np.isnan(v)],
                                bins=np.arange(VA_SCALE_MIN, VA_SCALE_MAX + 2))[0]
        bim_v = bimodality(v)
        rwg_v, rwg_a = rwg(v), rwg(a)
        sd_v = np.nanstd(v, ddof=1) if np.sum(~np.isnan(v)) > 1 else np.nan
        sd_a = np.nanstd(a, ddof=1) if np.sum(~np.isnan(a)) > 1 else np.nan
        low_agree = (np.nanmin([rwg_v, rwg_a]) < RWG_LOW)
        high_disp = (np.nanmax([sd_v, sd_a]) > SD_HIGH)
        recs.append({
            'excerpt_id': eid, 'piece': piece_of(eid), 'condition': condition_of(eid),
            'n_raters': int(np.sum(~np.isnan(v))),
            'valence_mean': np.nanmean(v), 'valence_sd': sd_v,
            'arousal_mean': np.nanmean(a), 'arousal_sd': sd_a,
            'valence_iqr': np.nanpercentile(v, 75) - np.nanpercentile(v, 25)
                           if np.sum(~np.isnan(v)) else np.nan,
            'rwg_valence': rwg_v, 'rwg_arousal': rwg_a,
            'tag_entropy': shannon_entropy(list(tag_counts.values())),
            'valence_entropy': shannon_entropy(val_hist),
            'bimodal_valence': bim_v[0], 'bimodal_stat': bim_v[1],
            'bimodal_p': bim_v[2], 'bimodal_method': bim_v[3],
            'top_tags': tag_counts.most_common(3),
            'flag_low_agreement': bool(low_agree),
            'flag_high_dispersion': bool(high_disp),
            # "contested" = disagreement present; sub-type tells you what kind
            'contested_type': ('split/bimodal' if bim_v[0]
                               else ('flat/uncertain' if (low_agree or high_disp)
                                     else 'consensus')),
        })
    df = pd.DataFrame(recs).sort_values(['piece', 'condition']).reset_index(drop=True)
    return df


# ── PIECE-LEVEL MANIPULATION CHECK ────────────────────────────────────────────
def build_manipulation_report(long_df):
    """Per piece: does mean arousal follow MEC ≤ EXP ≤ EXG (expressive intensity)?"""
    recs = []
    for piece, g in long_df.groupby('piece'):
        means = {c: g.loc[g['condition'] == c, 'arousal'].mean() for c in CONDITIONS}
        vmeans = {c: g.loc[g['condition'] == c, 'valence'].mean() for c in CONDITIONS}
        have = [c for c in ['MEC', 'EXP', 'EXG'] if not np.isnan(means[c])]
        # Kruskal–Wallis across available conditions (response-level arousal)
        samples = [g.loc[g['condition'] == c, 'arousal'].dropna().values for c in have]
        samples = [s for s in samples if len(s) > 0]
        if len(samples) >= 2 and all(len(s) >= 3 for s in samples):
            kw_h, kw_p = stats.kruskal(*samples)
        else:
            kw_h, kw_p = np.nan, np.nan
        monotonic = (not np.isnan(means['MEC']) and not np.isnan(means['EXP'])
                     and not np.isnan(means['EXG'])
                     and means['MEC'] <= means['EXP'] <= means['EXG'])
        inversion = (not np.isnan(means['MEC']) and not np.isnan(means['EXG'])
                     and means['EXG'] < means['MEC'])
        recs.append({
            'piece': piece,
            'arousal_MEC': means['MEC'], 'arousal_EXP': means['EXP'], 'arousal_EXG': means['EXG'],
            'valence_MEC': vmeans['MEC'], 'valence_EXP': vmeans['EXP'], 'valence_EXG': vmeans['EXG'],
            'delta_EXG_minus_MEC': (means['EXG'] - means['MEC'])
                                   if not (np.isnan(means['EXG']) or np.isnan(means['MEC'])) else np.nan,
            'kw_H': kw_h, 'kw_p': kw_p,
            'arousal_monotonic': bool(monotonic),
            'flag_inversion': bool(inversion),                 # EXG less arousing than MEC
            'flag_conditions_indistinct': bool(not np.isnan(kw_p) and kw_p > KW_P),
        })
    return pd.DataFrame(recs).sort_values('delta_EXG_minus_MEC').reset_index(drop=True)


# ── PLOTS ─────────────────────────────────────────────────────────────────────
def make_plots(exc, man, icc, output_dir):
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.25)

    ax = fig.add_subplot(gs[0, 0])
    colors = ['red' if f else 'steelblue' for f in exc['flag_low_agreement']]
    ax.bar(range(len(exc)), exc['rwg_valence'].fillna(0), color=colors)
    ax.axhline(RWG_LOW, color='red', ls='--', label=f'rWG={RWG_LOW}')
    ax.set_title('rWG agreement (valence) per excerpt')
    ax.set_xlabel('excerpt'); ax.set_ylabel('rWG'); ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(exc['arousal_sd'], exc['valence_sd'],
               c=['red' if f else 'teal' for f in exc['flag_high_dispersion']], s=30)
    ax.axvline(SD_HIGH, color='red', ls='--'); ax.axhline(SD_HIGH, color='red', ls='--')
    ax.set_title('Rating dispersion'); ax.set_xlabel('arousal SD'); ax.set_ylabel('valence SD')

    ax = fig.add_subplot(gs[1, 0])
    y = np.arange(len(man))
    colors = ['red' if f else 'seagreen' for f in man['flag_inversion']]
    ax.barh(y, man['delta_EXG_minus_MEC'].fillna(0), color=colors)
    ax.axvline(0, color='black', lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels(man['piece'], fontsize=7)
    ax.set_title('Manipulation check: arousal(EXG) − arousal(MEC)\n'
                 'red = inversion (EXG not more arousing than MEC)')
    ax.set_xlabel('Δ arousal')

    ax = fig.add_subplot(gs[1, 1])
    ax.hist(exc['tag_entropy'].dropna(), bins=15, color='mediumpurple', edgecolor='white')
    ax.set_title('Emotion-tag entropy (label disagreement) per excerpt')
    ax.set_xlabel('Shannon entropy (bits)'); ax.set_ylabel('excerpts')

    plt.suptitle(f'Excerpt / Stimulus QC — panel ICC(1,k) valence={icc["val_k"]:.2f}, '
                 f'arousal={icc["aro_k"]:.2f}', fontsize=13)
    out = f'{output_dir}/plots/excerpt_overview.png'
    fig.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f'  Saved: {out}')


# ── SUMMARY ───────────────────────────────────────────────────────────────────
def write_summary(exc, man, icc, output_dir):
    lines = [
        "EXCERPT / STIMULUS QC REPORT", "=" * 60,
        f"Excerpts analysed : {len(exc)}   |   Pieces : {man.shape[0]}",
        f"Rater coverage    : MEC/EXP≈2 sessions, EXG≈1 session (by design)", "",
        "── PANEL RELIABILITY (one-way ICC(1); higher = more reliable) ──",
        f"  Valence : ICC(1)={icc['val_1']:.3f}   ICC(1,k)={icc['val_k']:.3f}  (avg k≈{icc['val_k0']:.0f})",
        f"  Arousal : ICC(1)={icc['aro_1']:.3f}   ICC(1,k)={icc['aro_k']:.3f}  (avg k≈{icc['aro_k0']:.0f})",
        "",
        "── LOW-AGREEMENT / HIGH-DISPERSION EXCERPTS (diagnostic) ──",
    ]
    contested = exc[exc['flag_low_agreement'] | exc['flag_high_dispersion']]
    if len(contested) == 0:
        lines.append("  none")
    for _, r in contested.iterrows():
        lines.append(f"  {r['excerpt_id']:32s} [{r['contested_type']}]  "
                     f"rWG_v={r['rwg_valence']:.2f} rWG_a={r['rwg_arousal']:.2f}  "
                     f"SD_v={r['valence_sd']:.2f} SD_a={r['arousal_sd']:.2f}  n={int(r['n_raters'])}")
    lines += ["", "── MANIPULATION-CHECK FLAGS (candidate stimulus problems) ──"]
    inv = man[man['flag_inversion']]
    if len(inv) == 0:
        lines.append("  No arousal inversions — every piece's EXG ≥ MEC. Good.")
    for _, r in inv.iterrows():
        lines.append(f"  {r['piece']:28s} EXG−MEC arousal = {r['delta_EXG_minus_MEC']:+.2f}  "
                     f"(MEC={r['arousal_MEC']:.2f} EXP={r['arousal_EXP']:.2f} EXG={r['arousal_EXG']:.2f})")
    indist = man[man['flag_conditions_indistinct']]
    if len(indist):
        lines.append("")
        lines.append("  Pieces where the 3 conditions are NOT statistically distinguishable")
        lines.append("  (Kruskal–Wallis p > %.2f on arousal):" % KW_P)
        for _, r in indist.iterrows():
            lines.append(f"    {r['piece']:28s} KW p={r['kw_p']:.3f}")
    lines += [
        "", "── HOW TO USE THESE FLAGS ───────────────────────────────",
        "  • Disagreement / low rWG is NOT grounds for removal — it is often the",
        "    finding (a 'split/bimodal' excerpt = two genuine interpretations).",
        "  • Remove an excerpt only for an INDEPENDENT reason: a technical fault",
        "    in the recording, or a manipulation failure (inversion below).",
        "  • Report all flags, your decision per excerpt, and whether any stimulus",
        "    was removed and why.",
    ]
    text = "\n".join(lines)
    with open(f'{output_dir}/excerpt_summary.txt', 'w') as f:
        f.write(text)
    print(f'  Saved: {output_dir}/excerpt_summary.txt\n'); print(text)


# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  Excerpt / Stimulus QC — Violin Emotion Study")
    print("=" * 60)
    print(f"  (bimodality via {'Hartigan dip test' if DIP_OK else 'Sarle BC fallback — pip install diptest for the dip test'})")

    print("\n[1] Loading + mapping responses …")
    long_df = load_long(CSV_GLOB)
    print(f"    {len(long_df)} responses | {long_df['excerpt_id'].nunique()} excerpts | "
          f"{long_df['piece'].nunique()} pieces | {long_df['participant_id'].nunique()} participants")

    print("\n[2] Panel reliability (ICC(1)) …")
    v1, vk, vk0 = panel_icc1(long_df, 'valence')
    a1, ak, ak0 = panel_icc1(long_df, 'arousal')
    icc = {'val_1': v1, 'val_k': vk, 'val_k0': vk0,
           'aro_1': a1, 'aro_k': ak, 'aro_k0': ak0}
    print(f"    valence ICC(1,k)={vk:.3f} | arousal ICC(1,k)={ak:.3f}")

    print("\n[3] Excerpt-level diagnostics …")
    exc = build_excerpt_report(long_df)
    exc.to_csv(f'{OUTPUT_DIR}/excerpt_report.csv', index=False)
    print(f"    Saved: {OUTPUT_DIR}/excerpt_report.csv "
          f"({int((exc['flag_low_agreement'] | exc['flag_high_dispersion']).sum())} contested)")

    print("\n[4] Per-piece manipulation check …")
    man = build_manipulation_report(long_df)
    man.to_csv(f'{OUTPUT_DIR}/piece_manipulation_report.csv', index=False)
    print(f"    Saved: {OUTPUT_DIR}/piece_manipulation_report.csv "
          f"({int(man['flag_inversion'].sum())} inversion flag(s))")

    print("\n[5] Plots …");   make_plots(exc, man, icc, OUTPUT_DIR)
    print("\n[6] Summary …"); write_summary(exc, man, icc, OUTPUT_DIR)
    print(f"\n✅ Done. Results in {os.path.abspath(OUTPUT_DIR)}/")


if __name__ == '__main__':
    main()