"""Stage 2 - hypothesis contrasts for one factor, 2 or more levels.

Compares every level of a grouping factor across every numeric variable of a
dataset, choosing the test from the data and writing one plain CSV table per
dataset (raw numbers, no styling, no narrative document). Two levels
dispatch to ``compare_two_groups`` (inherited from amacrine-motion-detection's analysis code);
three or more levels dispatch to ``compare_n_groups`` (added for this
paper's multi-group designs — e.g. a within-genotype treatment x
timepoint factor). Every comparison here is single-factor; genotype
stratification and the between-genotype ratio datasets are prepared
upstream by ``ratio.py``, not modelled as a second factor in this module.

Test selection — 2 groups
--------------------------
    normality per group : Shapiro-Wilk, alpha = 0.05
                          (D'Agostino K^2 above ``max_n_shapiro``)
    equal variances     : Bartlett when both groups pass normality

    paired   + normal      -> paired t-test            effect: Cohen's dz
    paired   + non-normal  -> Wilcoxon signed-rank      effect: r
    unpaired + normal + equal var    -> Student's t     effect: Cohen's d
    unpaired + normal + unequal var  -> Welch's t       effect: Cohen's d
    unpaired + non-normal            -> Mann-Whitney U  effect: r

Test selection — 3+ groups
----------------------------
    unpaired + normal + equal var    -> One-way ANOVA        effect: eta^2
    unpaired + normal + unequal var  -> Welch's ANOVA         effect: eta^2
    unpaired + non-normal            -> Kruskal-Wallis        effect: epsilon^2
    paired   + normal                -> RM-ANOVA (+ Greenhouse-Geisser
                                          when Mauchly's sphericity fails)
                                                                effect: eta^2
    paired   + non-normal            -> Friedman              effect: epsilon^2
                                          (bootstrap CI stays NaN — resampling
                                          breaks the within-subject pairing)

Post-hoc (only run when the omnibus test is significant), priority order:
    1. Tukey HSD         — omnibus was One-way ANOVA (equal variances)
    2. Games-Howell      — omnibus was Welch's ANOVA (unequal variances)
    3. Holm-Bonferroni   — otherwise (Kruskal-Wallis via pairwise
                           Mann-Whitney, paired designs via pairwise paired
                           t-test/Wilcoxon)

Benjamini-Hochberg FDR is applied across every variable tested in one
dataset's table (one dataset = one family; see ``multicomparison.bh_fdr``),
reported alongside the raw, uncorrected p-value rather than replacing it —
for both the 2-group and 3+-group paths.

Designs measured at more than two levels declare a ``levels`` pair in the
registry to restrict a run to just two of them (selecting which two and
their order); leaving ``levels`` unset runs every level found through the
3+-group path instead.

Output
------
2 groups: one CSV table per dataset, as in amacrine-motion-detection's analysis code — test,
p-value, significance class, effect size, n/mean/SD/median per group,
direction, pretest p-values, Hedges' g, bootstrap CI, signed rank-biserial,
BH-FDR q-value.

3+ groups: one CSV omnibus table per dataset (one row per variable) plus
one CSV post-hoc table (one row per variable x pairwise comparison, only
for variables whose omnibus test was significant).
"""

from __future__ import annotations

import os
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import scipy.stats as stats

from .effect_sizes import (
    _cohen_d,
    _effect_r_from_u,
    _stars,
    _wilcoxon_r,
    bootstrap_ci,
    describe_group,
    hedges_g,
    rank_biserial,
)
from .multicomparison import bh_fdr, holm_adjust
from .reporting import save_csv

# Alpha for the normality and variance pretests and for the significance ladder,
# stated once rather than repeated as a literal at each decision point.
ALPHA = 0.05


# ============================================================================= #
# SECTION 1. DATA LOADING
# ============================================================================= #


# Column-name suffixes that are numeric/boolean but are diagnostic
# metadata, never a variable to test — e.g. the `{col}_outlier` flag
# columns preprocessing.detect_and_replace_outliers adds for cells its
# skip gate withheld. Kept in sync with preprocessing.NON_MEASURE_SUFFIXES.
NON_MEASURE_SUFFIXES = ("_outlier",)


def load_dataset(name: str, cfg: dict) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Load one preprocessed dataset and split its columns by role.

    A column is treated as categorical when pandas does not consider its
    dtype numeric (covers legacy ``object`` string columns as well as
    pandas >= 3's default PyArrow-backed ``str`` dtype for CSV text columns
    — checking ``dtype == "object"`` alone, as an earlier version of this
    function did, silently stops catching text columns under pandas 3 and
    lets them leak into ``numeric_cols`` instead), or when its name contains
    the substring ``id`` (case-insensitive), which keeps subject identifiers
    out of the numeric analysis. Columns ending in NON_MEASURE_SUFFIXES are
    also routed to categorical_cols even though pandas parses their
    True/False values as boolean — they are diagnostic flags, not a
    variable to contrast (booleans would also break the parametric/
    non-parametric tests below, which assume a continuous DV).

    Parameters
    ----------
    name : str
        Dataset key, used for the progress message.
    cfg : dict
        Configuration entry; must provide ``location``.

    Returns
    -------
    tuple
        ``(dataframe, numeric_columns, categorical_columns)``.
    """
    df = pd.read_csv(cfg["location"])

    categorical_cols = [
        col for col in df.columns
        if not pd.api.types.is_numeric_dtype(df[col])
        or df[col].dtype == "bool"
        or "id" in col.lower()
        or str(col).lower().endswith(NON_MEASURE_SUFFIXES)
    ]
    numeric_cols = [col for col in df.columns if col not in categorical_cols]

    print(f"Loaded '{name}' with {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Numeric columns: {numeric_cols}")
    print(f"Categorical columns: {categorical_cols}")

    return df, numeric_cols, categorical_cols


# ============================================================================= #
# SECTION 2. SAMPLE EXTRACTION AND PAIRING
# ============================================================================= #


def _extract_samples(
    df: pd.DataFrame,
    group_col: str,
    var: str,
    group1,
    group2,
    paired: bool,
    subject_col: str | None,
    dataset: str = "",
) -> tuple[pd.Series, pd.Series, str | None]:
    """Return the two samples to compare, aligned by subject when paired.

    Paired samples are matched through ``subject_col`` so that each subject's
    two values are compared with each other regardless of row order.
    Where the identifier cannot resolve the pairing, the samples are taken in
    row order instead, which assumes both groups list their subjects in the
    same sequence.

    Returns
    -------
    tuple
        ``(sample1, sample2, problem)``, where ``problem`` is None on success or
        a short description of why the contrast cannot be run.
    """
    if not paired:
        return (
            df.loc[df[group_col] == group1, var].dropna(),
            df.loc[df[group_col] == group2, var].dropna(),
            None,
        )

    if subject_col is None or subject_col not in df.columns:
        s1 = df.loc[df[group_col] == group1, var].dropna()
        s2 = df.loc[df[group_col] == group2, var].dropna()
        return s1, s2, None if len(s1) == len(s2) else "Paired size mismatch"

    subset = df[[subject_col, group_col, var]].dropna(subset=[var])

    if subset.duplicated(subset=[subject_col, group_col]).any():
        # An identifier appears more than once within a group, so the pairing
        # cannot be resolved from it; fall back to row order.
        s1 = df.loc[df[group_col] == group1, var].dropna()
        s2 = df.loc[df[group_col] == group2, var].dropna()
        return s1, s2, None if len(s1) == len(s2) else "Paired size mismatch"

    wide = subset.pivot(index=subject_col, columns=group_col, values=var)
    if group1 not in wide.columns or group2 not in wide.columns:
        return pd.Series(dtype=float), pd.Series(dtype=float), "Paired size mismatch"

    complete = wide[[group1, group2]].dropna()
    dropped = len(wide) - len(complete)
    if dropped:
        warnings.warn(
            f"[{dataset}:{var}] {dropped} subject(s) dropped for lacking one "
            f"member of the pair.",
            RuntimeWarning,
            stacklevel=2,
        )

    return complete[group1], complete[group2], None


# ============================================================================= #
# SECTION 3. ASSUMPTION PRETESTS
# ============================================================================= #


def _normality(data1, data2, max_n_shapiro: int) -> tuple[str, float, float]:
    """Per-group normality test name and p-values, by size.

    Shapiro-Wilk up to ``max_n_shapiro`` combined n, D'Agostino K^2 above
    it. On failure both p-values are set to 0, routing the contrast to the
    non-parametric branch.
    """
    n_total = len(data1) + len(data2)
    name = "Shapiro-Wilk" if n_total <= max_n_shapiro else "D'Agostino K^2"
    try:
        if n_total <= max_n_shapiro:
            return name, stats.shapiro(data1)[1], stats.shapiro(data2)[1]
        return name, stats.normaltest(data1)[1], stats.normaltest(data2)[1]
    except Exception:
        return name, 0.0, 0.0


def _equal_variance(data1, data2, both_normal: bool) -> tuple[str, float]:
    """Variance-homogeneity pretest and the name of the test used.

    Bartlett is applied when both groups pass normality, which is the case in
    which the result is consulted to choose between Student's and Welch's t.
    Brown-Forsythe and Fligner cover the non-normal case, where the contrast
    proceeds to Mann-Whitney.
    """
    n_total = len(data1) + len(data2)
    if both_normal:
        name = "Bartlett"
    elif n_total > 1000:
        name = "Fligner"
    else:
        name = "Brown-Forsythe"

    try:
        if name == "Bartlett":
            return name, stats.bartlett(data1, data2)[1]
        if name == "Fligner":
            return name, stats.fligner(data1, data2)[1]
        return name, stats.levene(data1, data2, center="median")[1]
    except Exception:
        return name, np.nan


# ============================================================================= #
# SECTION 4. THE TWO-GROUP CONTRAST
# ============================================================================= #


def _run_test(data1, data2, paired: bool, both_normal: bool, equal_var: bool):
    """Apply the selected test and its matching effect size.

    Returns ``(test_name, statistic, p_value, effect_value, effect_metric)``.
    """
    if paired:
        if both_normal:
            stat, p_value = stats.ttest_rel(data1, data2)
            return ("Paired t-test", stat, p_value,
                    _cohen_d(data1.values, data2.values, paired=True), "Cohen's dz")

        stat, p_value = stats.wilcoxon(data1, data2)
        return ("Wilcoxon signed-rank test", stat, p_value,
                _wilcoxon_r(p_value, len(data1)), "r")

    if both_normal:
        if equal_var:
            stat, p_value = stats.ttest_ind(data1, data2, equal_var=True)
            test_name = "Student's t-test"
        else:
            stat, p_value = stats.ttest_ind(data1, data2, equal_var=False)
            test_name = "Welch's t-test"
        return (test_name, stat, p_value,
                _cohen_d(data1.values, data2.values, paired=False), "Cohen's d")

    stat, p_value = stats.mannwhitneyu(data1, data2, alternative="two-sided")
    return ("Mann-Whitney U test", stat, p_value,
            _effect_r_from_u(stat, len(data1), len(data2)), "r")


def compare_two_groups(
    dataframes: dict,
    comparison_by: dict,
    variables: list,
    folder: str,
    paired: dict,
    subject: dict | None = None,
    table_prefix: str = "Group comparisons",
    min_n: int = 3,
    max_n_shapiro: int = 5000,
    with_intervals: bool = True,
    verbose: bool = True,
) -> dict:
    """Compare two groups across numeric variables and write one CSV table.

    For each dataset and each numeric variable this runs the normality and
    variance pretests, selects the test as documented in the module docstring
    and records the result together with the descriptive statistics needed to
    read the direction of the effect.

    Parameters
    ----------
    dataframes : dict
        ``{dataset name: DataFrame}``.
    comparison_by : dict
        ``{dataset name: grouping column}``.
    variables : list
        Numeric variables to test.
    folder : str
        Output directory for the CSV table.
    paired : dict
        ``{dataset name: bool}``.
    subject : dict, optional
        ``{dataset name: subject column}``, used to align paired samples.
    table_prefix : str
        Leading text of the output file name.
    min_n : int
        Minimum observations per group; below this the contrast is skipped.
    max_n_shapiro : int
        Above this total n, normality uses D'Agostino K^2 instead of Shapiro-Wilk.
    with_intervals : bool
        Compute bootstrap confidence intervals. Set False for a fast pass.
    verbose : bool
        Print progress.

    Returns
    -------
    dict
        ``{dataset name: {variable: result dict}}``.
    """
    os.makedirs(folder, exist_ok=True)
    subject = subject or {}
    results_all: dict[str, dict] = {}

    for name, df in dataframes.items():
        grp_col = comparison_by.get(name)
        is_paired = paired.get(name, False)
        subject_col = subject.get(name)

        if grp_col is None:
            if verbose:
                print(f"No comparison column defined for '{name}'. Skipping.")
            continue

        # Order of appearance, not alphabetical: this fixes which level is
        # group 1 and therefore the sign of every signed effect size.
        groups = df[grp_col].dropna().unique()
        if len(groups) != 2:
            if verbose:
                print(f"Dataset '{name}' has {len(groups)} levels in "
                      f"'{grp_col}', expected 2: {list(groups)}")
            continue

        group1, group2 = groups
        res_dict: dict[str, dict] = {}

        if verbose:
            print(f"\nProcessing '{name}': {group1} vs {group2} "
                  f"({'paired' if is_paired else 'independent'})")

        for var in variables:
            if var not in df.columns:
                continue

            data1, data2, problem = _extract_samples(
                df, grp_col, var, group1, group2, is_paired, subject_col, name)

            if problem is None and (len(data1) < min_n or len(data2) < min_n):
                problem = "Insufficient data"

            if problem is not None:
                res_dict[var] = _blank_row(group1, group2, problem, data1, data2)
                continue

            normality_test_name, p_norm1, p_norm2 = _normality(data1, data2, max_n_shapiro)
            both_normal = p_norm1 > ALPHA and p_norm2 > ALPHA

            variance_test, p_variance = _equal_variance(data1, data2, both_normal)
            equal_var = bool(p_variance > ALPHA) if np.isfinite(p_variance) else False

            test_name, _stat, p_value, effect_value, effect_metric = _run_test(
                data1, data2, is_paired, both_normal, equal_var)

            row = _blank_row(group1, group2, test_name, data1, data2,
                              normality_test=normality_test_name,
                              p_norm1=p_norm1, p_norm2=p_norm2,
                              variance_test=variance_test, p_variance=p_variance)
            row.update({
                "p_value": round(float(p_value), 6),
                "significance": _stars(p_value if p_value is not None else 1),
                "effect size metric": effect_metric,
                "effect size value": (
                    round(float(effect_value), 3)
                    if effect_value is not None and not np.isnan(effect_value)
                    else np.nan
                ),
                "Hedges g": round(hedges_g(data1.values, data2.values, is_paired), 3),
                "rank-biserial (signed)": round(
                    rank_biserial(data1.values, data2.values, is_paired), 3),
            })

            if with_intervals:
                low, high = bootstrap_ci(data1.values, data2.values, paired=is_paired)
                row["effect 95% CI"] = (
                    f"[{low:.2f}, {high:.2f}]" if np.isfinite(low) else "n/a")

            res_dict[var] = row

            if verbose:
                print(f"  {var}: {test_name}, p = {p_value:.4g} "
                      f"{row['significance']}, {effect_metric} = "
                      f"{row['effect size value']}")

        # Benjamini-Hochberg FDR across every variable tested in THIS
        # table (one dataset = one family). Adds two columns without
        # touching the raw p-value, its significance stars, or any
        # effect size — a reader who wants the uncorrected result still
        # has it. Variables skipped above (blank rows, p_value = NaN)
        # pass through untouched, per bh_fdr's NaN handling.
        variables_in_table = list(res_dict.keys())
        q_values = bh_fdr([res_dict[v]["p_value"] for v in variables_in_table])
        for var, q in zip(variables_in_table, q_values):
            res_dict[var]["p_value_fdr_bh"] = (
                round(float(q), 6) if np.isfinite(q) else np.nan)
            res_dict[var]["significance_fdr"] = _stars(q) if np.isfinite(q) else "n/a"

        results_all[name] = res_dict

        result_df = pd.DataFrame(
            [{"Variable": var, **res} for var, res in res_dict.items()])
        csv_path = os.path.join(folder, f"{table_prefix} - {name}.csv")
        save_csv(result_df, csv_path)

        if verbose:
            print(f"Results saved for '{name}' -> {csv_path}")

    return results_all


def _blank_row(group1, group2, test_label: str, data1=None, data2=None, *,
                normality_test="n/a", p_norm1=np.nan, p_norm2=np.nan,
                variance_test="n/a", p_variance=np.nan) -> dict:
    """Build a result row carrying the descriptives, with no test outcome yet.

    Descriptives are attached even when the contrast is skipped, so that the
    table still shows how much data each variable had. ``direction`` states in
    words which group scores higher. The assumption-check columns
    (normality/homoscedasticity) are positioned *before* the test columns,
    so the reason a test was chosen reads before the result it produced;
    they default to "n/a" when the contrast was skipped before those
    pretests ever ran.
    """
    d1 = describe_group(data1 if data1 is not None else [])
    d2 = describe_group(data2 if data2 is not None else [])

    if np.isfinite(d1["mean"]) and np.isfinite(d2["mean"]):
        if d1["mean"] > d2["mean"]:
            direction = f"{group1} > {group2}"
        elif d1["mean"] < d2["mean"]:
            direction = f"{group1} < {group2}"
        else:
            direction = "equal"
    else:
        direction = "n/a"

    return {
        "group 1": group1,
        "group 2": group2,
        "n (group 1)": d1["n"],
        "n (group 2)": d2["n"],
        "mean (group 1)": round(d1["mean"], 3) if np.isfinite(d1["mean"]) else np.nan,
        "SD (group 1)": round(d1["sd"], 3) if np.isfinite(d1["sd"]) else np.nan,
        "mean (group 2)": round(d2["mean"], 3) if np.isfinite(d2["mean"]) else np.nan,
        "SD (group 2)": round(d2["sd"], 3) if np.isfinite(d2["sd"]) else np.nan,
        "direction": direction,
        "normality test": normality_test,
        "normality p (group 1)": round(float(p_norm1), 4) if np.isfinite(p_norm1) else np.nan,
        "normality significance (group 1)": _stars(p_norm1) if np.isfinite(p_norm1) else "n/a",
        "normality p (group 2)": round(float(p_norm2), 4) if np.isfinite(p_norm2) else np.nan,
        "normality significance (group 2)": _stars(p_norm2) if np.isfinite(p_norm2) else "n/a",
        "homoscedasticity test": variance_test,
        "homoscedasticity p": round(float(p_variance), 4) if np.isfinite(p_variance) else np.nan,
        "homoscedasticity significance": _stars(p_variance) if np.isfinite(p_variance) else "n/a",
        "test": test_label,
        "p_value": np.nan,
        "significance": "ns",
        "effect size metric": "n/a",
        "effect size value": np.nan,
    }


# ============================================================================= #
# SECTION 4B. THREE-OR-MORE-GROUP CONTRASTS (one factor, N levels)
#
# Sibling of section 4 above for a factor with 3+ levels: the omnibus test,
# post-hoc dispatch and their supporting helpers.
# ============================================================================= #


_BOOTSTRAP_N_OMNIBUS = 2000
_BOOTSTRAP_CI_OMNIBUS = 0.95
_BOOTSTRAP_SEED_OMNIBUS = 42


def _extract_samples_n(
    df: pd.DataFrame,
    group_col: str,
    var: str,
    group_labels: list,
    paired: bool,
    subject_col: str | None,
    dataset: str = "",
) -> tuple[list, str | None]:
    """N-group generalization of ``_extract_samples``: one Series per
    label, in the given order. Paired samples are aligned by
    ``subject_col`` — complete cases across EVERY label at once, not
    just a pair — so a subject missing even one condition is dropped
    from that variable's model entirely."""
    if not paired:
        return ([df.loc[df[group_col] == g, var].dropna()
                  for g in group_labels], None)

    if subject_col is None or subject_col not in df.columns:
        series = [df.loc[df[group_col] == g, var].dropna()
                   for g in group_labels]
        problem = ("Paired size mismatch"
                    if len({len(s) for s in series}) > 1 else None)
        return series, problem

    subset = df[[subject_col, group_col, var]].dropna(subset=[var])
    if subset.duplicated(subset=[subject_col, group_col]).any():
        series = [df.loc[df[group_col] == g, var].dropna()
                   for g in group_labels]
        problem = ("Paired size mismatch"
                    if len({len(s) for s in series}) > 1 else None)
        return series, problem

    wide = subset.pivot(index=subject_col, columns=group_col, values=var)
    missing_labels = [g for g in group_labels if g not in wide.columns]
    if missing_labels:
        return ([pd.Series(dtype=float)] * len(group_labels),
                 "Paired size mismatch")

    complete = wide[group_labels].dropna()
    dropped = len(wide) - len(complete)
    if dropped:
        warnings.warn(
            f"[{dataset}:{var}] {dropped} subject(s) dropped for lacking "
            f"a member of every level.", RuntimeWarning, stacklevel=2)

    return [complete[g] for g in group_labels], None


def _bootstrap_ci_general(stat_fn, *groups,
                           n_iter: int = _BOOTSTRAP_N_OMNIBUS,
                           ci: float = _BOOTSTRAP_CI_OMNIBUS,
                           seed: int = _BOOTSTRAP_SEED_OMNIBUS,
                           min_n_per_group: int = 2,
                           clip: tuple | None = None) -> tuple[float, float]:
    """Percentile bootstrap CI for a statistic computed from N
    independently-resampled groups (each group resampled within
    itself). Generalizes ``effect_sizes.bootstrap_ci`` (which is
    2-group/Cohen's-d specific) to the omnibus eta-squared/epsilon-
    squared estimators and to N-group post-hoc effect sizes below."""
    sizes = [len(g) for g in groups]
    if any(n < min_n_per_group for n in sizes):
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    estimates = np.empty(n_iter, dtype=float)
    for i in range(n_iter):
        resampled = tuple(rng.choice(np.asarray(g), size=n, replace=True)
                           for g, n in zip(groups, sizes))
        try:
            estimates[i] = stat_fn(*resampled)
        except Exception:
            estimates[i] = np.nan

    valid = estimates[~np.isnan(estimates)]
    if len(valid) < 10:
        return float("nan"), float("nan")
    alpha = (1.0 - ci) / 2.0
    lo = float(np.percentile(valid, 100.0 * alpha))
    hi = float(np.percentile(valid, 100.0 * (1.0 - alpha)))
    if clip is not None:
        c_lo, c_hi = clip
        lo, hi = max(c_lo, lo), min(c_hi, hi)
    return lo, hi


def _welch_anova(groups: list) -> tuple[float, float]:
    """Welch's (1951) ANOVA F-test for unequal variances. Manual
    implementation, no scipy/pingouin equivalent exists."""
    k = len(groups)
    ns = np.array([len(g) for g in groups], dtype=float)
    ms = np.array([np.mean(g) for g in groups], dtype=float)
    vs = np.array([np.var(g, ddof=1) for g in groups], dtype=float)

    ws = ns / (vs + 1e-300)
    W = ws.sum()
    x_w = (ws * ms).sum() / W
    num = (ws * (ms - x_w) ** 2).sum() / (k - 1)
    lam = (((1 - ws / W) ** 2) / (ns - 1)).sum() * (2 * (k - 2)) / (k ** 2 - 1)
    F = num / (1 + lam)
    df2_inv = (((1 - ws / W) ** 2) / (ns - 1)).sum() * 3 / (k ** 2 - 1)
    df2 = 1 / df2_inv if df2_inv > 0 else 1e6
    df1 = k - 1
    p = float(1 - stats.f.cdf(F, df1, df2))
    return float(F), p


def _eta_from_groups(*groups) -> float:
    """eta-squared computed directly from group arrays (usable both
    directly and as the statistic bootstrapped by
    ``_bootstrap_ci_general``)."""
    all_vals = np.concatenate([np.asarray(g) for g in groups])
    grand_mean = all_vals.mean()
    ss_total = np.sum((all_vals - grand_mean) ** 2)
    if ss_total <= 0:
        return 0.0
    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
    return float(ss_between / ss_total)


def _eps_from_kruskal(*groups) -> float:
    """epsilon-squared from Kruskal-Wallis H = (H - k + 1) / (n - k)."""
    try:
        stat, _ = stats.kruskal(*groups)
    except ValueError:
        return 0.0
    n = sum(len(g) for g in groups)
    k = len(groups)
    return float((stat - k + 1) / (n - k)) if n > k else 0.0


def _rm_anova_with_sphericity(groups: list, labels: list) -> tuple:
    """RM-ANOVA with Greenhouse-Geisser correction applied when
    Mauchly's test of sphericity is violated. Requires pingouin
    (already a project dependency)."""
    try:
        import pingouin as pg
    except Exception:
        F, p = stats.f_oneway(*groups)
        return F, p, "sphericity not tested (pingouin unavailable)"

    n = len(groups[0])
    long_df = pd.DataFrame({
        "subject": list(range(n)) * len(groups),
        "cond": sum([[str(l)] * n for l in labels], []),
        "value": np.concatenate([np.asarray(g) for g in groups]),
    })
    sphericity_note = ""
    try:
        sp = pg.sphericity(long_df, dv="value", within="cond",
                            subject="subject")
        if sp.spher:
            sphericity_note = f"Mauchly W = {sp.W:.3g}, p = {sp.pval:.3g} (met)"
            corr_needed = False
        else:
            sphericity_note = (f"Mauchly W = {sp.W:.3g}, p = {sp.pval:.3g} "
                                f"(violated -> Greenhouse-Geisser)")
            corr_needed = True
    except Exception:
        corr_needed = False
        sphericity_note = "sphericity not tested"

    try:
        aov = pg.rm_anova(data=long_df, dv="value", within="cond",
                           subject="subject", correction=True, detailed=True)
        row = aov.iloc[0]
        F = float(row["F"])
        if (corr_needed and "p_GG_corr" in aov.columns
                and pd.notna(row.get("p_GG_corr"))):
            p = float(row["p_GG_corr"])
        else:
            p = (float(row["p_unc"]) if "p_unc" in aov.columns
                  else float(stats.f_oneway(*groups)[1]))
        return F, p, sphericity_note
    except Exception:
        F, p = stats.f_oneway(*groups)
        return F, p, sphericity_note


def _cohens_d_indep(x, y) -> float:
    """Pooled-SD Cohen's d for independent-samples post-hoc pairs."""
    return _cohen_d(np.asarray(x, dtype=float), np.asarray(y, dtype=float),
                     paired=False)


def _posthoc_dispatch(groups: list, labels: list, paired: bool,
                       omnibus_test: str) -> list[dict]:
    """3-route post-hoc dispatch, in priority order: Tukey HSD
    (equal-variance ANOVA) -> Games-Howell (unequal-variance ANOVA) ->
    Holm-Bonferroni step-down otherwise. Only called when the omnibus
    test is already significant — see ``_compare_multiple_groups``."""
    pairs = list(combinations(range(len(groups)), 2))

    gh_result = None
    use_games_howell = omnibus_test == "Welch's ANOVA"
    if use_games_howell:
        try:
            import pingouin as pg
            gh_df = pd.DataFrame({
                "_y": np.concatenate([np.asarray(g) for g in groups]),
                "_grp": sum(([labels[i]] * len(groups[i])
                              for i in range(len(groups))), []),
            })
            gh_result = pg.pairwise_gameshowell(data=gh_df, dv="_y",
                                                 between="_grp")
        except Exception:
            gh_result = None
        use_games_howell = gh_result is not None

    use_tukey = (omnibus_test == "One-way ANOVA" and not paired
                  and not use_games_howell and hasattr(stats, "tukey_hsd"))

    rows: list[dict] = []

    if use_games_howell:
        gh_lookup = {}
        for _, row in gh_result.iterrows():
            gh_lookup[(str(row["A"]), str(row["B"]))] = float(row["pval"])
        for i, j in pairs:
            a_lbl, b_lbl = labels[i], labels[j]
            key = ((a_lbl, b_lbl) if (a_lbl, b_lbl) in gh_lookup
                    else (b_lbl, a_lbl))
            adj_p = gh_lookup.get(key, float("nan"))
            d = _cohens_d_indep(groups[i], groups[j])
            ci_lo, ci_hi = _bootstrap_ci_general(
                lambda x, y: abs(_cohens_d_indep(x, y)), groups[i], groups[j])
            rows.append({
                "group_a": a_lbl, "group_b": b_lbl,
                "test": "Games-Howell",
                "statistic": round(float(d), 6),
                "p_value": adj_p if not np.isnan(adj_p) else 1.0,
                "p_adjusted": adj_p if not np.isnan(adj_p) else 1.0,
                "posthoc_method": "Games-Howell",
                "effect_size": round(abs(d), 6), "effect_name": "Cohen's d",
                "ci_low": round(ci_lo, 6) if np.isfinite(ci_lo) else None,
                "ci_high": round(ci_hi, 6) if np.isfinite(ci_hi) else None,
                "significant": (not np.isnan(adj_p)) and adj_p < ALPHA,
            })
        return rows

    if use_tukey:
        tukey_res = stats.tukey_hsd(*groups)
        try:
            tukey_ci = tukey_res.confidence_interval(confidence_level=0.95)
        except Exception:
            tukey_ci = None
        for i, j in pairs:
            _vt, pair_p = _equal_variance(groups[i], groups[j], True)
            pair_homo = bool(pair_p > ALPHA) if np.isfinite(pair_p) else False
            ph_test, _ph_stat, ph_p, ph_eff, ph_metric = _run_test(
                pd.Series(groups[i]), pd.Series(groups[j]), paired, True,
                pair_homo)
            adj_p = float(tukey_res.pvalue[i, j])
            if tukey_ci is not None:
                ci_lo = round(float(tukey_ci.low[i, j]), 6)
                ci_hi = round(float(tukey_ci.high[i, j]), 6)
            else:
                ci_lo, ci_hi = None, None
            rows.append({
                "group_a": labels[i], "group_b": labels[j],
                "test": "Tukey HSD",
                "statistic": round(float(tukey_res.statistic[i, j]), 6),
                "p_value": round(float(ph_p), 6),
                "p_adjusted": float(adj_p),
                "posthoc_method": "Tukey HSD",
                "effect_size": (round(float(ph_eff), 6)
                                 if ph_eff is not None
                                 and not np.isnan(ph_eff) else None),
                "effect_name": ph_metric,
                "ci_low": ci_lo, "ci_high": ci_hi,
                "significant": adj_p < ALPHA,
            })
        return rows

    # Holm-Bonferroni fallback: Kruskal-Wallis (via pairwise Mann-Whitney)
    # or a paired design (via pairwise paired t-test/Wilcoxon), whichever
    # _run_test's own 2-group dispatch selects for each pair.
    raw = []
    for i, j in pairs:
        _vt, pair_p = _equal_variance(groups[i], groups[j], False)
        pair_homo = bool(pair_p > ALPHA) if np.isfinite(pair_p) else False
        ph_test, _ph_stat, ph_p, ph_eff, ph_metric = _run_test(
            pd.Series(groups[i]), pd.Series(groups[j]), paired, False,
            pair_homo)
        raw.append((i, j, ph_test, ph_p, ph_eff, ph_metric))
    adj_ps = holm_adjust([r[3] for r in raw])
    for (i, j, ph_test, ph_p, ph_eff, ph_metric), adj_p in zip(raw, adj_ps):
        rows.append({
            "group_a": labels[i], "group_b": labels[j],
            "test": ph_test,
            "p_value": round(float(ph_p), 6),
            "p_adjusted": float(adj_p),
            "posthoc_method": "Holm-Bonferroni",
            "effect_size": (round(float(ph_eff), 6)
                             if ph_eff is not None
                             and not np.isnan(ph_eff) else None),
            "effect_name": ph_metric,
            "ci_low": None, "ci_high": None,
            "significant": adj_p < ALPHA,
        })
    return rows


def _compare_multiple_groups(groups: list, labels: list, paired: bool,
                              both_normal: bool, homo: bool) -> tuple:
    """Omnibus test + post-hoc for one factor with 3+ levels.

    Returns ``(test_name, statistic, p_value, effect_value, effect_metric,
    (ci_low, ci_high), posthoc_rows)``.
    """
    sphericity_note = ""
    if paired:
        if both_normal:
            stat, p, sphericity_note = _rm_anova_with_sphericity(groups, labels)
            test = ("Repeated-measures ANOVA (GG-corrected)"
                     if "Greenhouse-Geisser" in sphericity_note
                     else "Repeated-measures ANOVA")
        else:
            stat, p = stats.friedmanchisquare(*groups)
            test = "Friedman"
    else:
        if both_normal and homo:
            stat, p = stats.f_oneway(*groups)
            test = "One-way ANOVA"
        elif both_normal and not homo:
            stat, p = _welch_anova(groups)
            test = "Welch's ANOVA"
        else:
            stat, p = stats.kruskal(*groups)
            test = "Kruskal-Wallis"

    is_parametric = "ANOVA" in test
    if is_parametric:
        effect_value = _eta_from_groups(*groups)
        effect_metric = "η²"
        ci = _bootstrap_ci_general(_eta_from_groups, *groups, clip=(0.0, 1.0))
    else:
        n_total = sum(len(g) for g in groups)
        k = len(groups)
        effect_value = (float((stat - k + 1) / (n_total - k))
                          if n_total > k else 0.0)
        effect_metric = "ε²"
        if test == "Kruskal-Wallis":
            ci = _bootstrap_ci_general(_eps_from_kruskal, *groups,
                                        min_n_per_group=5, clip=(0.0, 1.0))
        else:
            # Friedman is paired: resampling with replacement would break
            # the within-subject structure, so the CI stays NaN.
            ci = (float("nan"), float("nan"))

    posthoc_rows: list[dict] = []
    if p < ALPHA:
        posthoc_rows = _posthoc_dispatch(groups, labels, paired, test)

    return test, float(stat), float(p), effect_value, effect_metric, ci, posthoc_rows


def _blank_row_n(labels: list, test_label: str, samples: list = None, *,
                  normality_test="n/a", p_norm_worst=np.nan,
                  variance_test="n/a", p_variance=np.nan) -> dict:
    """N-group analogue of ``_blank_row``: descriptives per level even
    when the contrast is skipped.

    Normality is summarised as the worst (minimum) p-value across the N
    groups rather than one column per group — same "worst cell" convention
    amacrine-motion-detection's ``rm_anova.py`` uses for its own multi-cell omnibus, since a
    fixed per-group column layout doesn't fit a factor whose level count
    varies by dataset. Positioned *before* the test columns, like
    ``_blank_row``.
    """
    samples = samples or [None] * len(labels)
    descs = [describe_group(s if s is not None else []) for s in samples]
    means = {lbl: d["mean"] for lbl, d in zip(labels, descs)}
    valid_means = {k: v for k, v in means.items() if np.isfinite(v)}
    if len(valid_means) == len(labels):
        ordered = sorted(valid_means.items(), key=lambda kv: -kv[1])
        direction = " > ".join(str(k) for k, _ in ordered)
    else:
        direction = "n/a"

    row = {
        "groups": " | ".join(str(l) for l in labels),
        "direction": direction,
        "normality test": normality_test,
        "normality p (worst group)": (round(float(p_norm_worst), 4)
                                       if np.isfinite(p_norm_worst) else np.nan),
        "normality significance (worst group)": (
            _stars(p_norm_worst) if np.isfinite(p_norm_worst) else "n/a"),
        "homoscedasticity test": variance_test,
        "homoscedasticity p": (round(float(p_variance), 4)
                                if np.isfinite(p_variance) else np.nan),
        "homoscedasticity significance": (
            _stars(p_variance) if np.isfinite(p_variance) else "n/a"),
        "test": test_label,
        "p_value": np.nan,
        "significance": "ns",
        "effect size metric": "n/a",
        "effect size value": np.nan,
    }
    for lbl, d in zip(labels, descs):
        row[f"n ({lbl})"] = d["n"]
        row[f"mean ({lbl})"] = (round(d["mean"], 3)
                                  if np.isfinite(d["mean"]) else np.nan)
        row[f"SD ({lbl})"] = (round(d["sd"], 3)
                                if np.isfinite(d["sd"]) else np.nan)
    return row


def compare_n_groups(
    dataframes: dict,
    comparison_by: dict,
    variables: list,
    folder: str,
    paired: dict,
    subject: dict | None = None,
    table_prefix: str = "Group comparisons",
    min_n: int = 3,
    max_n_shapiro: int = 5000,
    verbose: bool = True,
) -> dict:
    """Compare 3+ groups across numeric variables; sibling of
    ``compare_two_groups`` for the same registry keys, activated
    automatically by ``run_two_group_pipeline`` when a dataset's
    ``comparison_by`` column has more than 2 levels.

    Writes one CSV omnibus table (one row per variable) and, when at
    least one variable's omnibus test was significant, one CSV
    post-hoc table (one row per variable x pairwise comparison) per
    dataset.

    Returns
    -------
    dict
        ``{dataset name: {"omnibus": {variable: row}, "posthoc": [row, ...]}}``.
    """
    os.makedirs(folder, exist_ok=True)
    subject = subject or {}
    results_all: dict[str, dict] = {}

    for name, df in dataframes.items():
        grp_col = comparison_by.get(name)
        is_paired = paired.get(name, False)
        subject_col = subject.get(name)

        if grp_col is None:
            if verbose:
                print(f"No comparison column defined for '{name}'. Skipping.")
            continue

        labels = list(df[grp_col].dropna().unique())
        if len(labels) < 3:
            if verbose:
                print(f"'{name}' has {len(labels)} level(s) in '{grp_col}'; "
                      f"compare_n_groups needs >= 3 (use compare_two_groups "
                      f"for exactly 2).")
            continue

        res_dict: dict[str, dict] = {}
        posthoc_all: list[dict] = []

        if verbose:
            print(f"\nProcessing '{name}': {labels} "
                  f"({'paired' if is_paired else 'independent'})")

        for var in variables:
            if var not in df.columns:
                continue

            samples, problem = _extract_samples_n(
                df, grp_col, var, labels, is_paired, subject_col, name)

            if problem is None and any(len(s) < min_n for s in samples):
                problem = "Insufficient data"

            if problem is not None:
                res_dict[var] = _blank_row_n(labels, problem, samples)
                continue

            norm_ps = []
            norm_test_names = []
            for s in samples:
                try:
                    if len(s) < 3:
                        norm_ps.append(1.0)
                        norm_test_names.append("n/a (n<3)")
                    elif len(s) <= max_n_shapiro:
                        norm_ps.append(float(stats.shapiro(s)[1]))
                        norm_test_names.append("Shapiro-Wilk")
                    else:
                        norm_ps.append(float(stats.normaltest(s)[1]))
                        norm_test_names.append("D'Agostino K^2")
                except Exception:
                    norm_ps.append(0.0)
                    norm_test_names.append("n/a (failed)")
            both_normal = all(p > ALPHA for p in norm_ps)
            worst_idx = int(np.argmin(norm_ps))
            p_norm_worst = norm_ps[worst_idx]
            normality_test_name = norm_test_names[worst_idx]

            if is_paired:
                homo = True
                variance_test_name = "n/a (paired design)"
                p_variance = np.nan
            else:
                valid = [s for s in samples if len(s) >= 2]
                if len(valid) < 2:
                    homo = True
                    variance_test_name = "n/a (insufficient groups)"
                    p_variance = np.nan
                else:
                    try:
                        _lev_stat, p_lev = stats.levene(*valid, center="median")
                        homo = bool(p_lev > ALPHA)
                        variance_test_name = "Levene"
                        p_variance = p_lev
                    except Exception:
                        homo = True
                        variance_test_name = "Levene"
                        p_variance = np.nan

            (test_name, _stat, p_value, effect_value, effect_metric, ci,
             posthoc_rows) = _compare_multiple_groups(
                samples, labels, is_paired, both_normal, homo)

            row = _blank_row_n(labels, test_name, samples,
                                normality_test=normality_test_name,
                                p_norm_worst=p_norm_worst,
                                variance_test=variance_test_name,
                                p_variance=p_variance)
            row.update({
                "p_value": round(float(p_value), 6),
                "significance": _stars(p_value),
                "effect size metric": effect_metric,
                "effect size value": (
                    round(float(effect_value), 3)
                    if effect_value is not None and not np.isnan(effect_value)
                    else np.nan),
                "effect 95% CI": (f"[{ci[0]:.2f}, {ci[1]:.2f}]"
                                   if np.isfinite(ci[0]) else "n/a"),
                "n_posthoc": len(posthoc_rows),
                "posthoc_method": (posthoc_rows[0]["posthoc_method"]
                                     if posthoc_rows else ""),
            })
            res_dict[var] = row

            for ph in posthoc_rows:
                posthoc_all.append({"variable": var, **ph})

            if verbose:
                print(f"  {var}: {test_name}, p = {p_value:.4g} "
                      f"{row['significance']}, {effect_metric} = "
                      f"{row['effect size value']}, "
                      f"{len(posthoc_rows)} post-hoc pair(s)")

        # BH-FDR across every variable tested in this table (same
        # family-of-comparisons convention as compare_two_groups).
        variables_in_table = list(res_dict.keys())
        q_values = bh_fdr([res_dict[v]["p_value"] for v in variables_in_table])
        for var, q in zip(variables_in_table, q_values):
            res_dict[var]["p_value_fdr_bh"] = (
                round(float(q), 6) if np.isfinite(q) else np.nan)
            res_dict[var]["significance_fdr"] = _stars(q) if np.isfinite(q) else "n/a"

        results_all[name] = {"omnibus": res_dict, "posthoc": posthoc_all}

        result_df = pd.DataFrame(
            [{"Variable": var, **res} for var, res in res_dict.items()])
        csv_path = os.path.join(folder, f"{table_prefix} - {name}.csv")
        save_csv(result_df, csv_path)
        if verbose:
            print(f"Omnibus results saved for '{name}' -> {csv_path}")

        if posthoc_all:
            posthoc_df = pd.DataFrame(posthoc_all)
            ph_path = os.path.join(
                folder, f"{table_prefix} - {name} - posthoc.csv")
            save_csv(posthoc_df, ph_path)
            if verbose:
                print(f"Post-hoc results saved for '{name}' -> {ph_path}")

    return results_all


# ============================================================================= #
# SECTION 5. PIPELINE DRIVER
# ============================================================================= #


def run_two_group_pipeline(datasets: dict, folder: str,
                           table_prefix: str = "Group comparisons",
                           with_intervals: bool = True,
                           verbose: bool = True) -> dict:
    """Run the 1-factor contrast for every configured dataset, dispatching
    automatically on how many levels each dataset's comparison column
    has: exactly 2 -> ``compare_two_groups`` (amacrine-motion-detection's original path,
    unchanged); 3 or more -> ``compare_n_groups`` (new here). Name kept
    from amacrine-motion-detection for continuity even though it now covers both — the
    dispatch is what changed, not the calling convention.

    Datasets whose comparison column holds fewer than 2 levels are
    skipped with a message.

    Parameters
    ----------
    datasets : dict
        ``{name: {location, group_column, paired, subject_column, ...}}``.
        The comparison factor is ``group_column[0]`` -- the same column
        used to stratify imputation and outlier screening.
    folder : str
        Directory receiving the CSV table(s) per dataset.
    table_prefix : str
        Leading text of the output file name.
    with_intervals : bool
        Compute bootstrap confidence intervals (2-group path only; the
        3+-group path always computes its omnibus effect-size CI).
    verbose : bool
        Print progress.

    Returns
    -------
    dict
        ``{dataset name: result}`` for the datasets that ran — a
        ``{variable: row}`` dict for a 2-group dataset, or a
        ``{"omnibus": {...}, "posthoc": [...]}`` dict for a 3+-group one.
    """
    os.makedirs(folder, exist_ok=True)
    all_results: dict[str, dict] = {}
    # Tracks every dataset actually processed, including ones made entirely
    # of repeated_families columns (which never touch compare_two_groups
    # / compare_n_groups and so never gain an all_results entry) -- counting
    # only len(all_results) would silently under-report such a dataset as
    # "0 dataset(s) analysed" even though its RM-ANOVA tables were written
    # correctly.
    processed = set()

    for name, cfg in datasets.items():
        # Un-split parent datasets (e.g. a genotype-mixed source that only
        # exists to feed 02_derive_datasets.py) omit `paired` -- there is
        # no single within/between-genotype design until they're split.
        # Skipped here rather than defaulting to group_column[0] as their
        # comparison factor, which would silently contrast a blended,
        # not-yet-split pool (e.g. genotype ignoring the treatment groups
        # mixed together within it).
        if "paired" not in cfg:
            if verbose:
                print(f"Skipping '{name}': no 'paired' key -- this looks "
                      f"like an un-split source dataset for "
                      f"02_derive_datasets.py, not one to contrast directly.")
            continue

        df, numeric_cols, _ = load_dataset(name, cfg)
        # group_column also names the comparison factor -- no dataset in
        # this project's registries has ever had the imputation/outlier-
        # screening stratum differ from the tested factor, so the first
        # (and typically only) entry doubles as the comparison column.
        comparison_col = cfg["group_column"][0]

        # `levels`, where declared, restricts the run to exactly those
        # conditions and their order, so a design measured at more than
        # two timepoints can still be analysed one pairwise contrast at a
        # time via the 2-group path instead of the full N-group one.
        levels = cfg.get("levels")
        if levels:
            df = df[df[comparison_col].isin(levels)].copy()
            df[comparison_col] = pd.Categorical(
                df[comparison_col], categories=levels, ordered=True)
            df = df.sort_values(comparison_col, kind="stable")
            df[comparison_col] = df[comparison_col].astype(str)
            if verbose:
                print(f"Restricted '{name}' to {list(levels)!r}")

        n_levels = df[comparison_col].nunique()

        # Identifier columns are never a variable to test. `load_dataset`
        # only routes a column to `numeric_cols` when it parses as numbers
        # AND its name doesn't contain "id" — a numeric subject/sample
        # column named e.g. "Subject" or "Mouse" slips through that filter,
        # and comparing the subject column against itself corrupts the
        # paired-pivot in `_extract_samples`/`_extract_samples_n` (two
        # same-named columns selected at once). Excluded explicitly here
        # by the configured `subject_column`, on top of the existing "id"
        # substring filter.
        subject_col = cfg.get("subject_column")

        # Columns claimed by a `repeated_families` entry (see rm_anova.py)
        # are levels of one within-subject factor, not independent
        # variables — they go through the 2-way RM-ANOVA dispatch below
        # instead of the plain per-variable path, so they're excluded here
        # to avoid testing the same level twice under two different
        # engines. Deferred import: rm_anova imports the private test
        # dispatch from this module, so importing it at module level here
        # would be circular.
        from .rm_anova import repeated_family_columns, run_repeated_families
        claimed = repeated_family_columns(cfg, df.columns)
        test_vars = [v for v in numeric_cols if v != subject_col and v not in claimed]

        # A dataset declared purely as a repeated_families view of another
        # dataset's data (see config.resolve_preprocessed_csv_path) skips
        # the plain per-variable path entirely: its non-family columns
        # (e.g. the aggregate ERG components alongside the per-intensity
        # curves) are the SAME columns already tested by the parent, full-
        # timepoint dataset, and would otherwise be silently re-tested here
        # under a `levels`-truncated, lower-powered 2-group/N-group view.
        if cfg.get("repeated_families_only"):
            test_vars = []

        if test_vars:
            if n_levels == 2:
                all_results.update(compare_two_groups(
                    dataframes={name: df},
                    comparison_by={name: comparison_col},
                    variables=test_vars,
                    folder=folder,
                    paired={name: cfg.get("paired", False)},
                    subject={name: subject_col},
                    table_prefix=table_prefix,
                    with_intervals=with_intervals,
                    verbose=verbose,
                ))
                processed.add(name)
            elif n_levels >= 3:
                all_results.update(compare_n_groups(
                    dataframes={name: df},
                    comparison_by={name: comparison_col},
                    variables=test_vars,
                    folder=folder,
                    paired={name: cfg.get("paired", False)},
                    subject={name: subject_col},
                    table_prefix=table_prefix,
                    verbose=verbose,
                ))
                processed.add(name)
            elif verbose:
                print(f"Skipping '{name}': {n_levels} level(s) in "
                      f"'{comparison_col}'; need at least 2.\n")

        # repeated_families runs on cfg's own Condition column
        # (group_column[0]) regardless of how many levels the LEFTOVER
        # non-family variables' comparison_col ended up with above --
        # rm_anova.py enforces its own 2-level requirement internally.
        if run_repeated_families(df, cfg, folder=folder, verbose=verbose):
            processed.add(name)
            # Ensures the dataset is visible in the returned dict too (not
            # just this function's own print) even when it had no plain
            # variables at all -- 02_contrasts.py's own final tally reads
            # off this return value, and would otherwise silently drop a
            # repeated-families-only dataset the same way.
            all_results.setdefault(name, {})

    if verbose:
        print(f"\nFinished. {len(processed)} dataset(s) analysed.")
    return all_results
