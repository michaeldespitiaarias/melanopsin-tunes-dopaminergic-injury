"""Stage 1b - genotype split and between-genotype ratio datasets.

This paper's designs are all single-factor (see ``inference.py``), run
separately within each genotype rather than modelled as a genotype x group
interaction. Two data-preparation steps feed that:

- ``split_by_genotype`` — filters a base dataset down to one genotype, so
  the within-genotype Group/treatment/timepoint contrast in
  ``inference.py`` runs on a clean single-genotype sample.
- ``compute_ratio_dataset`` — for each genotype, normalizes the treated
  arm against its own reference arm (paired, per-subject Post/Pre when the
  design has one; unpaired, against the reference group's mean when it
  doesn't — see ``preprocessing.transform_variables(method="ratio")``,
  which already implements both), then re-attaches the genotype label and
  concatenates across genotypes into one dataset whose comparison factor
  IS genotype. This is how a within-subject or within-arm effect becomes
  comparable between genotypes without a multi-factor interaction
  model — see the module docstring in ``inference.py``.

Both steps write a plain CSV into the data root's ``input_dir``, ready to
be picked up by ``01_preprocess.py`` / ``02_contrasts.py`` like any other
dataset — they need their own entry in the registry's ``datasets`` block
(comparison_by, subject_column, paired, ...) in addition to the
``derived_datasets`` entry that describes how to build them.
"""

from __future__ import annotations

import pandas as pd

from .preprocessing import transform_variables


def split_by_genotype(df: pd.DataFrame, genotype_column: str,
                       genotype_value: str) -> pd.DataFrame:
    """Return the rows of *df* matching one genotype.

    A thin, explicit filter — kept as its own function (rather than an
    inline ``df[df[col] == value]`` at each call site) so every genotype
    split in the registry goes through the same, single code path.
    """
    if genotype_column not in df.columns:
        raise ValueError(f"Genotype column '{genotype_column}' not found.")
    return df[df[genotype_column] == genotype_value].copy()


def compute_ratio_dataset(
    df: pd.DataFrame,
    numeric_cols: list,
    group_column: str,
    reference_group: str,
    target_group: str,
    genotype_column: str,
    genotype_values: list,
    subject_column: str | None = None,
    paired: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """Build one ratio dataset spanning every genotype in *genotype_values*.

    For each genotype: restrict *df* to the two groups being contrasted
    (*reference_group*, *target_group*), then call
    ``preprocessing.transform_variables(method="ratio")`` — paired
    (per-subject *target*/`reference` x 100, matched by *subject_column*)
    or unpaired (*target* / mean(*reference*) x 100) per *paired*. The
    per-genotype ratio rows are concatenated with the genotype label
    re-attached (the ratio transform itself drops every column outside
    the id/group/numeric ones it needs), producing one dataset whose rows
    are the treated arm's ratio values and whose only remaining group
    factor is genotype.

    Parameters
    ----------
    df : DataFrame
        Source data — every genotype, both groups, ideally already
        cleaned (stage-1 no_outliers), so a per-subject/per-group outlier
        in the raw values doesn't propagate into the ratio.
    numeric_cols : list
        Measurement columns to convert to ratios.
    group_column, reference_group, target_group : str
        The within-genotype factor and its two levels being contrasted
        (e.g. "Group", "Baseline", "Treated").
    genotype_column, genotype_values : str, list
        Which column carries genotype and which values to build the
        ratio for (typically ``["WT", "KO"]``).
    subject_column : str, optional
        Required when *paired* is True.
    paired : bool
        Per-subject ratio (needs *subject_column*) vs. ratio against the
        reference group's mean.
    verbose : bool
        Print progress.

    Returns
    -------
    DataFrame
        One row per treated-arm subject per genotype, columns
        ``[genotype_column, group_column?, subject_column?] + numeric_cols``.
    """
    if paired and not subject_column:
        raise ValueError("subject_column is required when paired=True.")

    pieces = []
    for genotype in genotype_values:
        sub = split_by_genotype(df, genotype_column, genotype)
        sub = sub[sub[group_column].isin([reference_group, target_group])]

        ratio_df = transform_variables(
            sub, columns=numeric_cols, method="ratio",
            reference_group=reference_group, group_column=group_column,
            subject_column=subject_column, paired=paired, verbose=verbose,
        )
        # The ratio transform's own output already keeps only the
        # subject/group columns it needed plus the ratio values — the
        # genotype label was never in its scope, so it is re-attached
        # here rather than asking the transform to know about a factor
        # it has no other use for.
        ratio_df = ratio_df[ratio_df[group_column] == target_group].copy()
        ratio_df.insert(0, genotype_column, genotype)
        pieces.append(ratio_df)

    result = pd.concat(pieces, ignore_index=True)
    if verbose:
        counts = result[genotype_column].value_counts().to_dict()
        print(f"✅ Ratio dataset built: {counts}")
    return result
