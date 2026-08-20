"""Preprocessing and hypothesis-testing framework for tabular experiments,
extending amacrine-motion-detection's two-group analysis code to multi-group designs and to
genotype-stratified ratio comparisons.

A pipeline for repeated-measures tabular data: it cleans and screens a
dataset, reports the assumptions behind the tests that follow, and runs
one-factor contrasts — two levels or more — with effect sizes, confidence
intervals, and Benjamini-Hochberg / Holm-Bonferroni correction. Every
comparison in this paper is single-factor, run separately within each
stratum (typically genotype) rather than as a multi-factor interaction
model — see ``ratio.py`` for the genotype-split and ratio-normalization
step that prepares the strata and the between-genotype ratio datasets.

Nothing about a particular study is hard-coded. Datasets, grouping factors and
directory names are declared in an external JSON registry; see
``registry.example.json``.

Modules
-------
``config``            paths and the dataset registry
``preprocessing``      stage 1: cleaning, aggregation, screening, assumption reports
``effect_sizes``       effect-size estimators and their confidence intervals (2-group)
``multicomparison``     Benjamini-Hochberg FDR and Holm-Bonferroni corrections
``inference``           stage 2: 1-factor contrasts, 2 or more levels
``ratio``               genotype split + ratio-normalization data prep (stage 1b)
``reporting``           plain-CSV writer shared by every stage
"""

__version__ = "0.0.0"  # TODO: set to the tagged release version (e.g. v1.0.0) at Zenodo archival time

__all__ = [
    "config",
    "effect_sizes",
    "inference",
    "multicomparison",
    "preprocessing",
    "ratio",
    "reporting",
]
