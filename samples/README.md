# Samples

Synthetic demo data and its real, pre-baked pipeline output — proof the
code runs end-to-end, across every dataset shape this repo supports,
without needing the actual (unpublished) experimental dataset behind the
paper.

- `input/` — synthetic datasets, each with a deliberately planted effect
  (and, for `TwoOrMoreGroups`, planted missing values / outliers to
  exercise the preprocessing safety gates):
  - `TwoOrMoreGroups` — 1 factor, 2 levels, paired
  - `ThreePlusGroups` — 1 factor, 3 levels, independent, one group with a
    planted effect and one without
  - `GenotypeStudy` — WT/KO × Baseline/Treated, paired within subject,
    with a planted genotype × treatment interaction (WT responds ~1.8×,
    KO barely responds ~1.05×) — the source dataset `02_derive_datasets.py`
    splits into `GenotypeStudy_WT` / `GenotypeStudy_KO` (within-genotype
    Baseline-vs-Treated contrasts) and `GenotypeStudy_Ratio` (the
    Treated/Baseline × 100 ratio per subject, compared WT vs. KO).
  - `ERG_WT_Demo` — 1 within-subject Condition (Baseline/Treated) × 4
    ordered levels, with a planted Condition × Level interaction (flat at
    Level1/Level2, a large drop at Level3/Level4) — exercises
    `repeated_families`/`rm_anova.py` (ported from amacrine-motion-detection): the ART
    engine, a significant interaction, and the resulting gated per-level
    post-hoc.
- `registry.json` — the dataset registry describing all of the above,
  including the `derived_datasets` block for `GenotypeStudy` and the
  `repeated_families` block for `ERG_WT_Demo`.
- `results/` — real output of running the pipeline against `input/` (not
  hand-edited).

## Reproduce

```bash
export EXPDA_DATA="$(pwd)/samples"
export EXPDA_REGISTRY="$(pwd)/samples/registry.json"

# Datasets analysed directly
python scripts/01_preprocess.py --clean --datasets TwoOrMoreGroups ThreePlusGroups GenotypeStudy ERG_WT_Demo

# Build the genotype splits + ratio dataset from GenotypeStudy
python scripts/02_derive_datasets.py

# The derived datasets need their own cleaning pass
python scripts/01_preprocess.py --datasets GenotypeStudy_WT GenotypeStudy_KO GenotypeStudy_Ratio

# Contrasts — GenotypeStudy itself is a source only, deliberately excluded
python scripts/02_contrasts.py --datasets TwoOrMoreGroups ThreePlusGroups GenotypeStudy_WT GenotypeStudy_KO GenotypeStudy_Ratio ERG_WT_Demo
```

Every output file is a plain CSV — no plots, no HTML — matching the
project's `results.csv` / `posthoc.csv` convention (see the root
[README](../README.md)). Each design's planted effect is recovered by its
omnibus test; `GenotypeStudy_Ratio`'s WT-vs-KO contrast is the strongest
signal in the whole sample set (both genotypes respond individually, but
the *ratio* comparison is what isolates the interaction), which is the
entire point of building it as a separate dataset rather than reading the
interaction off two individually-significant within-genotype contrasts.
