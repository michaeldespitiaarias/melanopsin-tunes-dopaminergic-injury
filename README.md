# Analysis code for: Melanopsin tunes the retinal response to dopaminergic injury induced by 6-OHDA

Analysis code accompanying **"Melanopsin tunes the retinal response to
dopaminergic injury induced by 6-OHDA"**. It extends the statistical framework developed for
[amacrine-motion-detection](../amacrine-motion-detection/) — "Dopaminergic amacrine cells
modulate retinal movement detection" — from two-group contrasts to
multi-group, genotype-stratified designs.

This repository is vendored from amacrine-motion-detection's analysis code, not imported and
not a shared dependency, so each paper's code stays fully reproducible on
its own.

Every output is a plain CSV — raw numbers, no styling, no narrative
document, no figures. This is a statistical-analysis repository, not a
reporting tool.

## What's new relative to amacrine-motion-detection

Everything in [amacrine-motion-detection](../amacrine-motion-detection/) (preprocessing safety nets,
two-group contrasts, repeated-measures level families, BH-FDR) is unchanged
here, plus:

- **One factor, 3+ levels** (`inference.py`, `compare_n_groups`): One-way/
  Welch's ANOVA/Kruskal-Wallis (independent) or RM-ANOVA/Friedman (paired)
  omnibus, chosen from the data; post-hoc dispatch Tukey HSD → Games-Howell
  → Holm-Bonferroni step-down otherwise (`multicomparison.py` gains
  `holm_adjust` alongside amacrine-motion-detection's `bh_fdr`).
- **Genotype split + ratio datasets** (`ratio.py`, new): every comparison
  in this paper is single-factor, run separately within each genotype,
  rather than a genotype × group interaction model. `ratio.py` prepares
  the two kinds of derived dataset that come from that choice — see
  [Design](#design) below.

There is deliberately no multi-factor (N-way/mixed) ANOVA engine here —
none of this paper's designs call for one; see [Design](#design).

## Design

Every dataset in this paper reduces to one of two shapes:

1. **Within-genotype, one factor.** A dataset is split by genotype
   (`ratio.split_by_genotype`) and each genotype's split is analysed on
   its own — `inference.py` dispatching to a 2-level or 3+-level contrast
   however many levels that genotype's factor actually has (e.g. one
   genotype alone: Baseline / Timepoint-1 / Timepoint-2).
2. **Between-genotype ratio.** For designs where the interesting question
   is *how much* a genotype changed relative to its own reference
   condition, `ratio.compute_ratio_dataset` normalizes the treated arm
   against the reference arm per genotype (`preprocessing.transform_variables(
   method="ratio")` — paired, per-subject Post/Pre × 100 when a design has
   one; unpaired, against the reference group's mean, when it doesn't —
   e.g. a between-subjects comparison where the reference and treated arms
   are different animals, not the same animal's two timepoints) and
   concatenates the genotypes back together into one
   dataset whose only remaining factor IS genotype — a plain 2-group
   unpaired contrast.

Neither shape needs a genotype × group interaction term modelled directly;
splitting first and comparing the *effect size* (via the ratio) between
genotypes answers the same question without a multi-factor ANOVA engine.

## Repeated-measures level families

Ported from amacrine-motion-detection unchanged (`rm_anova.py`, same `repeated_families`
registry block, same engine dispatch, same interaction-gated post-hoc — see
[amacrine-motion-detection's README](../amacrine-motion-detection/README.md#repeated-measures-level-families)
for the full design and the double-counting bug the ART implementation
went through before it was trustworthy). Wired into `inference.py`'s
pipeline driver the same way: columns claimed by a family are excluded
from the plain per-variable path, and the family's own Condition column is
always `group_column[0]`, exactly like every other comparison factor in
this repository.

**Two-level datasets** use `repeated_families` directly: a `Group` column
with exactly two levels (e.g. Pre / Post) needs no extra configuration —
the ART/parametric dispatch, significance gating and post-hoc all run as
documented upstream.

**Three-or-more-level datasets need a `source` view.** `rm_anova.py`'s
Condition factor is fixed at exactly two levels (see
`compare_repeated_levels`), so a dataset whose comparison column has
three or more levels (e.g. a longitudinal recording sampled at several
timepoints) can't declare `repeated_families` on itself without losing a
level. Truncating the dataset directly with `levels` would also cut its
*other*, non-family variables down from their legitimate 3+-level
`compare_n_groups` run. `config.py` gained
`resolve_preprocessed_csv_path` to solve this: a registry entry can
declare `"source": "<other dataset>"` to read that dataset's
already-preprocessed CSV directly, with no raw input CSV or
`01_preprocess.py` pass of its own. Each timepoint-pair view then applies
its own `levels` restriction to the shared source and carries
`"repeated_families_only": true`, which skips `run_two_group_pipeline`'s
plain per-variable path for it — the non-family variables are already
tested, at full statistical power across every timepoint, by the
un-truncated parent dataset, and would otherwise be silently re-tested
under a weaker two-level view.

**Ratio datasets are not repeated-measures families.**
`ratio.compute_ratio_dataset` keeps one ratio column per original level, but
the ratio transform collapses each subject's Pre/Post pair into a single
number *before* concatenation — there is no within-subject Condition left
in a ratio dataset, only the between-subjects Genotype factor. Each level
column is correctly left on the plain per-variable path, tested as an
ordinary two-group Genotype contrast like any other variable.

## Layout

```
src/expda/
    config.py            paths and the dataset registry
    preprocessing.py       stage 1: cleaning, aggregation, screening, ratio normalisation
    effect_sizes.py         effect-size estimators and their intervals (ported, unchanged)
    multicomparison.py       Benjamini-Hochberg FDR + Holm-Bonferroni
    inference.py             stage 2: 1-factor contrasts, 2+ levels
    rm_anova.py              stage 2b: repeated-measures level families (ported, unchanged)
    ratio.py                 stage 1b: genotype split + ratio dataset prep
    reporting.py             plain-CSV writer shared by every stage
scripts/
    01_preprocess.py         run stage 1 over every dataset
    02_derive_datasets.py     build the registry's derived_datasets (split/ratio)
    02_contrasts.py           run stage 2 over every dataset
registry.example.json         template: a 2-level, a 3+-level, and a
                              genotype-split + ratio example
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10 or newer.

## Configuration

Same two environment variables as amacrine-motion-detection:

```bash
export EXPDA_DATA=/path/to/data-root
export EXPDA_REGISTRY=/path/to/registry.json   # optional
```

Every amacrine-motion-detection registry key still works unchanged, including the
`group_column`/`comparison_by` merge: a dataset's comparison factor is
`group_column[0]`, there is no separate `comparison_by` key. A dataset
that only exists as a source for derived datasets (never analysed
directly, e.g. a genotype-mixed parent) omits `paired` entirely —
`01_preprocess.py` doesn't require it, and `02_contrasts.py` skips any
dataset without a `paired` key rather than contrasting a not-yet-split
pool.

One key is new here: a `datasets` entry may set `"source": "<name>"` to
read that other dataset's already-preprocessed CSV directly instead of
having a raw input CSV of its own — used to give one longitudinal
recording several `levels`-restricted, repeated-measures-family views
(one per timepoint pair) without re-preprocessing or duplicating the
underlying data; see
[Repeated-measures level families](#repeated-measures-level-families)
above. Pair it with `"repeated_families_only": true` when the view exists
purely for its `repeated_families` and its non-family variables are
already covered by the un-truncated source dataset. This is a different
`source` from the one in the `derived_datasets` block below (that one
names a *raw* dataset to build a *new* CSV from; this one names an
already-*preprocessed* dataset to read as-is).

New top-level registry block, read by `02_derive_datasets.py`:

| Key (per entry) | Meaning |
|---|---|
| `kind` | `"split"` or `"ratio"` |
| `source` | name of the dataset this one is built from (its stage-1 cleaned CSV) |
| `genotype_column` | column carrying genotype (or any stratifying factor) |
| `genotype_value` | (`split` only) the value to filter down to |
| `group_column`, `reference_group`, `target_group` | (`ratio` only) the within-genotype factor and its two levels being contrasted |
| `genotype_values` | (`ratio` only) list of genotypes to build the ratio for and concatenate |
| `subject_column`, `paired` | (`ratio` only) passed straight through to `preprocessing.transform_variables(method="ratio")` |

Each derived name still needs its own ordinary entry in `datasets` too
(`group_column`, `subject_column`, `paired`) — `02_derive_datasets.py`
only materializes the CSV; `01_preprocess.py`/`02_contrasts.py` run on it
same as any other dataset. See `registry.example.json` for a worked,
fully synthetic example.

## Usage

```bash
# Stage 1 — clean, screen and write the diagnostic reports (unchanged from
# amacrine-motion-detection)
python scripts/01_preprocess.py --datasets <Dataset>

# Stage 1b — only for datasets with a genotype split and/or ratio step
# configured under derived_datasets
python scripts/02_derive_datasets.py

# Stage 1 again — the derived datasets need their own cleaning pass
python scripts/01_preprocess.py --datasets <Dataset>_WT <Dataset>_KO <Dataset>_Ratio

# Stage 2 — the contrasts themselves
python scripts/02_contrasts.py --datasets <Dataset>_WT <Dataset>_KO <Dataset>_Ratio
```

For a concrete, runnable version of this sequence against synthetic data,
see [`samples/README.md`](samples/README.md#reproduce).

## Pipeline

Stage 1 is identical to amacrine-motion-detection's — see
[its README](../amacrine-motion-detection/README.md#pipeline) for the missing-
value and outlier safety gates.

```
<input_dir>/<Dataset>.csv                          (source, e.g. all genotypes pooled)
    ↓  01_preprocess.py                             clean, screen -> (1) Preprocessed/<Dataset>_no_outliers.csv
    ↓  02_derive_datasets.py                         split by genotype and/or compute ratio
<input_dir>/<Dataset>_<Genotype>.csv, <Dataset>_Ratio.csv
    ↓  01_preprocess.py                             clean, screen each derived dataset
    ↓  02_contrasts.py                               compare_two_groups / compare_n_groups /
    │                                                 rm_anova.run_repeated_families
<results_dir>/<Dataset>/(2) Statistical inference/…csv
```

### Test selection — one factor, 3+ levels

| Design | Normality | Variance | Omnibus | Effect size |
|---|---|---|---|---|
| independent | pass | equal | One-way ANOVA | η² |
| independent | pass | unequal | Welch's ANOVA | η² |
| independent | fail | — | Kruskal-Wallis | ε² |
| paired | pass | — | RM-ANOVA (+ Greenhouse-Geisser if sphericity fails) | η² |
| paired | fail | — | Friedman | ε² (bootstrap CI stays NaN) |

Post-hoc only runs when the omnibus test is significant, priority order:
**Tukey HSD** (One-way ANOVA) → **Games-Howell** (Welch's ANOVA) →
**Holm-Bonferroni** otherwise.

## Output

2 levels: one `{table_prefix} - {dataset}.csv` table (test, p-value,
effect size, n/mean/SD per group, direction, assumption-check columns
before the test they gated, BH-FDR q-value as `significance_fdr` stars —
exactly amacrine-motion-detection's schema; default `table_prefix` is `"Group comparisons"`).

3+ levels: the same table plus a `{table_prefix} - {dataset} - posthoc.csv`,
only for variables whose omnibus test was significant.

Repeated-measures level families: `Repeated-measures omnibus - {family}.csv`
(always) and `Repeated-measures posthoc - {family}.csv` (only when the
Condition x Level interaction is itself significant) — see
[Repeated-measures level families](#repeated-measures-level-families) above.

Benjamini-Hochberg FDR is applied across every variable tested in one
dataset's table (one dataset = one family), exactly as in amacrine-motion-detection.

## Citation

See [`CITATION.cff`](CITATION.cff). Cite the Zenodo DOI of the specific
release used for this paper's results, not this repository's latest
state.

## License

[PolyForm Noncommercial 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0) —
see [`LICENSE`](LICENSE). Free for any noncommercial purpose (academic
research, teaching, peer review, personal study). Commercial use requires
a separate license from the author — contact esspitia@gmail.com.
