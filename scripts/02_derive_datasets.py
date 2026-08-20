#!/usr/bin/env python3
"""Stage 1b - build the genotype-split and ratio datasets.

Reads the registry's ``derived_datasets`` block and, for each entry,
writes a new CSV into ``input_dir`` — a genotype split (``kind: "split"``)
or a between-genotype ratio dataset (``kind: "ratio"``, see
``expda.ratio.compute_ratio_dataset``). Both read from their *source*
dataset's stage-1 cleaned CSV (``(1) Preprocessed/``), so run
``01_preprocess.py`` on the source datasets first.

Every derived dataset also needs its own entry in the registry's
``datasets`` block (comparison_by, subject_column, paired, ...) — this
script only materializes the CSV; ``01_preprocess.py`` /
``02_contrasts.py`` still need to run on the derived name afterward, same
as for any other dataset.

Usage
-----
    python scripts/02_derive_datasets.py
    python scripts/02_derive_datasets.py --datasets ERG_Ratio_30 ERG_Ratio_90
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from expda import config, preprocessing, ratio  # noqa: E402


def _numeric_cols_for_ratio(df, cfg: dict) -> list[str]:
    """Measurement columns a ratio dataset should be built over: numeric,
    minus identifiers/group columns, matching ``inference.py``'s own
    "not a variable to test" rule so the ratio step and the eventual
    contrast agree on what counts as a measurement."""
    exclude = {cfg.get("subject_column"), cfg.get("group_column")}
    numeric = df.select_dtypes(include="number").columns.tolist()
    return [c for c in numeric
            if c not in exclude and "id" not in c.lower()
            and not c.lower().endswith("_outlier")]


def build_one(name: str, entry: dict, layout: dict, verbose: bool) -> None:
    source = entry["source"]
    source_csv = config.preprocessed_csv_path(source, layout)
    if not source_csv.exists():
        raise FileNotFoundError(
            f"'{name}' needs '{source}' preprocessed first — "
            f"run 01_preprocess.py --datasets {source}"
        )
    df = preprocessing.load_dataset(source, {"location": str(source_csv)})[0]

    if entry["kind"] == "split":
        out_df = ratio.split_by_genotype(
            df, entry["genotype_column"], entry["genotype_value"])
    elif entry["kind"] == "ratio":
        numeric_cols = _numeric_cols_for_ratio(df, entry)
        out_df = ratio.compute_ratio_dataset(
            df, numeric_cols,
            group_column=entry["group_column"],
            reference_group=entry["reference_group"],
            target_group=entry["target_group"],
            genotype_column=entry["genotype_column"],
            genotype_values=entry["genotype_values"],
            subject_column=entry.get("subject_column"),
            paired=entry.get("paired", False),
            verbose=verbose,
        )
    else:
        raise ValueError(f"Unknown derived_datasets kind {entry['kind']!r} for '{name}'.")

    out_path = config.input_path(name, layout)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    if verbose:
        print(f"'{name}' ({entry['kind']}) -> {out_path}  ({len(out_df)} rows)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="subset of derived_datasets to build (default: all)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    try:
        registry = config.load_registry()
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 2

    layout = registry["layout"]
    derived = registry.get("derived_datasets", {})
    if not derived:
        print("No 'derived_datasets' section in the registry.", file=sys.stderr)
        return 1

    names = args.datasets or list(derived.keys())
    missing = [n for n in names if n not in derived]
    if missing:
        print(f"Not in derived_datasets: {missing}", file=sys.stderr)
        return 2

    for name in names:
        build_one(name, derived[name], layout, verbose=not args.quiet)

    print(f"\nBuilt {len(names)} derived dataset(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
