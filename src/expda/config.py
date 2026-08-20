"""Filesystem layout and dataset registry.

Nothing about a particular study is hard-coded. Both the location of the data
and the description of each analysis unit come from outside the package:

* ``EXPDA_DATA`` — root directory holding the data and the output tree.
* ``EXPDA_REGISTRY`` — path to a JSON file describing the datasets.
  Defaults to ``registry.json`` beside the data root.

See ``registry.example.json`` for the expected shape.

Expected tree under the data root::

    <DATA_ROOT>/
        <input_dir>/    <Dataset>.csv                          input to stage 1
        <results_dir>/<Dataset>/(1) Preprocessed/<Dataset>_no_outliers.csv
                                             stage 1's output, stage 2's input
        <results_dir>/<Dataset>/(2) Statistical inference/…csv

The dataset name is the join key across every stage: the same string names
the input CSV, the results folder and the table inside it. Stage 2 and
``02_derive_datasets.py`` (for genotype-split/ratio datasets) both read the
preprocessed CSV straight from ``(1) Preprocessed/``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = Path(os.environ.get("EXPDA_DATA", PACKAGE_ROOT / "data")).expanduser()

# Directory names, overridable through the registry file.
DEFAULT_LAYOUT = {
    "input_dir": "input",
    "results_dir": "results",
    "report_folders": {
        "intermediate": "(1) Preprocessed",
        "hypothesis_contrast": "(2) Statistical inference",
    },
    "table_prefix": "Group comparisons",
}


def registry_path() -> Path:
    """Location of the JSON registry describing the datasets."""
    return Path(
        os.environ.get("EXPDA_REGISTRY", DATA_ROOT / "registry.json")
    ).expanduser()


def load_registry() -> dict:
    """Read the registry, merged over the default layout.

    Returns
    -------
    dict
        ``{"layout": {...}, "datasets": {name: {...}}, "derived_datasets":
        {name: {...}}}``. ``derived_datasets`` describes the genotype-split
        and ratio datasets ``02_derive_datasets.py`` builds (see
        ``expda.ratio``) — empty when the registry doesn't declare any.

    Raises
    ------
    FileNotFoundError
        If no registry file is present. There is no built-in dataset list.
    """
    path = registry_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"No dataset registry at {path}. Set EXPDA_REGISTRY, or copy "
            f"registry.example.json and adapt it."
        )
    raw = json.loads(path.read_text(encoding="utf-8"))

    layout = {**DEFAULT_LAYOUT, **raw.get("layout", {})}
    layout["report_folders"] = {
        **DEFAULT_LAYOUT["report_folders"],
        **raw.get("layout", {}).get("report_folders", {}),
    }
    return {
        "layout": layout,
        "datasets": raw.get("datasets", {}),
        "derived_datasets": raw.get("derived_datasets", {}),
    }


def input_path(name: str, layout: dict) -> Path:
    """Path to the raw CSV that stage 1 consumes."""
    return DATA_ROOT / layout["input_dir"] / f"{name}.csv"


def results_path(name: str, folder_key: str, layout: dict) -> Path:
    """Path to one report subfolder of one dataset."""
    return DATA_ROOT / layout["results_dir"] / name / layout["report_folders"][folder_key]


def preprocessed_csv_path(name: str, layout: dict) -> Path:
    """Path to the one preprocessed CSV both stage 1 writes and stage 2 (and
    ``02_derive_datasets.py``) read — inside the dataset's own
    ``(1) Preprocessed`` results folder."""
    return results_path(name, "intermediate", layout) / f"{name}_no_outliers.csv"


def resolve_preprocessed_csv_path(name: str, cfg: dict, layout: dict) -> Path:
    """Like ``preprocessed_csv_path``, but honours an optional ``source`` key.

    A registry entry with a ``source`` has no raw input CSV or preprocessing
    pass of its own — it reads a *different* dataset's already-preprocessed
    CSV, restricted under its own ``levels``/``repeated_families`` (e.g. two
    timepoint-pair views of the same longitudinal recording, each with a
    2-way repeated-measures ANOVA that needs exactly two Condition levels
    out of a recording that actually has three or more). There is only ever
    one physical preprocessed copy per underlying recording, never one per
    view of it — ``01_preprocess.py`` skips any dataset with a ``source``
    rather than writing a redundant duplicate.
    """
    return preprocessed_csv_path(cfg.get("source") or name, layout)


def select(registry: dict, names: list[str] | None) -> dict:
    """Return the requested datasets, or all of them when ``names`` is None."""
    datasets = registry["datasets"]
    if not names:
        return dict(datasets)
    missing = [n for n in names if n not in datasets]
    if missing:
        raise KeyError(f"Not in the registry: {missing}")
    return {n: datasets[n] for n in names}


def describe_layout(layout: dict) -> str:
    """Summary of where the data is being read from, for logs."""
    return (
        f"DATA_ROOT = {DATA_ROOT}  (exists: {DATA_ROOT.is_dir()})\n"
        f"registry  = {registry_path()}"
    )
