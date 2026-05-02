import csv
import glob
import json
import os
from enum import IntEnum

import numpy as np


class TopologyClass(IntEnum):
    CLUSTERS          = 0
    SINGLE_TRAJECTORY = 1
    MULTI_BRANCHING   = 2
    # CYCLIC            = 3
    # SURFACE           = 4
    # ARCHETYPAL        = 5


NUM_CLASSES = len(TopologyClass)
CLASS_ORDER = [c.name for c in TopologyClass]

RAW_LABEL_MAP: dict[str, str | None] = {
    "clusters":          "CLUSTERS",
    "blob":              None,
    "simple_traj":       "SINGLE_TRAJECTORY",
    "bifurcation":       None,
    "multi_branch":      "MULTI_BRANCHING",
    "complex_tree":      None,
    "cyclic":            None,
    "surface":           None,
    "archetypal":        "ARCHETYPAL",
    "outlier_dominated": None,
    "batch_effect":      None,
}

VALID_METHODS = ("dreeb", "mapper", "paga")
VALID_K       = ("k25", "k50", "k50_ks100")


# ── Sample discovery ──────────────────────────────────────────────────────────

def discover_samples(
    data_dir: str,
    labels_file: str,
    k: str = "k25",
    method: str = "dreeb",
) -> list[tuple[str, list[float]]]:
    all_dirs = sorted(glob.glob(os.path.join(data_dir, "SCD-*")))
    complete = [d for d in all_dirs if os.path.isdir(os.path.join(d, k, method))]
    if not complete:
        raise FileNotFoundError(
            f"No complete SCD-* samples found in '{data_dir}' "
            f"for method='{method}' / k='{k}'"
        )

    with open(labels_file) as f:
        raw_labels: dict[str, list[str]] = json.load(f)["labels"]

    kept, dropped_unlabelled, dropped_no_class = [], [], []

    for d in complete:
        name = os.path.basename(d)
        if name not in raw_labels:
            dropped_unlabelled.append(name)
            continue

        mapped = {
            RAW_LABEL_MAP[tag]
            for tag in raw_labels[name]
            if RAW_LABEL_MAP.get(tag) is not None
        }
        if not mapped:
            dropped_no_class.append(f"{name}: {raw_labels[name]}")
            continue

        vec = [1.0 if c in mapped else 0.0 for c in CLASS_ORDER]
        kept.append((d, vec))

    print(f"Kept:                              {len(kept)} samples")
    print(f"Dropped (no entry in labels JSON): {len(dropped_unlabelled)}")
    print(f"Dropped (no valid TopologyClass):  {len(dropped_no_class)}")
    if dropped_no_class:
        print(f"  → {dropped_no_class}")

    return kept


# ── Barcode features (persistence MLP / SVM) ──────────────────────────────────

FEAT_DIM = 18  # 9 H0 stats + 9 H1 stats


def _load_barcode(method_dir: str) -> tuple[np.ndarray, np.ndarray]:
    h0_deaths = []
    with open(os.path.join(method_dir, "h0_finite.csv")) as f:
        for row in csv.DictReader(f):
            h0_deaths.append(float(row["death"]))

    h1_births = []
    with open(os.path.join(method_dir, "h1_essential.csv")) as f:
        for row in csv.DictReader(f):
            h1_births.append(float(row["birth"]))

    return np.array(h0_deaths, dtype=np.float32), np.array(h1_births, dtype=np.float32)


def _persistence_stats(vals: np.ndarray) -> np.ndarray:
    if len(vals) == 0:
        return np.zeros(9, dtype=np.float32)
    total = vals.sum()
    entropy = 0.0
    if total > 0:
        p = vals / total
        entropy = float(-np.sum(p * np.log(p + 1e-10)))
    return np.array([
        len(vals),
        total,
        float(vals.max()),
        float(vals.mean()),
        float(vals.std()),
        entropy,
        float(np.percentile(vals, 25)),
        float(np.percentile(vals, 50)),
        float(np.percentile(vals, 75)),
    ], dtype=np.float32)


def barcode_to_features(h0_deaths: np.ndarray, h1_births: np.ndarray) -> np.ndarray:
    return np.concatenate([_persistence_stats(h0_deaths), _persistence_stats(h1_births)])


def build_feature_matrix(
    samples: list[tuple[str, list[float]]],
    k: str = "k25",
    method: str = "dreeb",
) -> tuple[np.ndarray, np.ndarray]:
    X, Y = [], []
    for sample_dir, label in samples:
        method_dir = os.path.join(sample_dir, k, method)
        h0_deaths, h1_births = _load_barcode(method_dir)
        X.append(barcode_to_features(h0_deaths, h1_births))
        Y.append(label)
    return np.stack(X), np.array(Y, dtype=np.float32)
