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
    "archetypal":        None,
    "outlier_dominated": None,
    "batch_effect":      None,
}

VALID_METHODS = ("dreeb", "mapper", "paga")
VALID_K       = ("k25", "k50", "k50_ks100")

GRID_BINS = (32, 32)
SIGMA     = 0.05


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

# ── Persistence Functions ─────────────────────────────────────────────────────────

def build_persistence_images(
    samples: list[tuple[str, list[float]]],
    k: str = "k25",
    method: str = "dreeb",
    grid_bins: tuple = GRID_BINS,
    sigma: float = SIGMA,
) -> tuple[np.ndarray, np.ndarray]:
    method_dirs = [os.path.join(d, k, method) for d, _ in samples]

    # Load precomputed H0 diagrams directly from h0_finite.csv
    all_diags = []
    for method_dir in method_dirs:
        h0_deaths = []
        with open(os.path.join(method_dir, "h0_finite.csv")) as f:
            for row in csv.DictReader(f):
                h0_deaths.append(float(row["death"]))
        # H0 births are always 0 — reconstruct (birth, death) pairs
        arr = np.array([[0.0, d] for d in h0_deaths], dtype=float) if h0_deaths else np.zeros((0, 2))
        all_diags.append({"h0": arr})

    bounds = compute_global_bounds(all_diags)

    feat_list = []
    for diag in all_diags:
        pi_vec = make_persistence_image(
            diag.get("h0", np.zeros((0, 2))),
            grid_bins, sigma, bounds=bounds,
        )
        feat_list.append(pi_vec)

    X = np.stack(feat_list)
    Y = np.array([lbl for _, lbl in samples], dtype=np.float32)
    return X, Y

def _to_birth_persistence(diag: np.ndarray) -> np.ndarray:
    """Convert [(birth, death)] to [(birth, persistence)] with persistence=max(d-b, 0)."""
    if diag.size == 0:
        return diag
    out = np.zeros_like(diag)
    out[:, 0] = diag[:, 0]
    out[:, 1] = np.maximum(diag[:, 1] - diag[:, 0], 0)
    return out

def compute_global_bounds(diags_list: list[dict]) -> tuple:
    """
    Compute shared (birth, persistence) bounds across all samples so that
    persistence images are comparable 
    """
    vals = []
    for d in diags_list:
        arr = d.get("h0", np.zeros((0, 2)))
        if arr.size > 0:
            vals.append(_to_birth_persistence(arr))
    if not vals:
        return ((0.0, 1.0), (0.0, 1.0))
    M = np.vstack(vals)
    bmin, bmax = float(M[:, 0].min()), float(M[:, 0].max())
    pmin, pmax = 0.0, float(M[:, 1].max())
    if bmax == bmin: bmax = bmin + 1e-6
    if pmax == pmin: pmax = pmin + 1e-6
    return ((bmin, bmax), (pmin, pmax))

def make_persistence_image(
    diag_bd: np.ndarray,
    grid_bins: tuple = GRID_BINS,
    sigma: float     = SIGMA,
    bounds: tuple | None = None,
) -> np.ndarray:
    """
    Convert a (birth, death) persisstence diagram to a flattened persistence image
    """
    bp = _to_birth_persistence(diag_bd)

    if bp.size == 0:
        return np.zeros(grid_bins[0] * grid_bins[1], dtype=np.float32)

    if bounds is None:
        bmin, bmax = float(bp[:, 0].min()), float(bp[:, 0].max())
        pmin, pmax = float(bp[:, 1].min()), float(bp[:, 1].max())
    else:
        (bmin, bmax), (pmin, pmax) = bounds

    if bmax == bmin: bmax = bmin + 1e-6
    if pmax == pmin: pmax = pmin + 1e-6

    bp_norm = bp.copy()
    bp_norm[:, 0] = (bp[:, 0] - bmin) / (bmax - bmin)
    bp_norm[:, 1] = (bp[:, 1] - pmin) / (pmax - pmin)

    B_bins, P_bins = grid_bins
    b_edges = np.linspace(0, 1, B_bins + 1)
    p_edges = np.linspace(0, 1, P_bins + 1)
    b_centers = (b_edges[:-1] + b_edges[1:]) / 2
    p_centers = (p_edges[:-1] + p_edges[1:]) / 2
    B_grid, P_grid = np.meshgrid(b_centers, p_centers, indexing="ij")

    pi = np.zeros((B_bins, P_bins), dtype=np.float64)
    
    #weight by persistence and apply Gaussian kernel
    for b_pt, p_pt in bp_norm:
        weight  = p_pt
        dist_sq = (B_grid - b_pt) ** 2 + (P_grid - p_pt) ** 2
        pi += weight * np.exp(-dist_sq / (2 * sigma ** 2))

    return pi.flatten().astype(np.float32)

