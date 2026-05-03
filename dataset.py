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

# Taken from dReeb/src/dreeb/persistence.py
def compute_graph_persistence(num_nodes: int, edges: list, edge_lengths: list) -> dict:
    """
    Compute H0 persistent homology via Union-Find on an edge filtration.

    All vertices inserted at filtration 0. Edges processed in non-decreasing
    order of edge_lengths. Each component merge records (birth=0, death=w).

    Returns dict with keys "h0", "h0_essential", "h1".
    """
    if len(edges) != len(edge_lengths):
        raise ValueError("edges and edge_lengths must have the same length.")

    parent = np.arange(num_nodes, dtype=int)
    size   = np.ones(num_nodes, dtype=int)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    h0, h1 = [], []
    order = np.argsort(np.asarray(edge_lengths, dtype=float), kind="stable")

    for idx in order:
        u, v = edges[int(idx)]
        w    = float(edge_lengths[int(idx)])
        u, v = int(u), int(v)
        if u == v:
            h1.append((w, np.inf))
            continue
        ru, rv = find(u), find(v)
        if ru != rv:
            if size[ru] < size[rv]:
                ru, rv = rv, ru
            parent[rv]  = ru
            size[ru]   += size[rv]
            h0.append((0.0, w))
        else:
            h1.append((w, np.inf))

    roots        = {find(i) for i in range(num_nodes)}
    h0_essential = np.array([(0.0, np.inf) for _ in roots], dtype=float)

    return {
        "h0":           np.array(h0,  dtype=float).reshape(-1, 2),
        "h0_essential": h0_essential.reshape(-1, 2),
        "h1":           np.array(h1,  dtype=float).reshape(-1, 2),
    }


def load_graph_persistence(method_dir: str) -> dict:
    """Read nodes.csv + edges.csv and return persistence dict."""
    with open(os.path.join(method_dir, "nodes.csv")) as f:
        num_nodes = sum(1 for _ in csv.DictReader(f))

    edges, lengths = [], []
    with open(os.path.join(method_dir, "edges.csv")) as f:
        for row in csv.DictReader(f):
            edges.append((int(row["source"]), int(row["target"])))
            lengths.append(float(row["length"]) if row["length"] else 0.0)

    lengths = np.array(lengths, dtype=float)
    max_len = lengths.max()
    if max_len > 1e-8:
        lengths = lengths / max_len

    return compute_graph_persistence(num_nodes, edges, lengths.tolist())


def _to_birth_persistence(diag: np.ndarray) -> np.ndarray:
    """Convert [(birth, death)] → [(birth, persistence)] with persistence=max(d-b, 0)."""
    if diag.size == 0:
        return diag
    out       = np.zeros_like(diag)
    out[:, 0] = diag[:, 0]
    out[:, 1] = np.maximum(diag[:, 1] - diag[:, 0], 0)
    return out


def compute_global_bounds(diags_list: list[dict]) -> tuple:
    """
    Compute shared (birth, persistence) bounds across all samples so that
    persistence images are directly comparable — mirrors notebook's
    compute_global_bounds().
    """
    vals = []
    for d in diags_list:
        arr = d.get("h0", np.zeros((0, 2)))
        if arr.size > 0:
            vals.append(_to_birth_persistence(arr))
    if not vals:
        return ((0.0, 1.0), (0.0, 1.0))
    M          = np.vstack(vals)
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
    Convert a (birth, death) persistence diagram to a flattened persistence
    image — direct port of the notebook's make_persistence_image().
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

    bp_norm       = bp.copy()
    bp_norm[:, 0] = (bp[:, 0] - bmin) / (bmax - bmin)
    bp_norm[:, 1] = (bp[:, 1] - pmin) / (pmax - pmin)

    B_bins, P_bins = grid_bins
    b_edges        = np.linspace(0, 1, B_bins + 1)
    p_edges        = np.linspace(0, 1, P_bins + 1)
    b_centers      = (b_edges[:-1] + b_edges[1:]) / 2
    p_centers      = (p_edges[:-1] + p_edges[1:]) / 2
    B_grid, P_grid = np.meshgrid(b_centers, p_centers, indexing="ij")

    pi = np.zeros((B_bins, P_bins), dtype=np.float64)
    for b_pt, p_pt in bp_norm:
        weight  = p_pt
        dist_sq = (B_grid - b_pt) ** 2 + (P_grid - p_pt) ** 2
        pi     += weight * np.exp(-dist_sq / (2 * sigma ** 2))

    return pi.flatten().astype(np.float32)


# ── Sample discovery ──────────────────────────────────────────────────────────

def discover_samples(
    data_dir: str,
    labels_file: str,
    k: str = "k25",
    method: str = "dreeb",
) -> list[tuple[str, list[float]]]:
    """
    Find all labelled SCD-* directories that have the required method
    sub-folder, map raw JSON labels to CLASS_ORDER binary vectors, and
    drop samples with no valid class label.

    Returns:
        List of (sample_dir, label_vec) tuples where:
          sample_dir : path to the SCD-* folder (NOT the method sub-dir)
          label_vec  : list of floats [n_classes], one per CLASS_ORDER entry
    """
    all_dirs = sorted(glob.glob(os.path.join(data_dir, "SCD-*")))
    complete = [
        d for d in all_dirs
        if os.path.isdir(os.path.join(d, k, method))
    ]
    if not complete:
        raise FileNotFoundError(
            f"No complete SCD-* samples found in '{data_dir}' "
            f"for method='{method}' / k='{k}'"
        )

    with open(labels_file) as f:
        raw_labels: dict[str, list[str]] = json.load(f)["labels"]

    samples                            = []
    dropped_unlabelled, dropped_no_cls = [], []

    for d in complete:
        name = os.path.basename(d)
        if name not in raw_labels:
            dropped_unlabelled.append(name)
            continue

        mapped = set()
        for tag in raw_labels[name]:
            cls = RAW_LABEL_MAP.get(tag)
            if cls is not None:
                mapped.add(cls)

        if not mapped:
            dropped_no_cls.append(f"{name}: {raw_labels[name]}")
            continue

        vec = [1.0 if c in mapped else 0.0 for c in CLASS_ORDER]
        samples.append((d, vec))

    print(f"Kept:                              {len(samples)} samples")
    print(f"Dropped (no entry in labels JSON): {len(dropped_unlabelled)}")
    print(f"Dropped (no valid class):          {len(dropped_no_cls)}")
    if dropped_no_cls:
        print(f"  → {dropped_no_cls}")

    return samples