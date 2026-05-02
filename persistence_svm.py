"""
Persistence barcode SVM for single-cell graph classification.

Same pipeline as persistence_mlp.py but replaces the MLP with a
OneVsRest SVM (RBF kernel) from scikit-learn.

Pipeline:
  1. Load pre-computed barcodes from h0_finite.csv + h1_essential.csv.
  2. Vectorize: 9 statistics for H0 death values + 9 for H1 birth values = 18 features.
  3. Train a OneVsRest SVM with 5-fold cross-validation.
"""

import glob
import json
import os
import csv
from enum import IntEnum

import numpy as np
from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# ── Classes ───────────────────────────────────────────────────────────────────

class TopologyClass(IntEnum):
    CLUSTERS          = 0
    SINGLE_TRAJECTORY = 1
    MULTI_BRANCHING   = 2
    CYCLIC            = 3
    SURFACE           = 4
    ARCHETYPAL        = 5

NUM_CLASSES = len(TopologyClass)
CLASS_ORDER = [c.name for c in TopologyClass]

RAW_LABEL_MAP: dict[str, str | None] = {
    "clusters":          "CLUSTERS",
    "blob":              None,
    "simple_traj":       "SINGLE_TRAJECTORY",
    "bifurcation":       None,
    "multi_branch":      None,
    "complex_tree":      None,
    "cyclic":            None,
    "surface":           None,
    "archetypal":        "ARCHETYPAL",
    "outlier_dominated": None,
    "batch_effect":      None,
}


# ── Barcode Loading & Vectorization ───────────────────────────────────────────

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
    """9 summary statistics for one barcode dimension."""
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


FEAT_DIM = 18  # 9 H0 stats + 9 H1 stats


def barcode_to_features(h0_deaths: np.ndarray, h1_births: np.ndarray) -> np.ndarray:
    return np.concatenate([_persistence_stats(h0_deaths), _persistence_stats(h1_births)])


# ── Data Discovery ────────────────────────────────────────────────────────────

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

        mapped = {RAW_LABEL_MAP[tag] for tag in raw_labels[name] if RAW_LABEL_MAP.get(tag) is not None}
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


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(Y_true: np.ndarray, Y_pred: np.ndarray) -> tuple[float, list[float]]:
    """
    Returns:
        exact_acc:     Fraction of samples where ALL 6 predictions match.
        per_class_acc: Per-class accuracy list aligned to CLASS_ORDER.
    """
    exact_acc = float((Y_pred == Y_true).all(axis=1).mean())
    per_class_acc = (Y_pred == Y_true).mean(axis=0).tolist()
    return exact_acc, per_class_acc


# ── Cross-validation ──────────────────────────────────────────────────────────

def run_cross_validation(
    X: np.ndarray,
    Y: np.ndarray,
    k_folds: int = 5,
    C: float = 1.0,
    kernel: str = "rbf",
    gamma: str = "scale",
):
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f"\n{'='*50}")
        print(f"Fold {fold+1} / {k_folds}")
        print(f"{'='*50}")

        X_train, Y_train = X[train_idx], Y[train_idx]
        X_test, Y_test = X[test_idx], Y[test_idx]

        # Pipeline: StandardScaler + OneVsRest SVM
        # Scaler is fit on training data only — no leakage.
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", OneVsRestClassifier(SVC(C=C, kernel=kernel, gamma=gamma))),
        ])

        clf.fit(X_train, Y_train)
        Y_pred = clf.predict(X_test)

        exact_acc, per_cls = evaluate(Y_test, Y_pred)
        fold_results.append({"fold": fold + 1, "exact_acc": exact_acc, "per_class": per_cls})

        per_cls_str = "  ".join(
            f"{CLASS_ORDER[i][:4]}:{per_cls[i]:.2f}" for i in range(NUM_CLASSES)
        )
        print(f"  Exact Acc: {exact_acc:.4f} | {per_cls_str}")

    print(f"\n{'='*50}")
    print("Cross-validation summary")
    print(f"{'='*50}")
    all_exact = [r["exact_acc"] for r in fold_results]
    print(f"Exact Acc:  mean={sum(all_exact)/len(all_exact):.4f}  "
          f"min={min(all_exact):.4f}  max={max(all_exact):.4f}")
    for i, cls in enumerate(CLASS_ORDER):
        cls_accs = [r["per_class"][i] for r in fold_results]
        print(f"  {cls[:4]}: mean={sum(cls_accs)/len(cls_accs):.4f}")

    return fold_results


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pathlib

    # ── Configuration ─────────────────────────────────────────────────────────
    METHOD      = 'dreeb'        # 'dreeb' | 'mapper' | 'paga'

    DATA_DIR    = pathlib.Path(__file__).parent / "data"
    LABELS_JSON = pathlib.Path(__file__).parent / "phate_gallery_labels_2026-05-02/labels.json"
    K           = "k25"         # "k25" | "k50" | "k50_ks100"
    C           = 1.0           # SVM regularization strength
    KERNEL      = "rbf"         # 'rbf' | 'linear' | 'poly'
    GAMMA       = "scale"       # 'scale' | 'auto' | float
    # ──────────────────────────────────────────────────────────────────────────

    VALID_METHODS = ('dreeb', 'mapper', 'paga')
    if METHOD not in VALID_METHODS:
        raise ValueError(f"METHOD must be one of {VALID_METHODS}, got '{METHOD}'")

    samples = discover_samples(str(DATA_DIR), str(LABELS_JSON), k=K, method=METHOD)
    print(f"\nUsable samples: {len(samples)}  |  method: {METHOD}")
    print(f"Feature vector: {FEAT_DIM}d  (9 H0-death stats + 9 H1-birth stats)")
    print(f"SVM: kernel={KERNEL}  C={C}  gamma={GAMMA}\n")

    if not samples:
        raise RuntimeError("No usable samples found. Check DATA_DIR and LABELS_JSON paths.")

    print("Loading barcodes and building feature matrix...")
    X, Y = build_feature_matrix(samples, k=K, method=METHOD)
    print(f"X: {X.shape}  Y: {Y.shape}\n")

    run_cross_validation(X, Y, k_folds=5, C=C, kernel=KERNEL, gamma=GAMMA)
