"""
Persistence barcode MLP for single-cell graph classification.

Pipeline:
  1. Load pre-computed barcodes from h0_finite.csv + h1_essential.csv.
  2. Vectorize: 9 statistics for H0 death values + 9 for H1 birth values = 18 features.
  3. Train a small MLP with 5-fold cross-validation (same setup as gnn_model_improved.py).

H0 finite intervals: birth=0, death=finite  → death value = lifetime
H1 essential intervals: birth=finite, death=inf → birth value used as feature
"""

import glob
import json
import os
import csv
from enum import IntEnum

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold


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
    """
    Load pre-computed persistence barcodes.

    Returns:
        h0_deaths: finite H0 interval death values (all births are 0)
        h1_births: essential H1 interval birth values (all deaths are inf)
    """
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
        len(vals),                    # count
        total,                        # sum
        float(vals.max()),            # max
        float(vals.mean()),           # mean
        float(vals.std()),            # std
        entropy,                      # persistence entropy
        float(np.percentile(vals, 25)),
        float(np.percentile(vals, 50)),
        float(np.percentile(vals, 75)),
    ], dtype=np.float32)


# 9 H0 stats + 9 H1 stats
FEAT_DIM = 18


def barcode_to_features(h0_deaths: np.ndarray, h1_births: np.ndarray) -> np.ndarray:
    """Fixed-size feature vector from H0/H1 barcodes."""
    return np.concatenate([_persistence_stats(h0_deaths), _persistence_stats(h1_births)])


# ── Data Discovery ────────────────────────────────────────────────────────────

def discover_samples(
    data_dir: str,
    labels_file: str,
    k: str = "k25",
    method: str = "dreeb",
) -> list[tuple[str, list[float]]]:
    """
    Find all SCD-* directories that have the required method sub-folder,
    map raw JSON labels to the 6 TopologyClass binary vectors, and drop
    samples with no valid class (same logic as gnn_model_improved.py).
    """
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
    """
    Load barcodes for all samples and return feature matrix + label matrix.

    Returns:
        X: float32 array of shape [N, FEAT_DIM]
        Y: float32 array of shape [N, NUM_CLASSES]
    """
    X, Y = [], []
    for sample_dir, label in samples:
        method_dir = os.path.join(sample_dir, k, method)
        h0_deaths, h1_births = _load_barcode(method_dir)
        X.append(barcode_to_features(h0_deaths, h1_births))
        Y.append(label)
    return np.stack(X), np.array(Y, dtype=np.float32)


# ── Model ─────────────────────────────────────────────────────────────────────


class PersistenceMLP(torch.nn.Module):
    """Multi-label logistic regression (linear probing)."""

    def __init__(self, in_features=FEAT_DIM, num_classes=NUM_CLASSES):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.linear(x)
    

# ── Training & Evaluation ─────────────────────────────────────────────────────

def _make_loader(X: np.ndarray, Y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def _train_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    for X_batch, Y_batch in loader:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_batch), Y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.detach().item() * len(X_batch)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[float, list[float]]:
    """
    Returns:
        exact_acc:     Fraction of samples where ALL 6 predictions match.
        per_class_acc: Per-class accuracy list aligned to CLASS_ORDER.
    """
    model.eval()
    exact_correct = 0
    total = 0
    class_correct = torch.zeros(NUM_CLASSES)
    class_total = torch.zeros(NUM_CLASSES)

    for X_batch, Y_batch in loader:
        preds = (torch.sigmoid(model(X_batch.to(device))) > 0.5).float().cpu()
        exact_correct += (preds == Y_batch).all(dim=1).sum().item()
        total += len(X_batch)
        class_correct += (preds == Y_batch).sum(dim=0)
        class_total += len(X_batch)

    return exact_correct / total, (class_correct / class_total).tolist()

@torch.no_grad()
def evaluate_binary(model, loader, device):
    model.eval()

    class_correct = torch.zeros(NUM_CLASSES)
    class_total = torch.zeros(NUM_CLASSES)

    for X_batch, Y_batch in loader:
        logits = model(X_batch.to(device))
        probs = torch.sigmoid(logits).cpu()
        preds = (probs > 0.5).float()

        class_correct += (preds == Y_batch).sum(dim=0)
        class_total += len(X_batch)

    acc = (class_correct / class_total).tolist()

    for i, cls in enumerate(CLASS_ORDER):
        print(f"{cls}: {acc[i]:.3f}")

    return acc

def run_cross_validation(
    X: np.ndarray,
    Y: np.ndarray,
    k_folds: int = 5,
    hidden: int = 64,
    batch_size: int = 16,
    lr: float = 1e-3,
    epochs: int = 100,
    device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f"\n{'='*50}")
        print(f"Fold {fold+1} / {k_folds}")
        print(f"{'='*50}")

        X_train, Y_train = X[train_idx], Y[train_idx]
        X_test, Y_test = X[test_idx], Y[test_idx]

        # Normalize with training-set statistics to prevent data leakage
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0) + 1e-8
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std

        train_loader = _make_loader(X_train, Y_train, batch_size, shuffle=True)
        test_loader = _make_loader(X_test, Y_test, batch_size, shuffle=False)

        model = PersistenceMLP(in_features=FEAT_DIM, hidden=hidden).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = torch.nn.BCEWithLogitsLoss()

        for epoch in range(1, epochs + 1):
            loss = _train_epoch(model, train_loader, optimizer, criterion, device)
            if epoch % 20 == 0:
                exact_acc, per_cls = evaluate(model, test_loader, device)
                per_cls_str = "  ".join(
                    f"{CLASS_ORDER[i][:4]}:{per_cls[i]:.2f}" for i in range(NUM_CLASSES)
                )
                print(f"  Epoch {epoch:03d} | Loss: {loss:.4f} | Exact Acc: {exact_acc:.4f} | {per_cls_str}")

        exact_acc, per_cls = evaluate(model, test_loader, device)
        fold_results.append({"fold": fold + 1, "exact_acc": exact_acc, "per_class": per_cls})
        print(f"\n  Fold {fold+1} final → Exact Acc: {exact_acc:.4f}")

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
    METHOD      = 'mapper'        # 'dreeb' | 'mapper' | 'paga'

    DATA_DIR    = pathlib.Path(__file__).parent / "data"
    LABELS_JSON = pathlib.Path(__file__).parent / "phate_gallery_labels_2026-05-02/labels.json"
    K           = "k25"         # "k25" | "k50" | "k50_ks100"
    HIDDEN      = 64
    BATCH_SIZE  = 8
    LR          = 1e-3
    EPOCHS      = 1
    # ──────────────────────────────────────────────────────────────────────────

    VALID_METHODS = ('dreeb', 'mapper', 'paga')
    if METHOD not in VALID_METHODS:
        raise ValueError(f"METHOD must be one of {VALID_METHODS}, got '{METHOD}'")

    samples = discover_samples(str(DATA_DIR), str(LABELS_JSON), k=K, method=METHOD)
    print(f"\nUsable samples: {len(samples)}  |  method: {METHOD}")
    print(f"Feature vector: {FEAT_DIM}d  (9 H0-death stats + 9 H1-birth stats)\n")

    if not samples:
        raise RuntimeError("No usable samples found. Check DATA_DIR and LABELS_JSON paths.")

    print("Loading barcodes and building feature matrix...")
    X, Y = build_feature_matrix(samples, k=K, method=METHOD)
    print(f"X: {X.shape}  Y: {Y.shape}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    run_cross_validation(
        X, Y,
        k_folds=5,
        hidden=HIDDEN,
        batch_size=BATCH_SIZE,
        lr=LR,
        epochs=EPOCHS,
        device=device,
    )
