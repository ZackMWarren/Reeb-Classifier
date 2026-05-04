"""
Persistence barcode MLP — one binary classifier per TopologyClass.

For each class C the model predicts: does this sample belong to C?
Pipeline per class:
  1. Normalize features (train-set statistics only — no leakage).
  2. Train a small MLP with BCEWithLogitsLoss.
  3. Evaluate with 5-fold CV and report Accuracy / F1 / AUC / Precision / Recall.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from dataset import CLASS_ORDER, FEAT_DIM, NUM_CLASSES
from eval import binary_metrics, make_kfold, print_fold_header, summarize_binary_folds


# ── Model ─────────────────────────────────────────────────────────────────────

class BinaryMLP(torch.nn.Module):
    def __init__(self, in_features: int = FEAT_DIM, hidden: int = 64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_features, hidden),
            torch.nn.BatchNorm1d(hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(hidden, hidden // 2),
            torch.nn.BatchNorm1d(hidden // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)  # [N] logits


# ── Training helpers ──────────────────────────────────────────────────────────

def _make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, drop_last: bool = False) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)


def _train_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.detach().item() * len(X_batch)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def _predict(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_logits, all_labels = [], []
    for X_batch, y_batch in loader:
        logits = model(X_batch.to(device)).cpu()
        all_logits.append(logits)
        all_labels.append(y_batch)
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy().astype(int)
    probs  = 1 / (1 + np.exp(-logits))   # sigmoid
    preds  = (probs > 0.5).astype(int)
    return labels, preds, probs


# ── Main entry point ──────────────────────────────────────────────────────────

def run(
    X: np.ndarray,
    Y: np.ndarray,
    k_folds: int = 5,
    hidden: int = 64,
    batch_size: int = 16,
    lr: float = 1e-3,
    epochs: int = 100,
    device=None,
) -> dict[str, list[dict]]:
    """
    Run 5-fold binary CV for each TopologyClass.

    Returns:
        fold_results[class_name] = list of per-fold metric dicts
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    kf = make_kfold(k_folds)
    fold_results: dict[str, list[dict]] = {cls: [] for cls in CLASS_ORDER}

    for cls_idx, cls_name in enumerate(CLASS_ORDER):
        y_binary = Y[:, cls_idx].astype(np.float32)
        pos, neg = int(y_binary.sum()), int((1 - y_binary).sum())
        print(f"\n{'─'*50}")
        print(f"  {cls_name}  (pos={pos}, neg={neg})")
        print(f"{'─'*50}")

        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            print_fold_header(fold, k_folds)

            X_train, y_train = X[train_idx], y_binary[train_idx]
            X_test,  y_test  = X[test_idx],  y_binary[test_idx]

            if len(np.unique(y_train)) < 2:
                print(f"  Skipping fold {fold + 1}: training split has only one class.")
                continue

            mean = X_train.mean(axis=0)
            std  = X_train.std(axis=0) + 1e-8
            X_train = (X_train - mean) / std
            X_test  = (X_test  - mean) / std

            train_loader = _make_loader(X_train, y_train, batch_size, shuffle=True, drop_last=True)
            test_loader  = _make_loader(X_test,  y_test,  batch_size, shuffle=False)

            model     = BinaryMLP(in_features=FEAT_DIM, hidden=hidden).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
            criterion = torch.nn.BCEWithLogitsLoss()

            for epoch in range(1, epochs + 1):
                _train_epoch(model, train_loader, optimizer, criterion, device)

            labels, preds, probs = _predict(model, test_loader, device)
            m = binary_metrics(labels, preds, y_score=probs)
            fold_results[cls_name].append({"fold": fold + 1, **m})
            print(f"  Acc: {m['accuracy']:.4f}  F1: {m['f1']:.4f}  AUC: {m.get('auc', float('nan')):.4f}")

    summarize_binary_folds(fold_results, CLASS_ORDER, label="Persistence MLP")
    return fold_results


# ── Standalone ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pathlib
    from dataset import build_feature_matrix, discover_samples

    METHOD      = "dreeb"
    DATA_DIR    = pathlib.Path(__file__).parent.parent / "data"
    LABELS_JSON = sorted((pathlib.Path(__file__).parent.parent).glob("phate_gallery_labels*/labels.json"))[-1]
    K           = "k25"

    samples = discover_samples(str(DATA_DIR), str(LABELS_JSON), k=K, method=METHOD)
    print(f"\nUsable samples: {len(samples)}  |  method: {METHOD}")
    print(f"Feature vector: {FEAT_DIM}d  (9 H0-death stats + 9 H1-birth stats)\n")

    X, Y = build_feature_matrix(samples, k=K, method=METHOD)
    print(f"X: {X.shape}  Y: {Y.shape}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    run(X, Y, device=device)
