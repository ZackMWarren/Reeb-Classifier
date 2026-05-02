"""
Persistence barcode SVM — one binary classifier per TopologyClass.

For each class C the model predicts: does this sample belong to C?
Pipeline per class:
  1. StandardScaler (fit on training fold only).
  2. SVC with probability=True for AUC scoring.
  3. Evaluate with 5-fold CV and report Accuracy / F1 / AUC / Precision / Recall.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from dataset import CLASS_ORDER, FEAT_DIM
from eval import binary_metrics, make_kfold, print_fold_header, summarize_binary_folds


# ── Main entry point ──────────────────────────────────────────────────────────

def run(
    X: np.ndarray,
    Y: np.ndarray,
    k_folds: int = 5,
    C: float = 1.0,
    kernel: str = "rbf",
    gamma: str = "scale",
) -> dict[str, list[dict]]:
    """
    Run 5-fold binary CV for each TopologyClass.

    Returns:
        fold_results[class_name] = list of per-fold metric dicts
    """
    kf = make_kfold(k_folds)
    fold_results: dict[str, list[dict]] = {cls: [] for cls in CLASS_ORDER}

    for cls_idx, cls_name in enumerate(CLASS_ORDER):
        y_binary = Y[:, cls_idx].astype(int)
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

            clf = Pipeline([
                ("scaler", StandardScaler()),
                ("svm",    SVC(C=C, kernel=kernel, gamma=gamma, probability=True)),
            ])
            clf.fit(X_train, y_train)

            y_pred  = clf.predict(X_test)
            y_score = clf.predict_proba(X_test)[:, 1]

            m = binary_metrics(y_test, y_pred, y_score=y_score)
            fold_results[cls_name].append({"fold": fold + 1, **m})
            print(f"  Acc: {m['accuracy']:.4f}  F1: {m['f1']:.4f}  AUC: {m.get('auc', float('nan')):.4f}")

    summarize_binary_folds(fold_results, CLASS_ORDER, label=f"Persistence SVM  kernel={kernel}  C={C}")
    return fold_results


# ── Standalone ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pathlib
    from dataset import build_feature_matrix, discover_samples

    METHOD      = "dreeb"
    DATA_DIR    = pathlib.Path(__file__).parent.parent / "data"
    LABELS_JSON = sorted((pathlib.Path(__file__).parent.parent).glob("phate_gallery_labels*/labels.json"))[-1]
    K           = "k25"
    C           = 1.0
    KERNEL      = "rbf"
    GAMMA       = "scale"

    samples = discover_samples(str(DATA_DIR), str(LABELS_JSON), k=K, method=METHOD)
    print(f"\nUsable samples: {len(samples)}  |  method: {METHOD}")
    print(f"Feature vector: {FEAT_DIM}d  (9 H0-death stats + 9 H1-birth stats)")
    print(f"SVM: kernel={KERNEL}  C={C}  gamma={GAMMA}\n")

    X, Y = build_feature_matrix(samples, k=K, method=METHOD)
    print(f"X: {X.shape}  Y: {Y.shape}\n")

    run(X, Y, C=C, kernel=KERNEL, gamma=GAMMA)
