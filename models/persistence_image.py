"""
Persistence Image SVM classifier for scShapeBench.

Follows the notebook approach:
  1. Persistence image IS the feature vector (built by dataset.build_feature_matrix)
  2. StandardScale features
  3. k-fold CV with one binary SVM per TopologyClass

Each class is treated as an independent binary classification problem:
  "Does this sample belong to CLUSTERS?" etc.

Usage (standalone):
    python persistence_svm.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from dataset import CLASS_ORDER, NUM_CLASSES
from eval import binary_metrics, make_kfold, print_fold_header, summarize_binary_folds


def run(
    X: np.ndarray,
    Y: np.ndarray,
    k_folds: int = 5,
) -> dict[str, list[dict]]:
    """
    Run k-fold binary CV for each TopologyClass using an RBF-SVM.

    Args:
        X:       Feature matrix [n_samples, feat_dim] — raw, not yet scaled.
        Y:       Binary label matrix [n_samples, NUM_CLASSES].
        k_folds: Number of CV folds.

    Returns:
        fold_results[class_name] = list of per-fold metric dicts, each with
        keys: accuracy, f1, auc, precision, recall.
    """
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kf           = make_kfold(k_folds)
    fold_results = {cls: [] for cls in CLASS_ORDER}

    for cls_idx, cls_name in enumerate(CLASS_ORDER):
        y_binary = Y[:, cls_idx]
        pos, neg = int(y_binary.sum()), int((1 - y_binary).sum())
        print(f"\n{'─'*50}")
        print(f"  {cls_name}  (pos={pos}, neg={neg})")
        print(f"{'─'*50}")

        for fold, (train_idx, test_idx) in enumerate(kf.split(X_scaled)):
            print_fold_header(fold, k_folds)

            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train         = y_binary[train_idx]
            y_test          = y_binary[test_idx]

            # Skip fold if training split has only one class
            if len(np.unique(y_train)) < 2:
                print(f"  Skipping fold {fold + 1}: training split has only one class.")
                continue

            clf = SVC(kernel="rbf", class_weight="balanced",
                      gamma="scale", probability=True)
            clf.fit(X_train, y_train)

            y_pred  = clf.predict(X_test)
            y_proba = clf.predict_proba(X_test)[:, 1]

            m = binary_metrics(y_test, y_pred, y_score=y_proba)
            fold_results[cls_name].append({"fold": fold + 1, **m})
            print(f"  Acc: {m['accuracy']:.4f}  F1: {m['f1']:.4f}  "
                  f"AUC: {m.get('auc', float('nan')):.4f}")

    summarize_binary_folds(fold_results, CLASS_ORDER, label="SVM (RBF)")
    return fold_results


# ── Standalone ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pathlib
    from dataset import build_feature_matrix, discover_samples

    METHOD      = "dreeb"
    DATA_DIR    = pathlib.Path(__file__).parent.parent / "data"
    LABELS_JSON = sorted(
        (pathlib.Path(__file__).parent.parent).glob("phate_gallery_labels*/labels.json")
    )[-1]
    K = "k25"

    samples = discover_samples(str(DATA_DIR), str(LABELS_JSON), k=K, method=METHOD)
    print(f"\nUsable samples: {len(samples)}  |  method: {METHOD}\n")

    print("Building feature matrix...")
    X, Y = build_feature_matrix(samples, k=K, method=METHOD)
    print(f"X: {X.shape}  Y: {Y.shape}\n")

    run(X, Y)