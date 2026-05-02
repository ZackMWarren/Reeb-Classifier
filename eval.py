import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import KFold


def make_kfold(n_splits: int = 5) -> KFold:
    return KFold(n_splits=n_splits, shuffle=True, random_state=42)


def print_fold_header(fold: int, k_folds: int) -> None:
    print(f"\n{'='*50}")
    print(f"Fold {fold + 1} / {k_folds}")
    print(f"{'='*50}")


def binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
) -> dict:
    metrics = {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
    }
    if y_score is not None and len(np.unique(y_true)) > 1:
        metrics["auc"] = float(roc_auc_score(y_true, y_score))
    else:
        metrics["auc"] = float("nan") #if a class has a single class (all true or all false)
    return metrics


def summarize_binary_folds(
    fold_results: dict[str, list[dict]],
    class_order: list[str],
    label: str = "",
) -> None:
    header = f"Cross-validation summary{f'  [{label}]' if label else ''}"
    print(f"\n{'='*60}")
    print(header)
    print(f"{'='*60}")
    print(f"  {'Class':<22} {'Acc':>7}  {'F1':>7}  {'AUC':>7}  {'Prec':>7}  {'Rec':>7}")
    print(f"  {'-'*58}")
    for cls in class_order:
        folds = fold_results.get(cls, [])
        if not folds:
            continue
        acc  = float(np.mean([f["accuracy"]  for f in folds]))
        f1   = float(np.mean([f["f1"]        for f in folds]))
        prec = float(np.mean([f["precision"] for f in folds]))
        rec  = float(np.mean([f["recall"]    for f in folds]))
        auc_vals = [f["auc"] for f in folds if not np.isnan(f["auc"])]
        auc_str  = f"{np.mean(auc_vals):7.4f}" if auc_vals else "    N/A"
        print(f"  {cls:<22} {acc:7.4f}  {f1:7.4f}  {auc_str}  {prec:7.4f}  {rec:7.4f}")
