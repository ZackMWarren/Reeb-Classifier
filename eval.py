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
        
def binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
) -> dict:
    """
    Compute binary classification metrics.
 
    Args:
        y_true:  Ground truth binary labels.
        y_pred:  Predicted binary labels.
        y_score: Predicted probabilities for the positive class (for AUC).
 
    Returns:
        dict with keys: accuracy, f1, precision, recall, auc (nan if unavailable).
    """
    m = {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "auc":       float("nan"),
    }
    if y_score is not None and len(np.unique(y_true)) > 1:
        try:
            m["auc"] = float(roc_auc_score(y_true, y_score))
        except Exception:
            pass
    return m
 
 
def make_kfold(k_folds: int = 5) -> KFold:
    """Return a KFold splitter with standard settings."""
    return KFold(n_splits=k_folds, shuffle=True, random_state=42)
 
 
def print_fold_header(fold: int, k_folds: int) -> None:
    """Print a consistent fold header."""
    print(f"  Fold {fold + 1}/{k_folds}")
 
 
def summarize_binary_folds(
    fold_results: dict[str, list[dict]],
    class_order: list[str],
    label: str = "",
) -> None:
    """
    Print a per-class summary table of mean metrics over all folds.
 
    Args:
        fold_results: dict[class_name → list of per-fold metric dicts].
        class_order:  List of class names for row ordering.
        label:        Optional label printed in the header.
    """
    header = f"  Summary — {label}" if label else "  Summary"
    print(f"\n{'='*58}")
    print(header)
    print(f"{'='*58}")
    print(f"  {'Class':<22}  {'Acc':>7}  {'F1':>7}  {'AUC':>7}  {'Prec':>7}  {'Rec':>7}")
    print(f"  {'-'*54}")
 
    for cls in class_order:
        folds = fold_results.get(cls, [])
        if not folds:
            print(f"  {cls:<22}  {'N/A':>7}  {'N/A':>7}  {'N/A':>7}  {'N/A':>7}  {'N/A':>7}")
            continue
 
        def _mean(key):
            vals = [f[key] for f in folds if not np.isnan(f.get(key, float("nan")))]
            return float(np.mean(vals)) if vals else float("nan")
 
        acc  = _mean("accuracy")
        f1   = _mean("f1")
        auc  = _mean("auc")
        prec = _mean("precision")
        rec  = _mean("recall")
 
        def _fmt(v):
            return f"{v:7.4f}" if not np.isnan(v) else "    N/A"
 
        print(f"  {cls:<22}  {_fmt(acc)}  {_fmt(f1)}  {_fmt(auc)}  {_fmt(prec)}  {_fmt(rec)}")
 
    print(f"{'='*58}\n")
 

