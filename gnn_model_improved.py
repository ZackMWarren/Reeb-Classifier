"""
Single-input GNN for single-cell graph classification.

Each sample is represented by one graph built from a topology method:
  - dreeb  : DREEB algorithm (diffusion eigenfunction, ~141 nodes)
  - mapper : Kepler-Mapper TDA graph (~88 nodes)
  - paga   : Scanpy PAGA cluster graph (~20 nodes)

The METHOD variable in __main__ controls which graph is used.
Labels are multi-label (a sample can belong to multiple topology classes).
Samples whose labels map to none of the 6 TopologyClass values are dropped.
"""

import glob
import json
import os
import csv
from enum import IntEnum

import torch
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.transforms import AddLaplacianEigenvectorPE

from sklearn.model_selection import KFold

# Number of LPE eigenvectors. Must be < smallest graph size.
# paga ~20 nodes → keep this well below 20. 8 is a safe default.
LPE_DIM = 8
# Node feature dim = (x, y) coords + LPE_DIM eigenvectors
NODE_FEATURES = 2 + LPE_DIM


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

# Map raw JSON label strings → canonical TopologyClass names.
# Tags that map to None are silently ignored.
# Tags not in this dict at all are also ignored.
RAW_LABEL_MAP: dict[str, str | None] = {
    "clusters":          "CLUSTERS",
    "blob":              None,
    "simple_traj":       "SINGLE_TRAJECTORY",
    "bifurcation":       None,
    "multi_branch":      None,
    "complex_tree":       None,
    "cyclic":            None,
    "surface":           None,
    "archetypal":        "ARCHETYPAL",
    "outlier_dominated": None,
    "batch_effect":      None,
}


# ── Data Loading ──────────────────────────────────────────────────────────────

def _load_graph(method_dir: str) -> Data:
    """
    Read nodes.csv + edges.csv from a method directory into a PyG Data.
    Node features = (x, y) coords concatenated with LPE_DIM eigenvectors.
    """
    with open(os.path.join(method_dir, "nodes.csv")) as f:
        nodes = [(float(r["x"]), float(r["y"])) for r in csv.DictReader(f)]
    coords = torch.tensor(nodes, dtype=torch.float)  # [N, 2]

    src, dst, lengths = [], [], []
    with open(os.path.join(method_dir, "edges.csv")) as f:
        for row in csv.DictReader(f):
            src.append(int(row["source"]))
            dst.append(int(row["target"]))
            lengths.append(float(row["length"]) if row["length"] else 0.0)

    edge_index  = torch.tensor([src + dst, dst + src], dtype=torch.long)
    edge_weight = torch.tensor(lengths + lengths, dtype=torch.float)
    if edge_weight.numel() > 0:
        edge_weight = edge_weight / edge_weight.max().clamp(min=1e-8)

    # Compute LPE with adaptive k — AddLaplacianEigenvectorPE requires k < n_nodes.
    # Pad with zeros when the graph is smaller than LPE_DIM so feature size stays
    # consistent at NODE_FEATURES = 2 + LPE_DIM.
    n_nodes = coords.size(0)
    k_eff   = min(LPE_DIM, n_nodes - 1)
    if k_eff > 0:
        lpe = AddLaplacianEigenvectorPE(k=k_eff, attr_name="pe", is_undirected=True)
        tmp = Data(x=coords, edge_index=edge_index, edge_weight=edge_weight)
        tmp = lpe(tmp)
        pe  = tmp.pe                                         # [N, k_eff]
        if k_eff < LPE_DIM:
            pe = F.pad(pe, (0, LPE_DIM - k_eff))            # [N, LPE_DIM]
    else:
        pe = torch.zeros(n_nodes, LPE_DIM)

    x = torch.cat([coords, pe], dim=1)                      # [N, NODE_FEATURES]
    return Data(x=x, edge_index=edge_index, edge_weight=edge_weight)


class GraphData(Data):
    """PyG Data object holding a single method graph."""

    def __cat_dim__(self, key: str, value, *args, **kwargs):
        if key == "edge_index":
            return 1
        return 0

    def __inc__(self, key: str, value, *args, **kwargs):
        if key == "edge_index":
            return self.x.size(0)
        return super().__inc__(key, value, *args, **kwargs)


def load_sample(
    sample_dir: str,
    label: list[float] | None = None,
    k: str = "k25",
    method: str = "dreeb",
) -> GraphData:
    """
    Build a GraphData from one sample directory.

    Args:
        sample_dir: Path to the sample folder (e.g. data/SCD-0001).
        label:      Binary float vector of length NUM_CLASSES, or None.
        k:          k-NN run sub-directory (default "k25").
        method:     Which graph method to load.
    """
    g = _load_graph(os.path.join(sample_dir, k, method))
    d = GraphData(x=g.x, edge_index=g.edge_index, edge_weight=g.edge_weight)
    if label is not None:
        d.y = torch.tensor(label, dtype=torch.float)
    return d


class SampleDataset(Dataset):
    """In-memory dataset from a list of (sample_dir, label) pairs."""

    def __init__(
        self,
        samples: list[tuple[str, list[float]]],
        k: str = "k25",
        method: str = "dreeb",
    ):
        super().__init__()
        self.data_list = [load_sample(d, lbl, k, method) for d, lbl in samples]

    def len(self) -> int:
        return len(self.data_list)

    def get(self, idx: int) -> GraphData:
        return self.data_list[idx]


def make_loader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
) -> DataLoader:
    """Return a DataLoader for a single-method dataset."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def discover_samples(
    data_dir: str,
    labels_file: str,
    k: str = "k25",
    method: str = "dreeb",
) -> list[tuple[str, list[float]]]:
    """
    Find all SCD-* directories that have the required method sub-folder,
    map their raw JSON labels to the 6 TopologyClass binary vectors, and
    drop any sample whose labels produce an all-zero vector (i.e. none of
    the raw tags map to a known TopologyClass).

    Args:
        data_dir:    Root folder containing SCD-* sample directories.
        labels_file: Path to labels JSON (must have a top-level "labels" key).
        k:           k-NN run sub-directory to check for completeness.
        method:      Method folder that must exist for a sample to be included.

    Returns:
        List of (sample_dir_path, binary_label_vector) sorted by sample name.
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

    kept                = []
    dropped_unlabelled  = []
    dropped_no_class    = []

    for d in complete:
        name = os.path.basename(d)

        if name not in raw_labels:
            dropped_unlabelled.append(name)
            continue

        # Map raw tags → canonical TopologyClass names, ignore unknown/None
        mapped = set()
        for tag in raw_labels[name]:
            cls = RAW_LABEL_MAP.get(tag)
            if cls is not None:
                mapped.add(cls)

        if not mapped:
            # All tags were ignored (e.g. only outlier_dominated / batch_effect)
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


# ── Model ─────────────────────────────────────────────────────────────────────

class _GraphBranch(torch.nn.Module):
    """Single-layer GCN with batch norm and global mean pooling."""

    def __init__(self, in_channels: int, hidden_channels: int):
        super().__init__()
        self.conv = GCNConv(in_channels, hidden_channels)
        self.bn   = BatchNorm1d(hidden_channels)

    def forward(self, x, edge_index, edge_weight, batch):
        x = self.bn(self.conv(x, edge_index, edge_weight).relu())
        return global_mean_pool(x, batch)      # [batch_size, hidden_channels]


class SingleGraphGNN(torch.nn.Module):
    """
    Multi-label graph classifier using a single topology-derived graph.

    Args:
        method:          Which graph method is being used (stored for reference).
        in_channels:     Node feature dimension (default: 2 coords + LPE_DIM).
        hidden_channels: Width of the GCN branch.
        num_classes:     Number of output classes (one sigmoid per class).
    """

    def __init__(
        self,
        method: str,
        in_channels: int = NODE_FEATURES,
        hidden_channels: int = 64,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()
        self.method = method
        self.branch = _GraphBranch(in_channels, hidden_channels)
        self.lin1   = Linear(hidden_channels, hidden_channels)
        self.lin2   = Linear(hidden_channels, num_classes)

    def forward(self, data: GraphData) -> torch.Tensor:
        h = self.branch(data.x, data.edge_index, data.edge_weight, data.batch)
        h = self.lin1(h).relu()
        h = F.dropout(h, p=0.5, training=self.training)
        return self.lin2(h)                    # raw logits, sigmoid applied by loss


# ── Training & Evaluation ─────────────────────────────────────────────────────

def run_cross_validation(
    samples,
    k_folds: int = 5,
    k: str = "k25",
    method: str = "dreeb",
    hidden: int = 64,
    batch_size: int = 16,
    lr: float = 1e-3,
    epochs: int = 50,
    device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(samples)):
        print(f"\n{'='*50}")
        print(f"Fold {fold+1} / {k_folds}")
        print(f"{'='*50}")

        train_samples = [samples[i] for i in train_idx]
        test_samples  = [samples[i] for i in test_idx]

        train_ds = SampleDataset(train_samples, k=k, method=method)
        test_ds  = SampleDataset(test_samples,  k=k, method=method)

        train_loader = make_loader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader  = make_loader(test_ds, batch_size=batch_size, shuffle=False)

        model     = SingleGraphGNN(method=method, hidden_channels=hidden).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.BCEWithLogitsLoss()

        for epoch in range(1, epochs + 1):
            loss = train_epoch(model, train_loader, optimizer, criterion, device)
            if epoch % 10 == 0:
                exact_acc, per_cls = evaluate(model, test_loader, device)
                per_cls_str = "  ".join(
                    f"{CLASS_ORDER[i][:4]}:{per_cls[i]:.2f}" for i in range(NUM_CLASSES)
                )
                print(f"  Epoch {epoch:03d} | Loss: {loss:.4f} | Exact Acc: {exact_acc:.4f} | {per_cls_str}")

        # final result for this fold
        exact_acc, per_cls = evaluate(model, test_loader, device)
        fold_results.append({"fold": fold+1, "exact_acc": exact_acc, "per_class": per_cls})
        print(f"\n  Fold {fold+1} final → Exact Acc: {exact_acc:.4f}")

    # summary across all folds
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

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        y = batch.y.view(-1, NUM_CLASSES)
        loss = criterion(model(batch), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.detach().item() * batch.num_graphs 
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    """
    Returns:
        exact_acc:     Fraction of samples where ALL 6 predictions are correct.
        per_class_acc: Per-class accuracy list aligned to CLASS_ORDER.
    """
    model.eval()
    exact_correct = 0
    total         = 0
    class_correct = torch.zeros(NUM_CLASSES)
    class_total   = torch.zeros(NUM_CLASSES)

    for batch in loader:
        batch  = batch.to(device)
        preds  = (torch.sigmoid(model(batch)) > 0.5).float().cpu()
        labels = batch.y.view(-1, NUM_CLASSES).cpu()

        exact_correct += (preds == labels).all(dim=1).sum().item()
        total         += batch.num_graphs
        class_correct += (preds == labels).sum(dim=0)
        class_total   += batch.num_graphs

    exact_acc     = exact_correct / total
    per_class_acc = (class_correct / class_total).tolist()
    return exact_acc, per_class_acc


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pathlib

    # ── Configuration ─────────────────────────────────────────────────────────
    METHOD      = 'dreeb'        # 'dreeb' | 'mapper' | 'paga'

    DATA_DIR    = pathlib.Path(__file__).parent / "data"
    LABELS_JSON = pathlib.Path(__file__).parent / "phate_gallery_labels_2026-05-02/labels.json"
    K           = "k25"         # "k25" | "k50" | "k50_ks100"
    HIDDEN      = 64
    BATCH_SIZE  = 16
    LR          = 1e-3
    EPOCHS      = 50
    # ──────────────────────────────────────────────────────────────────────────

    VALID_METHODS = ('dreeb', 'mapper', 'paga')
    if METHOD not in VALID_METHODS:
        raise ValueError(f"METHOD must be one of {VALID_METHODS}, got '{METHOD}'")

    samples = discover_samples(str(DATA_DIR), str(LABELS_JSON), k=K, method=METHOD)
    print(f"\nUsable samples: {len(samples)}  |  method: {METHOD}\n")

    if len(samples) == 0:
        raise RuntimeError("No usable samples found. Check DATA_DIR and LABELS_JSON paths.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fold_results = run_cross_validation(
        samples,
        k_folds=5,
        k=K,
        method=METHOD,
        hidden=HIDDEN,
        batch_size=BATCH_SIZE,
        lr=LR,
        epochs=EPOCHS,
        device=device,
    )