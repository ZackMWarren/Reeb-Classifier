"""
Single-input GNN — one binary classifier per TopologyClass.

For each class C the model predicts: does this sample belong to C?
Each sample is represented by one graph built from a topology method:
  - dreeb  : DREEB algorithm (~141 nodes)
  - mapper : Kepler-Mapper TDA graph (~88 nodes)
  - paga   : Scanpy PAGA cluster graph (~20 nodes)

Node features = (x, y) coords + LPE_DIM Laplacian eigenvectors.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import csv
import os

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse.linalg._eigen.arpack.arpack import ArpackNoConvergence
from torch.nn import BatchNorm1d, Linear
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.transforms import AddLaplacianEigenvectorPE

from dataset import CLASS_ORDER, NUM_CLASSES
from eval import binary_metrics, make_kfold, print_fold_header, summarize_binary_folds

# Number of LPE eigenvectors. Must be < smallest graph size.
# paga ~20 nodes → keep well below 20. 8 is a safe default.
LPE_DIM       = 8
NODE_FEATURES = 2 + LPE_DIM


# ── Graph data structures ─────────────────────────────────────────────────────

class GraphData(Data):
    def __cat_dim__(self, key: str, value, *args, **kwargs):
        if key == "edge_index":
            return 1
        return 0

    def __inc__(self, key: str, value, *args, **kwargs):
        if key == "edge_index":
            return self.x.size(0)
        return super().__inc__(key, value, *args, **kwargs)


def _dense_lpe(n_nodes: int, edge_index: torch.Tensor, edge_weight: torch.Tensor, k: int) -> torch.Tensor:
    """Dense fallback for Laplacian PE when ARPACK fails to converge."""
    L = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()
    w   = edge_weight.numpy() if edge_weight.numel() > 0 else np.ones(len(src))
    for i, j, wij in zip(src, dst, w):
        L[i, j] -= wij
        L[i, i] += wij
    # make symmetric (already is, but ensure numerical symmetry)
    L = (L + L.T) / 2
    eig_vals, eig_vecs = np.linalg.eigh(L)
    # skip the zero eigenvalue (index 0), take next k
    pe = eig_vecs[:, 1:k + 1].astype(np.float32)
    if pe.shape[1] < k:
        pe = np.pad(pe, ((0, 0), (0, k - pe.shape[1])))
    return torch.from_numpy(pe)


def _load_graph(method_dir: str) -> GraphData:
    with open(os.path.join(method_dir, "nodes.csv")) as f:
        nodes = [(float(r["x"]), float(r["y"])) for r in csv.DictReader(f)]
    coords = torch.tensor(nodes, dtype=torch.float)

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

    n_nodes = coords.size(0)
    k_eff   = min(LPE_DIM, n_nodes - 1)
    if k_eff > 0:
        try:
            lpe = AddLaplacianEigenvectorPE(k=k_eff, attr_name="pe", is_undirected=True)
            tmp = lpe(Data(x=coords, edge_index=edge_index, edge_weight=edge_weight))
            pe  = tmp.pe
        except ArpackNoConvergence:
            pe = _dense_lpe(n_nodes, edge_index, edge_weight, k_eff)
        if k_eff < LPE_DIM:
            pe = F.pad(pe, (0, LPE_DIM - k_eff))
    else:
        pe = torch.zeros(n_nodes, LPE_DIM)

    x = torch.cat([coords, pe], dim=1)
    return GraphData(x=x, edge_index=edge_index, edge_weight=edge_weight)


class SampleDataset(Dataset):
    """In-memory dataset with a binary label for one TopologyClass."""

    def __init__(
        self,
        samples: list[tuple[str, list[float]]],
        cls_idx: int,
        k: str = "k25",
        method: str = "dreeb",
    ):
        super().__init__()
        self.data_list = []
        for sample_dir, label_vec in samples:
            g = _load_graph(os.path.join(sample_dir, k, method))
            g.y = torch.tensor([label_vec[cls_idx]], dtype=torch.float)
            self.data_list.append(g)

    def len(self) -> int:
        return len(self.data_list)

    def get(self, idx: int) -> GraphData:
        return self.data_list[idx]


# ── Model ─────────────────────────────────────────────────────────────────────

class _GraphBranch(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int):
        super().__init__()
        self.conv = GCNConv(in_channels, hidden_channels)
        self.bn   = BatchNorm1d(hidden_channels)

    def forward(self, x, edge_index, edge_weight, batch):
        x = self.bn(self.conv(x, edge_index, edge_weight).relu())
        return global_mean_pool(x, batch)


class BinaryGraphGNN(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = NODE_FEATURES,
        hidden_channels: int = 64,
    ):
        super().__init__()
        self.branch = _GraphBranch(in_channels, hidden_channels)
        self.lin1   = Linear(hidden_channels, hidden_channels)
        self.lin2   = Linear(hidden_channels, 1)

    def forward(self, data: GraphData) -> torch.Tensor:
        h = self.branch(data.x, data.edge_index, data.edge_weight, data.batch)
        h = self.lin1(h).relu()
        h = F.dropout(h, p=0.5, training=self.training)
        return self.lin2(h).squeeze(1)  # [N] logits


# ── Training helpers ──────────────────────────────────────────────────────────

def _train_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(batch), batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.detach().item() * batch.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def _predict(model, loader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        batch = batch.to(device)
        all_logits.append(model(batch).cpu())
        all_labels.append(batch.y.cpu())
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy().astype(int)
    probs  = 1 / (1 + np.exp(-logits))
    preds  = (probs > 0.5).astype(int)
    return labels, preds, probs


# ── Main entry point ──────────────────────────────────────────────────────────

def run(
    samples: list[tuple[str, list[float]]],
    k: str = "k25",
    method: str = "dreeb",
    k_folds: int = 5,
    hidden: int = 64,
    batch_size: int = 16,
    lr: float = 1e-3,
    epochs: int = 50,
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
        y_binary = np.array([lbl[cls_idx] for _, lbl in samples])
        pos, neg = int(y_binary.sum()), int((1 - y_binary).sum())
        print(f"\n{'─'*50}")
        print(f"  {cls_name}  (pos={pos}, neg={neg})")
        print(f"{'─'*50}")

        for fold, (train_idx, test_idx) in enumerate(kf.split(samples)):
            print_fold_header(fold, k_folds)

            train_samples = [samples[i] for i in train_idx]
            test_samples  = [samples[i] for i in test_idx]

            y_train_fold = np.array([lbl[cls_idx] for _, lbl in train_samples])
            if len(np.unique(y_train_fold)) < 2:
                print(f"  Skipping fold {fold + 1}: training split has only one class.")
                continue

            train_ds = SampleDataset(train_samples, cls_idx=cls_idx, k=k, method=method)
            test_ds  = SampleDataset(test_samples,  cls_idx=cls_idx, k=k, method=method)

            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

            model     = BinaryGraphGNN(hidden_channels=hidden).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = torch.nn.BCEWithLogitsLoss()

            for epoch in range(1, epochs + 1):
                _train_epoch(model, train_loader, optimizer, criterion, device)

            labels, preds, probs = _predict(model, test_loader, device)
            m = binary_metrics(labels, preds, y_score=probs)
            fold_results[cls_name].append({"fold": fold + 1, **m})
            print(f"  Acc: {m['accuracy']:.4f}  F1: {m['f1']:.4f}  AUC: {m.get('auc', float('nan')):.4f}")

    summarize_binary_folds(fold_results, CLASS_ORDER, label=f"GNN  method={method}")
    return fold_results


# ── Standalone ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pathlib
    from dataset import discover_samples

    METHOD      = "dreeb"
    DATA_DIR    = pathlib.Path(__file__).parent.parent / "data"
    LABELS_JSON = sorted((pathlib.Path(__file__).parent.parent).glob("phate_gallery_labels*/labels.json"))[-1]
    K           = "k25"

    samples = discover_samples(str(DATA_DIR), str(LABELS_JSON), k=K, method=METHOD)
    print(f"\nUsable samples: {len(samples)}  |  method: {METHOD}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    run(samples, k=K, method=METHOD, device=device)
