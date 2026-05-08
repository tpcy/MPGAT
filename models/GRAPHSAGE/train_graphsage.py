import argparse
import importlib.util
import random
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Planetoid import Planetoid


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_num_neighbors(text: str, num_layers: int) -> List[int]:
    vals = [int(x.strip()) for x in text.split(",") if x.strip()]
    vals = [max(-1, v) for v in vals]
    if not vals:
        vals = [25, 10]
    if len(vals) < num_layers:
        vals.extend([vals[-1]] * (num_layers - len(vals)))
    return vals[:num_layers]


def has_neighbor_sampling_backend() -> bool:
    if importlib.util.find_spec("pyg_lib") is not None:
        try:
            import pyg_lib  # noqa: F401
            return True
        except Exception:
            pass

    if importlib.util.find_spec("torch_sparse") is not None:
        try:
            import torch_sparse  # noqa: F401
            return True
        except Exception:
            pass

    return False


class GraphSAGE(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be >= 2")

        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


def build_train_loader(data, num_neighbors: List[int], batch_size: int, num_workers: int):
    return NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        input_nodes=data.train_mask,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )


def train_step_full_batch(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    logits = model(data.x, data.edge_index)
    loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return float(loss.item())


def train_step(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_examples = 0

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch.x, batch.edge_index)
        target = batch.y[: batch.batch_size]
        loss = F.cross_entropy(logits[: batch.batch_size], target)
        loss.backward()
        optimizer.step()

        examples = int(batch.batch_size)
        total_loss += float(loss.item()) * examples
        total_examples += examples

    return total_loss / max(total_examples, 1)


@torch.no_grad()
def evaluate(model, data):
    model.eval()
    logits = model(data.x, data.edge_index)
    pred = logits.argmax(dim=-1)
    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        acc = int((pred[mask] == data.y[mask]).sum()) / int(mask.sum())
        accs.append(acc)
    return accs


def run_once(args, data_cpu, data_eval, dataset, run_id):
    seed = args.seed + run_id
    set_seed(seed)
    model = GraphSAGE(
        in_channels=dataset.num_features,
        hidden_channels=args.hidden_channels,
        out_channels=dataset.num_classes,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(data_eval.x.device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    use_sampled_training = has_neighbor_sampling_backend()
    num_neighbors = parse_num_neighbors(args.num_neighbors, args.num_layers)
    train_loader = None
    if use_sampled_training:
        train_loader = build_train_loader(
            data=data_cpu,
            num_neighbors=num_neighbors,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

    best_val_acc = 0.0
    best_test_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        if use_sampled_training:
            loss = train_step(model, train_loader, optimizer, data_eval.x.device)
        else:
            loss = train_step_full_batch(model, data_eval, optimizer)
        train_acc, val_acc, test_acc = evaluate(model, data_eval)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(
                f"Run {run_id + 1:02d} | Epoch {epoch:04d} | "
                f"loss {loss:.4f} | train {train_acc:.4f} | "
                f"val {val_acc:.4f} | test {test_acc:.4f} | "
                f"time {time.time() - t0:.4f}s"
            )

    return best_val_acc, best_test_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="pubmed", choices=["cora", "citeseer", "pubmed"])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=20)

    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-4)

    parser.add_argument("--num_neighbors", type=str, default="25,10")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument(
        "--result_file",
        type=str,
        default="graphsage_baseline02/results_graphsage02.txt",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = Planetoid("./data/Planetoid/", args.dataset, transform=T.NormalizeFeatures())
    data_cpu = dataset[0]
    data_eval = data_cpu.to(device)

    best_test_list = []
    best_val_list = []

    print(
        f"GraphSAGE baseline02 | dataset={args.dataset} | runs={args.runs} | "
        f"layers={args.num_layers} hidden={args.hidden_channels} "
        f"neighbors={args.num_neighbors} batch={args.batch_size} "
        f"dropout={args.dropout} lr={args.lr} wd={args.weight_decay}"
    )
    if not has_neighbor_sampling_backend():
        print("Warning: 'pyg-lib/torch-sparse' not found, fallback to full-batch training.")

    for run_id in range(args.runs):
        best_val, best_test = run_once(args, data_cpu, data_eval, dataset, run_id)
        best_val_list.append(best_val)
        best_test_list.append(best_test)
        print(f"Run {run_id + 1:02d} best val={best_val:.4f} | best test={best_test:.4f}")

    avg_val = sum(best_val_list) / len(best_val_list)
    avg_test = sum(best_test_list) / len(best_test_list)
    print(f"\nAverage best val over {args.runs} runs: {avg_val:.5f}")
    print(f"Average best test over {args.runs} runs: {avg_test:.5f}")

    result_path = Path(args.result_file)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("a", encoding="utf-8") as f:
        f.write(
            f"dataset={args.dataset} runs={args.runs} epochs={args.epochs} "
            f"layers={args.num_layers} hidden={args.hidden_channels} "
            f"neighbors={args.num_neighbors} batch_size={args.batch_size} "
            f"dropout={args.dropout} lr={args.lr} wd={args.weight_decay} "
            f"avg_best_val={avg_val:.5f} avg_best_test={avg_test:.5f}\n"
        )
    print(f"Saved summary to: {result_path}")


if __name__ == "__main__":
    main()
