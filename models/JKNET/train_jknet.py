import argparse
import random
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, JumpingKnowledge

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


class JKNet(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        jk_mode: str,
        dropout: float,
        include_input_proj: bool = True,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if jk_mode not in {"cat", "max", "lstm"}:
            raise ValueError("jk_mode must be one of: cat, max, lstm")

        self.dropout = dropout
        self.num_layers = num_layers
        self.include_input_proj = include_input_proj

        self.input_proj = None
        if include_input_proj:
            self.input_proj = torch.nn.Linear(in_channels, hidden_channels, bias=False)

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        num_reps = num_layers + (1 if include_input_proj else 0)
        self.jk = JumpingKnowledge(jk_mode, channels=hidden_channels, num_layers=num_reps)

        if jk_mode == "cat":
            out_in_channels = hidden_channels * num_reps
        else:
            out_in_channels = hidden_channels
        self.classifier = torch.nn.Linear(out_in_channels, out_channels)

    def forward(self, x, edge_index):
        reps = []
        if self.input_proj is not None:
            reps.append(self.input_proj(x))

        h = x
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            reps.append(h)

        h = self.jk(reps)
        return self.classifier(h)


def train_step(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    logits = model(data.x, data.edge_index)
    loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return float(loss.item())


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


def run_single_model(args, data, dataset, run_seed: int, num_layers: int):
    set_seed(run_seed)
    model = JKNet(
        in_channels=dataset.num_features,
        hidden_channels=args.hidden_channels,
        out_channels=dataset.num_classes,
        num_layers=num_layers,
        jk_mode=args.jk_mode,
        dropout=args.dropout,
        include_input_proj=(not args.no_input_proj),
    ).to(data.x.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = 0.0
    best_test_acc = 0.0
    best_epoch = 1
    stale = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        loss = train_step(model, data, optimizer)
        train_acc, val_acc, test_acc = evaluate(model, data)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            best_epoch = epoch
            stale = 0
        else:
            stale += 1

        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(
                f"  layers={num_layers} | epoch {epoch:04d} | loss {loss:.4f} | "
                f"train {train_acc:.4f} | val {val_acc:.4f} | test {test_acc:.4f} | "
                f"time {time.time() - t0:.4f}s"
            )

        if args.patience > 0 and stale >= args.patience:
            break

    return best_val_acc, best_test_acc, best_epoch


def parse_layer_candidates(text: str) -> List[int]:
    vals = []
    for s in text.split(","):
        s = s.strip()
        if not s:
            continue
        vals.append(int(s))
    vals = sorted(set([v for v in vals if v >= 1]))
    if not vals:
        raise ValueError("layer_candidates is empty after parsing.")
    return vals


def run_once(args, data, dataset, run_id):
    base_seed = args.seed + run_id
    if args.sweep_layers:
        layer_candidates = parse_layer_candidates(args.layer_candidates)
    else:
        layer_candidates = [args.num_layers]

    best_overall = {
        "val": -1.0,
        "test": -1.0,
        "layers": None,
        "epoch": None,
    }

    for layers in layer_candidates:
        print(f"Run {run_id + 1:02d} | training JK-Net with layers={layers}")
        best_val, best_test, best_epoch = run_single_model(
            args=args,
            data=data,
            dataset=dataset,
            run_seed=base_seed,
            num_layers=layers,
        )
        print(
            f"Run {run_id + 1:02d} | layers={layers} | "
            f"best val={best_val:.4f} best test={best_test:.4f} @epoch={best_epoch}"
        )
        if best_val > best_overall["val"]:
            best_overall["val"] = best_val
            best_overall["test"] = best_test
            best_overall["layers"] = layers
            best_overall["epoch"] = best_epoch

    return best_overall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="pubmed", choices=["cora", "citeseer", "pubmed"])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--patience", type=int, default=100)

    # JK-Net settings:
    parser.add_argument("--jk_mode", type=str, default="max", choices=["cat", "max", "lstm"])
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--sweep_layers", action="store_true", help="Search best depth on val set.")
    parser.add_argument("--layer_candidates", type=str, default="1,2,3,4,5,6")
    parser.add_argument("--no_input_proj", action="store_true", help="Disable h0 input projection in JK fusion.")

    # Paper-style defaults (citation setup):
    parser.add_argument("--hidden_channels", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight_decay", type=float, default=5e-4)

    parser.add_argument("--result_file", type=str, default="jknet_baseline02/results_jknet02.txt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = Planetoid("./data/Planetoid/", args.dataset, transform=T.NormalizeFeatures())
    data = dataset[0].to(device)

    print(
        f"JK-Net baseline02 | dataset={args.dataset} runs={args.runs} | "
        f"jk_mode={args.jk_mode} hidden={args.hidden_channels} "
        f"dropout={args.dropout} lr={args.lr} wd={args.weight_decay} "
        f"sweep_layers={args.sweep_layers}"
    )

    best_vals = []
    best_tests = []
    best_layers = []

    for run_id in range(args.runs):
        result = run_once(args, data, dataset, run_id)
        best_vals.append(result["val"])
        best_tests.append(result["test"])
        best_layers.append(result["layers"])
        print(
            f"Run {run_id + 1:02d} final best -> "
            f"layers={result['layers']} val={result['val']:.4f} test={result['test']:.4f}"
        )

    avg_val = float(sum(best_vals) / len(best_vals))
    avg_test = float(sum(best_tests) / len(best_tests))
    layer_str = ",".join(map(str, best_layers))

    print(f"\nAverage best val over {args.runs} runs: {avg_val:.5f}")
    print(f"Average best test over {args.runs} runs: {avg_test:.5f}")
    print(f"Chosen layers per run: {layer_str}")

    result_path = Path(args.result_file)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("a", encoding="utf-8") as f:
        f.write(
            f"dataset={args.dataset} runs={args.runs} epochs={args.epochs} "
            f"jk_mode={args.jk_mode} sweep_layers={args.sweep_layers} "
            f"num_layers={args.num_layers} layer_candidates={args.layer_candidates} "
            f"hidden={args.hidden_channels} dropout={args.dropout} lr={args.lr} "
            f"wd={args.weight_decay} patience={args.patience} "
            f"avg_best_val={avg_val:.5f} avg_best_test={avg_test:.5f} "
            f"best_layers={layer_str}\n"
        )
    print(f"Saved summary to: {result_path}")


if __name__ == "__main__":
    main()
