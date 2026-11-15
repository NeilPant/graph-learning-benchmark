"""Compare citation edges, shuffled node assignments, and self-loops on Cora."""

import argparse
import hashlib
import json
import platform
from pathlib import Path
from statistics import mean, stdev

import sklearn
import torch
import torch.nn.functional as F
import torch_geometric
from sklearn.metrics import accuracy_score, f1_score
from torch import Tensor, nn
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv


CONDITIONS = ("citation_edges", "self_loops_only", "shuffled_node_ids")
DEFAULT_SEEDS = (42, 43, 44, 45, 46)


class Classifier(nn.Module):
    def __init__(self, inputs: int, classes: int) -> None:
        super().__init__()
        self.hidden = GCNConv(inputs, 16, cached=True)
        self.output = GCNConv(16, classes, cached=True)

    def forward(self, features: Tensor, edges: Tensor) -> Tensor:
        hidden = self.hidden(features, edges).relu()
        hidden = F.dropout(hidden, p=0.5, training=self.training)
        return self.output(hidden, edges)


def condition_edges(data: Data, condition: str, seed: int) -> Tensor:
    if condition == "citation_edges":
        return data.edge_index
    if condition == "self_loops_only":
        # GCNConv adds a self-loop at each node, so both layers reduce to linear transforms.
        return data.edge_index[:, :0]
    if condition == "shuffled_node_ids":
        generator = torch.Generator().manual_seed(seed)
        permutation = torch.randperm(data.num_nodes, generator=generator)
        return permutation[data.edge_index]
    raise ValueError(f"Unknown condition: {condition}")


def run_seed(data: Data, classes: int, condition: str, seed: int, epochs: int, patience: int) -> dict:
    torch.manual_seed(seed)
    edges = condition_edges(data, condition, seed)
    model = Classifier(data.num_node_features, classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    best_loss = float("inf")
    best_epoch = 0
    best_weights = None
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(data.x, edges)
        train_loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_logits = model(data.x, edges)[data.val_mask]
            validation_loss = F.cross_entropy(validation_logits, data.y[data.val_mask]).item()
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_weights = {name: weight.detach().clone() for name, weight in model.state_dict().items()}
        if epoch - best_epoch >= patience:
            break

    if best_weights is None:
        raise RuntimeError("Training produced no finite validation checkpoint.")
    model.load_state_dict(best_weights)
    with torch.no_grad():
        predicted = model(data.x, edges)[data.test_mask].argmax(dim=1).tolist()
    targets = data.y[data.test_mask].tolist()
    return {
        "condition": condition,
        "seed": seed,
        "best_epoch": best_epoch,
        "validation_loss": best_loss,
        "test_accuracy": accuracy_score(targets, predicted),
        "test_macro_f1": f1_score(targets, predicted, labels=list(range(classes)), average="macro", zero_division=0),
    }


def summarize(runs: list[dict]) -> dict:
    summary = {}
    for condition in CONDITIONS:
        group = [run for run in runs if run["condition"] == condition]
        scores = {}
        for metric in ("test_accuracy", "test_macro_f1"):
            values = [run[metric] for run in group]
            scores[metric] = {"mean": mean(values), "sample_sd": stdev(values) if len(values) > 1 else None}
        summary[condition] = scores
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/planetoid"))
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--output", type=Path, help="Also save the JSON results to this path.")
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 1:
        parser.error("--epochs and --patience must be positive.")
    if len(set(args.seeds)) != len(args.seeds) or any(not 0 <= seed < 2**32 for seed in args.seeds):
        parser.error("--seeds must contain distinct integers from 0 through 2**32 - 1.")

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    dataset = Planetoid(str(args.data_dir), "Cora", split="public")
    data = dataset[0]
    runs = [
        run_seed(data, dataset.num_classes, condition, seed, args.epochs, args.patience)
        for seed in args.seeds
        for condition in CONDITIONS
    ]
    result = {
        "dataset": "Cora",
        "split": "Planetoid public",
        "nodes": data.num_nodes,
        "directed_edge_entries": data.num_edges,
        "features": data.num_node_features,
        "classes": dataset.num_classes,
        "raw_data_sha256": {
            Path(path).name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for path in dataset.raw_paths
        },
        "split_sizes": {
            "train": int(data.train_mask.sum()),
            "validation": int(data.val_mask.sum()),
            "test": int(data.test_mask.sum()),
        },
        "epochs_max": args.epochs,
        "patience": args.patience,
        "seeds": args.seeds,
        "environment": {
            "python": platform.python_version(),
            "os": platform.system(),
            "architecture": platform.machine(),
            "torch": torch.__version__,
            "torch_geometric": torch_geometric.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "summary": summarize(runs),
        "runs": runs,
    }
    rendered = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
