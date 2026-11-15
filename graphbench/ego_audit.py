"""Reproduce the degree-label diagnostic from the original Ego-Facebook notebook."""

import argparse
import json
from collections import Counter
from pathlib import Path


def degree_bucket(degree: int) -> str:
    if degree > 10:
        return "degree_above_10"
    if degree > 1:
        return "degree_2_to_10"
    return "degree_at_most_1"


def audit(edge_list: Path) -> dict:
    edges = set()
    degrees = Counter()
    with edge_list.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            fields = line.split("#", 1)[0].split()
            if not fields:
                continue
            if len(fields) != 2:
                raise ValueError(f"{edge_list}:{line_number}: expected two node IDs")
            edge = tuple(sorted(fields))
            if edge not in edges:
                edges.add(edge)
                degrees.update(edge)
    if not degrees:
        raise ValueError("The edge list contains no nodes.")

    # The same raw degree was used for the target and, after normalization, as an input feature.
    labels = {node: degree_bucket(degree) for node, degree in degrees.items()}
    class_counts = Counter(labels.values())
    predictions = {node: degree_bucket(degree) for node, degree in degrees.items()}
    return {
        "nodes": len(degrees),
        "undirected_edges": len(edges),
        "class_counts": dict(sorted(class_counts.items())),
        "majority_class_accuracy": max(class_counts.values()) / len(degrees),
        "target_definition_accuracy": sum(predictions[node] == label for node, label in labels.items()) / len(degrees),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("edge_list", type=Path)
    args = parser.parse_args()
    try:
        result = audit(args.edge_list)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
