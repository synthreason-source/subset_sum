#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import networkx as nx
from tqdm import tqdm


def parse_size(value: str) -> int:
    """
    Parse sizes such as:

        500MB
        500MiB
        1GB
    """
    text = value.strip().upper()

    units = {
        "B": 1,
        "KB": 1_000,
        "MB": 1_000_000,
        "GB": 1_000_000_000,
        "KIB": 1024,
        "MIB": 1024**2,
        "GIB": 1024**3,
    }

    for unit in sorted(units, key=len, reverse=True):
        if text.endswith(unit):
            number = text[:-len(unit)]
            return int(float(number) * units[unit])

    return int(text)


def generate_social_graph(
    rng: random.Random,
    min_nodes: int = 20,
    max_nodes: int = 80,
):
    """
    Generate a connected labelled social-network graph.
    """
    node_count = rng.randint(min_nodes, max_nodes)
    probability = rng.uniform(0.03, 0.12)

    while True:
        graph = nx.gnp_random_graph(
            node_count,
            probability,
            seed=rng.randrange(2**32),
        )

        if nx.is_connected(graph):
            break

    roles = [
        "member",
        "admin",
        "moderator",
        "researcher",
        "developer",
        "manager",
    ]

    departments = [
        "engineering",
        "science",
        "operations",
        "design",
        "management",
    ]

    relationship_types = [
        "friend",
        "follows",
        "colleague",
        "collaborates",
        "messages",
    ]

    for node in graph.nodes:
        graph.nodes[node]["role"] = rng.choice(roles)
        graph.nodes[node]["department"] = rng.choice(
            departments
        )

    for u, v in graph.edges:
        graph.edges[u, v]["relationship"] = rng.choice(
            relationship_types
        )
        graph.edges[u, v]["weight"] = round(
            rng.uniform(0.1, 1.0),
            4,
        )

    return graph


def graph_to_record(graph: nx.Graph):
    """
    Convert a NetworkX graph to a compact JSON-compatible object.
    """
    nodes = [
        {
            "id": int(node),
            "role": attributes["role"],
            "department": attributes["department"],
        }
        for node, attributes in graph.nodes(data=True)
    ]

    edges = [
        {
            "u": int(u),
            "v": int(v),
            "relationship": attributes["relationship"],
            "weight": attributes["weight"],
        }
        for u, v, attributes in graph.edges(data=True)
    ]

    return {
        "nodes": nodes,
        "edges": edges,
    }


def relabel_graph_randomly(
    graph: nx.Graph,
    rng: random.Random,
):
    """
    Create an isomorphic graph with randomly permuted node IDs.
    """
    original_nodes = list(graph.nodes())
    shuffled_nodes = original_nodes.copy()

    rng.shuffle(shuffled_nodes)

    mapping = dict(
        zip(original_nodes, shuffled_nodes)
    )

    permuted_graph = nx.relabel_nodes(
        graph,
        mapping,
        copy=True,
    )

    return permuted_graph, mapping


def make_non_isomorphic_pair(
    graph: nx.Graph,
    rng: random.Random,
):
    """
    Generate a graph that is not isomorphic to graph.
    """
    for _ in range(100):
        candidate = generate_social_graph(
            rng,
            min_nodes=graph.number_of_nodes(),
            max_nodes=graph.number_of_nodes(),
        )

        if not nx.is_isomorphic(graph, candidate):
            return candidate

    raise RuntimeError(
        "Could not generate a non-isomorphic graph."
    )


def make_dataset_record(
    rng: random.Random,
    non_isomorphic_fraction: float,
):
    """
    Generate one graph pair and its ground-truth label.
    """
    graph_a = generate_social_graph(rng)

    is_non_isomorphic = (
        rng.random() < non_isomorphic_fraction
    )

    if is_non_isomorphic:
        graph_b = make_non_isomorphic_pair(
            graph_a,
            rng,
        )
        mapping = None
        isomorphic = False
    else:
        graph_b, mapping = relabel_graph_randomly(
            graph_a,
            rng,
        )
        isomorphic = True

    return {
        "isomorphic": isomorphic,
        "graph_a": graph_to_record(graph_a),
        "graph_b": graph_to_record(graph_b),
        "mapping": (
            {
                str(source): int(target)
                for source, target in mapping.items()
            }
            if mapping is not None
            else None
        ),
    }


def encode_record(record: dict) -> bytes:
    """
    Encode one record as compact newline-delimited JSON.
    """
    return (
        json.dumps(
            record,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def generate_dataset(
    output_path: str,
    target_bytes: int,
    seed: int = 12345,
    non_isomorphic_fraction: float = 0.5,
):
    """
    Generate a dataset until it reaches target_bytes.

    The final size can be slightly larger than target_bytes because complete
    records are written atomically.
    """
    rng = random.Random(seed)

    output = Path(output_path)
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    record_count = 0
    bytes_written = 0

    with output.open("wb") as file:
        with tqdm(
            total=target_bytes,
            desc="Generating dataset",
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            dynamic_ncols=True,
        ) as progress:
            while bytes_written < target_bytes:
                record = make_dataset_record(
                    rng,
                    non_isomorphic_fraction,
                )

                encoded = encode_record(record)
                file.write(encoded)

                record_size = len(encoded)
                bytes_written += record_size
                record_count += 1

                progress.update(record_size)
                progress.set_postfix(
                    records=f"{record_count:,}",
                    refresh=False,
                )

    metadata = {
        "target_bytes": target_bytes,
        "actual_bytes": bytes_written,
        "actual_megabytes": bytes_written / 1_000_000,
        "actual_mebibytes": bytes_written / (1024**2),
        "records": record_count,
        "seed": seed,
        "non_isomorphic_fraction": non_isomorphic_fraction,
        "format": "newline-delimited JSON",
    }

    metadata_path = output.with_suffix(
        output.suffix + ".metadata.json"
    )

    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print()
    print(f"Dataset written to: {output}")
    print(f"Records: {record_count:,}")
    print(
        f"Size: {bytes_written / (1024**2):.2f} MiB"
    )
    print(f"Metadata written to: {metadata_path}")


def graph_from_record(record: dict):
    """
    Reconstruct a graph from a serialized graph record.
    """
    graph = nx.Graph()

    for node in record["nodes"]:
        graph.add_node(
            node["id"],
            role=node["role"],
            department=node["department"],
        )

    for edge in record["edges"]:
        graph.add_edge(
            edge["u"],
            edge["v"],
            relationship=edge["relationship"],
            weight=edge["weight"],
        )

    return graph


def verify_dataset(
    dataset_path: str,
    records_to_check: int = 100,
):
    """
    Verify the labels in the first records.
    """
    node_match = nx.algorithms.isomorphism.categorical_node_match(
        ["role", "department"],
        [None, None],
    )

    edge_match = nx.algorithms.isomorphism.categorical_edge_match(
        ["relationship", "weight"],
        [None, None],
    )

    checked = 0
    failures = 0

    with open(dataset_path, "rb") as file:
        with tqdm(
            total=records_to_check,
            desc="Verifying records",
            unit="record",
            dynamic_ncols=True,
        ) as progress:
            for line in file:
                if checked >= records_to_check:
                    break

                record = json.loads(line)

                graph_a = graph_from_record(
                    record["graph_a"]
                )
                graph_b = graph_from_record(
                    record["graph_b"]
                )

                actual = nx.is_isomorphic(
                    graph_a,
                    graph_b,
                    node_match=node_match,
                    edge_match=edge_match,
                )

                if actual != record["isomorphic"]:
                    failures += 1

                checked += 1
                progress.update(1)
                progress.set_postfix(
                    failures=failures,
                    refresh=False,
                )

    print(
        f"Checked {checked:,} records; "
        f"failures: {failures:,}"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output",
        default="social_isomorphism_10mb.jsonl",
    )

    parser.add_argument(
        "--size",
        default="10MB",
        help="Examples: 500MB, 500MiB, 1GB",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
    )

    parser.add_argument(
        "--non-isomorphic-fraction",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--verify",
        type=int,
        default=0,
        help="Verify this many records after generation.",
    )

    args = parser.parse_args()

    generate_dataset(
        output_path=args.output,
        target_bytes=parse_size(args.size),
        seed=args.seed,
        non_isomorphic_fraction=(
            args.non_isomorphic_fraction
        ),
    )

    if args.verify > 0:
        verify_dataset(
            args.output,
            records_to_check=args.verify,
        )


if __name__ == "__main__":
    main()