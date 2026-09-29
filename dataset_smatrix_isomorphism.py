#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random

from dataclasses import dataclass
from typing import Any, Dict, Hashable, Iterator, Optional, Tuple

import networkx as nx
import numpy as np

from tqdm import tqdm

from networkx.algorithms import isomorphism as iso
from networkx.algorithms.graph_hashing import (
    weisfeiler_lehman_graph_hash,
)


Node = Hashable
GraphLike = nx.Graph


# ============================================================================
# EXACT / HEURISTIC GRAPH COMPARISON
# ============================================================================

@dataclass
class IsomorphismResult:
    """
    Result of an isomorphism comparison.
    """

    isomorphic: bool
    mapping: Optional[Dict[Node, Node]]
    rejected_by: Optional[str]
    wl_hash_equal: bool
    spectral_distance: float
    similarity: float


def _graph_kind(
    graph: GraphLike,
) -> Tuple[bool, bool]:
    return (
        graph.is_directed(),
        graph.is_multigraph(),
    )


def _node_degree_signature(graph: GraphLike):
    """
    Degree signature for directed and undirected graphs.
    """
    if graph.is_directed():
        values = [
            (
                graph.in_degree(node),
                graph.out_degree(node),
            )
            for node in graph.nodes()
        ]
    else:
        values = [
            graph.degree(node)
            for node in graph.nodes()
        ]

    return tuple(sorted(values))


def _edge_count_signature(graph: GraphLike):
    """
    Edge multiplicity signature.
    """
    if not graph.is_multigraph():
        return tuple(
            [1] * graph.number_of_edges()
        )

    multiplicities = []

    for source, target in graph.edges():
        multiplicities.append(
            graph.number_of_edges(source, target)
        )

    return tuple(sorted(multiplicities))


def _component_signature(graph: GraphLike):
    """
    Component signature.

    Weak components are used for directed graphs.
    """
    if graph.is_directed():
        components = nx.weakly_connected_components(
            graph
        )
    else:
        components = nx.connected_components(
            graph
        )

    signatures = []

    for component in components:
        component = set(component)

        edge_count = sum(
            1
            for source, target in graph.edges()
            if source in component
            and target in component
        )

        signatures.append(
            (
                len(component),
                edge_count,
            )
        )

    return tuple(sorted(signatures))


def _triangle_signature(graph: GraphLike):
    """
    Sorted local triangle counts.

    This is a prefilter only.
    """
    simple_graph = nx.Graph(graph)

    if simple_graph.number_of_nodes() == 0:
        return ()

    triangles = nx.triangles(simple_graph)

    return tuple(
        sorted(triangles.values())
    )


def _adjacency_spectrum(graph: GraphLike):
    """
    Sorted adjacency eigenvalues.

    Directed graphs use their underlying undirected graph.
    """
    simple_graph = nx.Graph(graph)

    if simple_graph.number_of_nodes() == 0:
        return np.empty(
            0,
            dtype=np.float64,
        )

    adjacency = nx.to_numpy_array(
        simple_graph,
        nodelist=list(simple_graph.nodes()),
        dtype=np.float64,
    )

    eigenvalues = np.linalg.eigvalsh(adjacency)

    return np.sort(eigenvalues)


def _normalized_spectral_distance(
    graph_1: GraphLike,
    graph_2: GraphLike,
):
    spectrum_1 = _adjacency_spectrum(graph_1)
    spectrum_2 = _adjacency_spectrum(graph_2)

    if len(spectrum_1) != len(spectrum_2):
        return float("inf")

    scale = max(
        np.linalg.norm(spectrum_1),
        np.linalg.norm(spectrum_2),
        1.0,
    )

    return float(
        np.linalg.norm(
            spectrum_1 - spectrum_2
        )
        / scale
    )


def _stable_attribute_token(value: Any) -> str:
    """
    Convert common Python values into deterministic text.
    """
    if isinstance(value, dict):
        items = sorted(
            (
                str(key),
                _stable_attribute_token(item),
            )
            for key, item in value.items()
        )

        return "{" + ",".join(
            f"{key}:{item}"
            for key, item in items
        ) + "}"

    if isinstance(value, (list, tuple)):
        return "[" + ",".join(
            _stable_attribute_token(item)
            for item in value
        ) + "]"

    if isinstance(value, set):
        return "{" + ",".join(
            sorted(
                _stable_attribute_token(item)
                for item in value
            )
        ) + "}"

    return repr(value)


def _attribute_label(
    attributes: Dict[str, Any],
    keys: Optional[Tuple[str, ...]],
) -> str:
    if keys is None:
        selected = attributes
    else:
        selected = {
            key: attributes.get(key)
            for key in keys
        }

    return _stable_attribute_token(selected)


def _label_graph(
    graph: GraphLike,
    node_attributes: Optional[Tuple[str, ...]],
    edge_attributes: Optional[Tuple[str, ...]],
):
    """
    Copy a graph and add canonical WL labels.
    """
    labelled = graph.copy()

    for _, attributes in labelled.nodes(data=True):
        attributes["_wl_node_label"] = _attribute_label(
            attributes,
            node_attributes,
        )

    if labelled.is_multigraph():
        for (
            source,
            target,
            key,
            attributes,
        ) in labelled.edges(
            keys=True,
            data=True,
        ):
            attributes["_wl_edge_label"] = _attribute_label(
                attributes,
                edge_attributes,
            )
    else:
        for (
            source,
            target,
            attributes,
        ) in labelled.edges(data=True):
            attributes["_wl_edge_label"] = _attribute_label(
                attributes,
                edge_attributes,
            )

    return labelled


def _wl_hash(
    graph: GraphLike,
    node_attributes: Optional[Tuple[str, ...]],
    edge_attributes: Optional[Tuple[str, ...]],
    iterations: int,
):
    labelled = _label_graph(
        graph,
        node_attributes,
        edge_attributes,
    )

    return weisfeiler_lehman_graph_hash(
        labelled,
        node_attr="_wl_node_label",
        edge_attr="_wl_edge_label",
        iterations=iterations,
    )


def compute_graph_s_matrix(
    graph: GraphLike,
    max_hops: int = 4,
):
    """
    Compute a multi-scale structural Gram matrix.

    This is a heuristic diagnostic only.
    """
    if graph.number_of_nodes() == 0:
        return np.zeros(
            (1, 1),
            dtype=np.float64,
        )

    simple_graph = nx.Graph(graph)
    nodes = list(simple_graph.nodes())

    adjacency = nx.to_numpy_array(
        simple_graph,
        nodelist=nodes,
        dtype=np.float64,
    )

    degrees = np.sum(
        adjacency,
        axis=1,
        keepdims=True,
    )

    safe_degrees = degrees.copy()
    safe_degrees[
        safe_degrees == 0.0
    ] = 1.0

    transition = adjacency / safe_degrees

    features = [
        degrees,
        np.diag(adjacency).reshape(-1, 1),
    ]

    current_transition = transition.copy()

    for _ in range(2, max_hops + 1):
        features.append(
            np.sum(
                current_transition,
                axis=1,
                keepdims=True,
            )
        )

        current_transition = (
            current_transition @ transition
        )

    feature_matrix = np.hstack(features)

    mean = np.mean(
        feature_matrix,
        axis=0,
    )

    standard_deviation = np.std(
        feature_matrix,
        axis=0,
    )

    standard_deviation[
        standard_deviation < 1e-12
    ] = 1.0

    normalized = (
        feature_matrix - mean
    ) / standard_deviation

    return normalized.T @ normalized


def s_matrix_spectral_distance(
    graph_1: GraphLike,
    graph_2: GraphLike,
    max_hops: int = 4,
):
    """
    Compare the singular spectra of two S-matrices.
    """
    matrix_1 = compute_graph_s_matrix(
        graph_1,
        max_hops=max_hops,
    )

    matrix_2 = compute_graph_s_matrix(
        graph_2,
        max_hops=max_hops,
    )

    _, singular_values_1, _ = np.linalg.svd(
        matrix_1,
        full_matrices=False,
    )

    _, singular_values_2, _ = np.linalg.svd(
        matrix_2,
        full_matrices=False,
    )

    singular_values_1 /= max(
        np.linalg.norm(singular_values_1),
        1e-12,
    )

    singular_values_2 /= max(
        np.linalg.norm(singular_values_2),
        1e-12,
    )

    return float(
        np.linalg.norm(
            singular_values_1 - singular_values_2
        )
    )


def _numeric_similarity(
    graph_1: GraphLike,
    graph_2: GraphLike,
    wl_hash_equal: bool,
    spectral_distance: float,
):
    """
    Heuristic score only.
    """
    if (
        graph_1.number_of_nodes()
        != graph_2.number_of_nodes()
    ):
        return 0.0

    if (
        graph_1.number_of_edges()
        != graph_2.number_of_edges()
    ):
        return 0.0

    degree_equal = (
        _node_degree_signature(graph_1)
        == _node_degree_signature(graph_2)
    )

    triangle_equal = (
        _triangle_signature(graph_1)
        == _triangle_signature(graph_2)
    )

    component_equal = (
        _component_signature(graph_1)
        == _component_signature(graph_2)
    )

    score = 0.0
    score += 0.35 if degree_equal else 0.0
    score += 0.25 if triangle_equal else 0.0
    score += 0.20 if component_equal else 0.0
    score += 0.15 if wl_hash_equal else 0.0
    score += 0.05 * np.exp(
        -spectral_distance
    )

    return float(min(score, 1.0))


def _build_node_match(
    node_attributes: Optional[Tuple[str, ...]],
):
    if node_attributes is None:
        return None

    def node_match(
        attributes_1,
        attributes_2,
    ):
        return all(
            attributes_1.get(key)
            == attributes_2.get(key)
            for key in node_attributes
        )

    return node_match


def _build_edge_match(
    edge_attributes: Optional[Tuple[str, ...]],
):
    if edge_attributes is None:
        return None

    def edge_match(
        attributes_1,
        attributes_2,
    ):
        return all(
            attributes_1.get(key)
            == attributes_2.get(key)
            for key in edge_attributes
        )

    return edge_match


def _exact_mapping(
    graph_1: GraphLike,
    graph_2: GraphLike,
    node_attributes: Optional[Tuple[str, ...]],
    edge_attributes: Optional[Tuple[str, ...]],
):
    """
    Exact VF2 isomorphism matching.
    """
    if (
        graph_1.is_directed()
        != graph_2.is_directed()
    ):
        return None

    if (
        graph_1.is_multigraph()
        != graph_2.is_multigraph()
    ):
        return None

    node_match = _build_node_match(
        node_attributes
    )

    edge_match = _build_edge_match(
        edge_attributes
    )

    if graph_1.is_directed():
        if graph_1.is_multigraph():
            matcher_class = iso.MultiDiGraphMatcher
        else:
            matcher_class = iso.DiGraphMatcher
    else:
        if graph_1.is_multigraph():
            matcher_class = iso.MultiGraphMatcher
        else:
            matcher_class = iso.GraphMatcher

    matcher = matcher_class(
        graph_1,
        graph_2,
        node_match=node_match,
        edge_match=edge_match,
    )

    if not matcher.is_isomorphic():
        return None

    return dict(matcher.mapping)


def compare_graphs(
    graph_1: GraphLike,
    graph_2: GraphLike,
    *,
    node_attributes: Optional[Tuple[str, ...]] = None,
    edge_attributes: Optional[Tuple[str, ...]] = None,
    wl_iterations: int = 3,
    use_prefilters: bool = True,
    strict_prefilters: bool = False,
    spectral_tolerance: float = 1e-8,
):
    """
    Compare two graphs.

    The final positive result always comes from exact VF2 matching.

    By default, WL and spectral mismatches do not reject a graph pair,
    because those invariants are heuristic and incomplete.
    """
    valid_types = (
        nx.Graph,
        nx.DiGraph,
        nx.MultiGraph,
        nx.MultiDiGraph,
    )

    if not isinstance(graph_1, valid_types):
        raise TypeError(
            "graph_1 must be a NetworkX graph."
        )

    if not isinstance(graph_2, valid_types):
        raise TypeError(
            "graph_2 must be a NetworkX graph."
        )

    directed_1, multigraph_1 = _graph_kind(
        graph_1
    )

    directed_2, multigraph_2 = _graph_kind(
        graph_2
    )

    if directed_1 != directed_2:
        return IsomorphismResult(
            False,
            None,
            "directedness",
            False,
            float("inf"),
            0.0,
        )

    if multigraph_1 != multigraph_2:
        return IsomorphismResult(
            False,
            None,
            "multigraph_type",
            False,
            float("inf"),
            0.0,
        )

    if (
        graph_1.number_of_nodes()
        != graph_2.number_of_nodes()
    ):
        return IsomorphismResult(
            False,
            None,
            "node_count",
            False,
            float("inf"),
            0.0,
        )

    if (
        graph_1.number_of_edges()
        != graph_2.number_of_edges()
    ):
        return IsomorphismResult(
            False,
            None,
            "edge_count",
            False,
            float("inf"),
            0.0,
        )

    if (
        _node_degree_signature(graph_1)
        != _node_degree_signature(graph_2)
    ):
        return IsomorphismResult(
            False,
            None,
            "degree_signature",
            False,
            float("inf"),
            0.0,
        )

    if (
        _edge_count_signature(graph_1)
        != _edge_count_signature(graph_2)
    ):
        return IsomorphismResult(
            False,
            None,
            "edge_multiplicity",
            False,
            float("inf"),
            0.0,
        )

    if (
        _component_signature(graph_1)
        != _component_signature(graph_2)
    ):
        return IsomorphismResult(
            False,
            None,
            "component_signature",
            False,
            float("inf"),
            0.0,
        )

    if (
        _triangle_signature(graph_1)
        != _triangle_signature(graph_2)
    ):
        return IsomorphismResult(
            False,
            None,
            "triangle_signature",
            False,
            float("inf"),
            0.0,
        )

    hash_1 = _wl_hash(
        graph_1,
        node_attributes,
        edge_attributes,
        wl_iterations,
    )

    hash_2 = _wl_hash(
        graph_2,
        node_attributes,
        edge_attributes,
        wl_iterations,
    )

    wl_hash_equal = hash_1 == hash_2

    spectral_distance = (
        _normalized_spectral_distance(
            graph_1,
            graph_2,
        )
    )

    similarity = _numeric_similarity(
        graph_1,
        graph_2,
        wl_hash_equal,
        spectral_distance,
    )

    if (
        use_prefilters
        and strict_prefilters
        and not wl_hash_equal
    ):
        return IsomorphismResult(
            False,
            None,
            "weisfeiler_lehman_hash",
            False,
            spectral_distance,
            similarity,
        )

    if (
        use_prefilters
        and strict_prefilters
        and spectral_tolerance is not None
        and spectral_distance > spectral_tolerance
    ):
        return IsomorphismResult(
            False,
            None,
            "adjacency_spectrum",
            wl_hash_equal,
            spectral_distance,
            similarity,
        )

    mapping = _exact_mapping(
        graph_1,
        graph_2,
        node_attributes,
        edge_attributes,
    )

    return IsomorphismResult(
        isomorphic=mapping is not None,
        mapping=mapping,
        rejected_by=(
            None
            if mapping is not None
            else "vf2_exact_match"
        ),
        wl_hash_equal=wl_hash_equal,
        spectral_distance=spectral_distance,
        similarity=similarity,
    )


# ============================================================================
# JSON GRAPH INPUT AND GENERATOR FORMAT
# ============================================================================

def parse_size(value: str) -> int:
    value = value.strip().upper()

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
        if value.endswith(unit):
            return int(
                float(value[:-len(unit)])
                * units[unit]
            )

    return int(value)


def iter_json_objects(
    path: str,
) -> Iterator[Any]:
    """
    Accept:

    - a JSON array;
    - one JSON graph object;
    - newline-delimited JSON.

    JSONL is recommended for very large source files.
    """
    with open(path, "r", encoding="utf-8") as file:
        content = file.read().lstrip()

    if not content:
        raise ValueError(
            "Input JSON file is empty."
        )

    if content.startswith("["):
        data = json.loads(content)

        if not isinstance(data, list):
            raise ValueError(
                "Expected a JSON array."
            )

        yield from data
        return

    try:
        data = json.loads(content)

        if isinstance(data, list):
            yield from data
        else:
            yield data

        return

    except json.JSONDecodeError:
        pass

    with open(path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line "
                    f"{line_number}: {error}"
                ) from error


def unwrap_graph_object(obj: Any) -> dict:
    """
    Find an object with nodes and edges.
    """
    if isinstance(obj, dict):
        if (
            "nodes" in obj
            and "edges" in obj
        ):
            return obj

        for key in (
            "graph",
            "graph_a",
            "data",
            "network",
        ):
            if key in obj:
                try:
                    return unwrap_graph_object(
                        obj[key]
                    )
                except ValueError:
                    pass

    raise ValueError(
        "Could not find graph fields "
        "'nodes' and 'edges'."
    )


def graph_from_json(
    obj: Any,
    directed: bool = False,
) -> GraphLike:
    """
    Convert common JSON graph formats into NetworkX.
    """
    data = unwrap_graph_object(obj)

    if directed:
        graph = nx.DiGraph()
    else:
        graph = nx.Graph()

    for node in data["nodes"]:
        if isinstance(node, dict):
            if "id" not in node:
                raise ValueError(
                    "Node object lacks an 'id' field."
                )

            node_id = node["id"]

            attributes = {
                key: value
                for key, value in node.items()
                if key != "id"
            }

        elif isinstance(node, (list, tuple)):
            node_id = node[0]

            if (
                len(node) > 1
                and isinstance(node[1], dict)
            ):
                attributes = node[1]
            else:
                attributes = {}

        else:
            node_id = node
            attributes = {}

        graph.add_node(
            node_id,
            **attributes,
        )

    for edge in data["edges"]:
        if isinstance(edge, dict):
            if (
                "u" in edge
                and "v" in edge
            ):
                source = edge["u"]
                target = edge["v"]

                attributes = {
                    key: value
                    for key, value in edge.items()
                    if key not in {"u", "v"}
                }

            elif (
                "source" in edge
                and "target" in edge
            ):
                source = edge["source"]
                target = edge["target"]

                attributes = {
                    key: value
                    for key, value in edge.items()
                    if key not in {
                        "source",
                        "target",
                    }
                }

            else:
                raise ValueError(
                    "Edge needs u/v or source/target."
                )

        elif isinstance(edge, (list, tuple)):
            if len(edge) < 2:
                raise ValueError(
                    "Edge needs at least two values."
                )

            source = edge[0]
            target = edge[1]

            if (
                len(edge) > 2
                and isinstance(edge[2], dict)
            ):
                attributes = edge[2]
            else:
                attributes = {}

        else:
            raise ValueError(
                "Unsupported edge format."
            )

        graph.add_edge(
            source,
            target,
            **attributes,
        )

    return graph


def generator_format(
    graph: GraphLike,
) -> dict:
    """
    Preserve the original generator format exactly:

        {
            "nodes": [...],
            "edges": [...]
        }
    """
    nodes = []

    for node, attributes in graph.nodes(
        data=True
    ):
        item = {
            "id": node,
        }

        item.update(attributes)
        nodes.append(item)

    edges = []

    if graph.is_multigraph():
        for (
            source,
            target,
            key,
            attributes,
        ) in graph.edges(
            keys=True,
            data=True,
        ):
            item = {
                "u": source,
                "v": target,
                "key": key,
            }

            item.update(attributes)
            edges.append(item)
    else:
        for (
            source,
            target,
            attributes,
        ) in graph.edges(data=True):
            item = {
                "u": source,
                "v": target,
            }

            item.update(attributes)
            edges.append(item)

    return {
        "nodes": nodes,
        "edges": edges,
    }


# ============================================================================
# DATASET GENERATION
# ============================================================================

def relabel_graph_randomly(
    graph: GraphLike,
    rng: random.Random,
):
    """
    Create an isomorphic copy with permuted IDs.
    """
    old_nodes = list(graph.nodes())
    new_nodes = old_nodes.copy()

    rng.shuffle(new_nodes)

    mapping = dict(
        zip(old_nodes, new_nodes)
    )

    return (
        nx.relabel_nodes(
            graph,
            mapping,
            copy=True,
        ),
        mapping,
    )


def make_non_isomorphic_graph(
    graph: GraphLike,
    rng: random.Random,
):
    """
    Change the topology while preserving node IDs and attributes.
    """
    candidate = graph.copy()
    nodes = list(candidate.nodes())

    if len(nodes) < 2:
        return candidate, False

    for _ in range(100):
        candidate = graph.copy()

        operation = rng.choice(
            (
                "add_edge",
                "remove_edge",
                "rewire_edge",
            )
        )

        if operation == "add_edge":
            possible_edges = [
                (source, target)
                for index, source in enumerate(nodes)
                for target in nodes[index + 1:]
                if not candidate.has_edge(
                    source,
                    target,
                )
            ]

            if possible_edges:
                source, target = rng.choice(
                    possible_edges
                )

                candidate.add_edge(
                    source,
                    target,
                )

        elif operation == "remove_edge":
            if candidate.number_of_edges() > 1:
                edge = rng.choice(
                    list(candidate.edges())
                )

                candidate.remove_edge(*edge)

        elif operation == "rewire_edge":
            if candidate.number_of_edges() > 0:
                old_source, old_target = rng.choice(
                    list(candidate.edges())
                )

                candidate.remove_edge(
                    old_source,
                    old_target,
                )

                possible_edges = [
                    (source, target)
                    for index, source in enumerate(nodes)
                    for target in nodes[index + 1:]
                    if not candidate.has_edge(
                        source,
                        target,
                    )
                ]

                if possible_edges:
                    new_source, new_target = rng.choice(
                        possible_edges
                    )

                    candidate.add_edge(
                        new_source,
                        new_target,
                    )

        if not nx.is_isomorphic(
            graph,
            candidate,
        ):
            return candidate, True

    return graph.copy(), False


def create_dataset_record(
    graph: GraphLike,
    rng: random.Random,
    non_isomorphic_fraction: float,
) -> dict:
    """
    Create the exact original record format.
    """
    if (
        rng.random()
        < non_isomorphic_fraction
    ):
        graph_b, changed = (
            make_non_isomorphic_graph(
                graph,
                rng,
            )
        )

        return {
            "isomorphic": not changed,
            "graph_a": generator_format(
                graph
            ),
            "graph_b": generator_format(
                graph_b
            ),
            "mapping": None,
        }

    graph_b, mapping = (
        relabel_graph_randomly(
            graph,
            rng,
        )
    )

    return {
        "isomorphic": True,
        "graph_a": generator_format(
            graph
        ),
        "graph_b": generator_format(
            graph_b
        ),
        "mapping": {
            str(source): target
            for source, target in mapping.items()
        },
    }


def encode_record(record: dict) -> bytes:
    """
    Compact newline-delimited JSON.
    """
    return (
        json.dumps(
            record,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def generate_dataset(
    source_path: str,
    output_path: str,
    target_bytes: int,
    seed: int = 12345,
    non_isomorphic_fraction: float = 0.5,
    directed: bool = False,
):
    """
    Load source graphs and repeatedly generate records until the target size
    is reached.
    """
    rng = random.Random(seed)
    source_graphs = []

    with tqdm(
        desc="Loading source JSON",
        unit="graph",
        dynamic_ncols=True,
    ) as progress:
        for obj in iter_json_objects(
            source_path
        ):
            graph = graph_from_json(
                obj,
                directed=directed,
            )

            source_graphs.append(graph)
            progress.update(1)

    if not source_graphs:
        raise ValueError(
            "No source graphs were found."
        )

    output = Path(output_path)
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_bytes = 0
    record_count = 0
    source_index = 0

    with output.open("wb") as output_file:
        with tqdm(
            total=target_bytes,
            desc="Generating dataset",
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            dynamic_ncols=True,
        ) as progress:

            while total_bytes < target_bytes:
                source_graph = source_graphs[
                    source_index
                    % len(source_graphs)
                ]

                source_index += 1

                record = create_dataset_record(
                    source_graph,
                    rng,
                    non_isomorphic_fraction,
                )

                encoded = encode_record(
                    record
                )

                output_file.write(encoded)

                record_size = len(encoded)
                total_bytes += record_size
                record_count += 1

                progress.update(record_size)
                progress.set_postfix(
                    records=f"{record_count:,}",
                    refresh=False,
                )

    metadata = {
        "source": source_path,
        "output": output_path,
        "target_bytes": target_bytes,
        "actual_bytes": total_bytes,
        "actual_megabytes": (
            total_bytes / 1_000_000
        ),
        "actual_mebibytes": (
            total_bytes / (1024**2)
        ),
        "source_graphs": len(source_graphs),
        "records": record_count,
        "seed": seed,
        "non_isomorphic_fraction": (
            non_isomorphic_fraction
        ),
        "directed": directed,
        "format": "newline-delimited JSON",
    }

    metadata_path = Path(
        output_path
        + ".metadata.json"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        json.dumps(
            metadata,
            indent=2,
        )
    )


# ============================================================================
# DATASET VERIFICATION USING compare_graphs()
# ============================================================================

def graph_from_generator_record(
    graph_record: dict,
    directed: bool = False,
) -> GraphLike:
    return graph_from_json(
        graph_record,
        directed=directed,
    )


def verify_dataset(
    dataset_path: str,
    records_to_check: int = 100,
    directed: bool = False,
    node_attributes: Optional[Tuple[str, ...]] = None,
    edge_attributes: Optional[Tuple[str, ...]] = None,
):
    """
    Verify generated labels using the supplied exact comparison engine.
    """
    checked = 0
    failures = 0

    with open(
        dataset_path,
        "r",
        encoding="utf-8",
    ) as file:
        with tqdm(
            total=records_to_check,
            desc="Verifying dataset",
            unit="record",
            dynamic_ncols=True,
        ) as progress:

            for line in file:
                if checked >= records_to_check:
                    break

                line = line.strip()

                if not line:
                    continue

                record = json.loads(line)

                graph_a = (
                    graph_from_generator_record(
                        record["graph_a"],
                        directed=directed,
                    )
                )

                graph_b = (
                    graph_from_generator_record(
                        record["graph_b"],
                        directed=directed,
                    )
                )

                result = compare_graphs(
                    graph_a,
                    graph_b,
                    node_attributes=node_attributes,
                    edge_attributes=edge_attributes,
                    use_prefilters=True,
                    strict_prefilters=False,
                )

                if result.isomorphic != record[
                    "isomorphic"
                ]:
                    failures += 1

                checked += 1
                progress.update(1)
                progress.set_postfix(
                    failures=failures,
                    refresh=False,
                )

    print(
        f"Checked: {checked:,}; "
        f"failures: {failures:,}"
    )


# ============================================================================
# COMMAND-LINE INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a graph-isomorphism dataset "
            "from JSON using exact VF2 comparison."
        )
    )

    parser.add_argument(
        "source",
        help=(
            "Source JSON, JSONL, or JSON-array "
            "graph file."
        ),
    )

    parser.add_argument(
        "--output",
        default="social_isomorphism_1mb.jsonl",
        help="Output JSONL file.",
    )

    parser.add_argument(
        "--size",
        default="1MB",
        help="Examples: 500MB, 500MiB, 1GB.",
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
        help=(
            "Fraction of generated records "
            "that should be non-isomorphic."
        ),
    )

    parser.add_argument(
        "--directed",
        action="store_true",
        help="Treat source graphs as directed.",
    )

    parser.add_argument(
        "--verify",
        type=int,
        default=0,
        help=(
            "Verify this many generated records "
            "using compare_graphs()."
        ),
    )

    parser.add_argument(
        "--node-attributes",
        nargs="*",
        default=None,
        help=(
            "Node attributes that must match. "
            "Example: --node-attributes role department"
        ),
    )

    parser.add_argument(
        "--edge-attributes",
        nargs="*",
        default=None,
        help=(
            "Edge attributes that must match. "
            "Example: --edge-attributes relationship"
        ),
    )

    args = parser.parse_args()

    node_attributes = (
        tuple(args.node_attributes)
        if args.node_attributes
        else None
    )

    edge_attributes = (
        tuple(args.edge_attributes)
        if args.edge_attributes
        else None
    )

    generate_dataset(
        source_path=args.source,
        output_path=args.output,
        target_bytes=parse_size(args.size),
        seed=args.seed,
        non_isomorphic_fraction=(
            args.non_isomorphic_fraction
        ),
        directed=args.directed,
    )

    if args.verify > 0:
        verify_dataset(
            dataset_path=args.output,
            records_to_check=args.verify,
            directed=args.directed,
            node_attributes=node_attributes,
            edge_attributes=edge_attributes,
        )


if __name__ == "__main__":
    main()