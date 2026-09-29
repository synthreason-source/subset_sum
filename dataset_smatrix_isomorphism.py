#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Hashable, Optional, Tuple

import hashlib
import numpy as np
import networkx as nx

from networkx.algorithms import isomorphism as iso
from networkx.algorithms.graph_hashing import weisfeiler_lehman_graph_hash


Node = Hashable
GraphLike = nx.Graph


@dataclass
class IsomorphismResult:
    """
    Result of an isomorphism comparison.

    isomorphic:
        True only when an exact structure-preserving mapping was found.

    mapping:
        Dictionary mapping nodes from G1 to nodes in G2.

    rejected_by:
        Name of a cheap invariant that proved the graphs non-isomorphic,
        or None when exact matching was required.

    wl_hash_equal:
        Whether the Weisfeiler-Lehman hashes matched.

    spectral_distance:
        Distance between normalized adjacency spectra.

    similarity:
        A heuristic structural similarity score in [0, 1].
    """

    isomorphic: bool
    mapping: Optional[Dict[Node, Node]]
    rejected_by: Optional[str]
    wl_hash_equal: bool
    spectral_distance: float
    similarity: float


def _graph_kind(G: GraphLike) -> Tuple[bool, bool]:
    """
    Returns:

        directed:
            Whether G is directed.

        multigraph:
            Whether G supports parallel edges.
    """
    return G.is_directed(), G.is_multigraph()


def _node_degree_signature(G: GraphLike):
    """
    Degree signature suitable for directed, undirected, and multigraph graphs.
    """
    if G.is_directed():
        values = [
            (
                G.in_degree(node),
                G.out_degree(node),
            )
            for node in G.nodes()
        ]
    else:
        values = [G.degree(node) for node in G.nodes()]

    return tuple(sorted(values))


def _edge_count_signature(G: GraphLike):
    """
    Returns a sorted signature of edge multiplicities.

    For simple graphs, every multiplicity is one.
    """
    if not G.is_multigraph():
        return tuple([1] * G.number_of_edges())

    multiplicities = []
    for u, v in G.edges():
        multiplicities.append(G.number_of_edges(u, v))

    return tuple(sorted(multiplicities))


def _component_signature(G: GraphLike):
    """
    Connected-component signature.

    For directed graphs, weak connectivity is used because social-network
    structure is usually interpreted through the underlying connectivity.
    """
    if G.is_directed():
        components = nx.weakly_connected_components(G)
    else:
        components = nx.connected_components(G)

    return tuple(
        sorted(
            (
                len(component),
                sum(
                    1
                    for u, v in G.edges()
                    if u in component and v in component
                ),
            )
            for component in components
        )
    )


def _triangle_signature(G: GraphLike):
    """
    Sorted local triangle counts.

    Triangle counts are used only as a prefilter. They are not sufficient
    to establish isomorphism.
    """
    H = nx.Graph(G)

    if H.number_of_nodes() == 0:
        return ()

    triangles = nx.triangles(H)
    return tuple(sorted(triangles.values()))


def _adjacency_spectrum(G: GraphLike):
    """
    Returns sorted adjacency eigenvalues.

    For directed graphs, the underlying undirected graph is used. This makes
    the spectrum a cheap coarse invariant rather than a directional proof.
    """
    H = nx.Graph(G)

    if H.number_of_nodes() == 0:
        return np.empty(0, dtype=np.float64)

    A = nx.to_numpy_array(
        H,
        nodelist=list(H.nodes()),
        dtype=np.float64,
    )

    eigenvalues = np.linalg.eigvalsh(A)
    return np.sort(eigenvalues)


def _normalized_spectral_distance(G1: GraphLike, G2: GraphLike):
    """
    Computes a scale-normalized adjacency spectral distance.
    """
    s1 = _adjacency_spectrum(G1)
    s2 = _adjacency_spectrum(G2)

    if len(s1) != len(s2):
        return float("inf")

    norm = max(
        np.linalg.norm(s1),
        np.linalg.norm(s2),
        1.0,
    )

    return float(np.linalg.norm(s1 - s2) / norm)


def _stable_attribute_token(value: Any) -> str:
    """
    Converts arbitrary common Python attribute values into deterministic text.
    """
    if isinstance(value, dict):
        items = sorted(
            (
                str(key),
                _stable_attribute_token(item),
            )
            for key, item in value.items()
        )
        return "{" + ",".join(f"{k}:{v}" for k, v in items) + "}"

    if isinstance(value, (list, tuple)):
        return "[" + ",".join(
            _stable_attribute_token(item)
            for item in value
        ) + "]"

    if isinstance(value, set):
        return "{" + ",".join(
            sorted(_stable_attribute_token(item) for item in value)
        ) + "}"

    return repr(value)


def _attribute_label(
    attributes: Dict[str, Any],
    keys: Optional[Tuple[str, ...]],
) -> str:
    """
    Converts selected attributes into a categorical label.

    If keys is None, all attributes are used.
    """
    if keys is None:
        selected = attributes
    else:
        selected = {
            key: attributes.get(key)
            for key in keys
        }

    return _stable_attribute_token(selected)


def _label_graph(
    G: GraphLike,
    node_attributes: Optional[Tuple[str, ...]],
    edge_attributes: Optional[Tuple[str, ...]],
):
    """
    Creates a shallow copy with canonical string labels for WL hashing.
    """
    H = G.copy()

    for node, attributes in H.nodes(data=True):
        attributes["_wl_node_label"] = _attribute_label(
            attributes,
            node_attributes,
        )

    if H.is_multigraph():
        for u, v, key, attributes in H.edges(
            keys=True,
            data=True,
        ):
            attributes["_wl_edge_label"] = _attribute_label(
                attributes,
                edge_attributes,
            )
    else:
        for u, v, attributes in H.edges(data=True):
            attributes["_wl_edge_label"] = _attribute_label(
                attributes,
                edge_attributes,
            )

    return H


def _wl_hash(
    G: GraphLike,
    node_attributes: Optional[Tuple[str, ...]],
    edge_attributes: Optional[Tuple[str, ...]],
    iterations: int,
):
    """
    Weisfeiler-Lehman graph hash with optional node and edge labels.
    """
    H = _label_graph(
        G,
        node_attributes=node_attributes,
        edge_attributes=edge_attributes,
    )

    return weisfeiler_lehman_graph_hash(
        H,
        node_attr="_wl_node_label",
        edge_attr="_wl_edge_label",
        iterations=iterations,
    )


def compute_graph_s_matrix(
    G: GraphLike,
    max_hops: int = 4,
):
    """
    Computes a multi-scale structural feature Gram matrix.

    This is retained as a heuristic diagnostic. It is not used as the final
    isomorphism decision.
    """
    n = G.number_of_nodes()

    if n == 0:
        return np.zeros((1, 1), dtype=np.float64)

    H = nx.Graph(G)
    nodes = list(H.nodes())

    A = nx.to_numpy_array(
        H,
        nodelist=nodes,
        dtype=np.float64,
    )

    degrees = np.sum(A, axis=1, keepdims=True)
    safe_degrees = degrees.copy()
    safe_degrees[safe_degrees == 0.0] = 1.0

    P = A / safe_degrees

    features = [
        degrees,
        np.diag(A).reshape(-1, 1),
    ]

    # One feature for every walk length from 2 through max_hops.
    current_P = P.copy()

    for _ in range(2, max_hops + 1):
        features.append(
            np.sum(current_P, axis=1, keepdims=True)
        )
        current_P = current_P @ P

    X = np.hstack(features)

    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    std[std < 1e-12] = 1.0

    X_norm = (X - mean) / std
    return X_norm.T @ X_norm


def s_matrix_spectral_distance(
    G1: GraphLike,
    G2: GraphLike,
    max_hops: int = 4,
):
    """
    Compares the singular spectra of the S-matrices.

    This is a heuristic similarity measure and must not be treated as a
    complete isomorphism test.
    """
    S1 = compute_graph_s_matrix(G1, max_hops=max_hops)
    S2 = compute_graph_s_matrix(G2, max_hops=max_hops)

    _, sv1, _ = np.linalg.svd(S1, full_matrices=False)
    _, sv2, _ = np.linalg.svd(S2, full_matrices=False)

    norm1 = np.linalg.norm(sv1)
    norm2 = np.linalg.norm(sv2)

    sv1 = sv1 / max(norm1, 1e-12)
    sv2 = sv2 / max(norm2, 1e-12)

    return float(np.linalg.norm(sv1 - sv2))


def _numeric_similarity(
    G1: GraphLike,
    G2: GraphLike,
    wl_hash_equal: bool,
    spectral_distance: float,
):
    """
    Produces a heuristic score for ranking candidate graph pairs.

    The score is not an isomorphism certificate.
    """
    if G1.number_of_nodes() != G2.number_of_nodes():
        return 0.0

    if G1.number_of_edges() != G2.number_of_edges():
        return 0.0

    degree_equal = (
        _node_degree_signature(G1)
        == _node_degree_signature(G2)
    )

    triangle_equal = (
        _triangle_signature(G1)
        == _triangle_signature(G2)
    )

    component_equal = (
        _component_signature(G1)
        == _component_signature(G2)
    )

    score = 0.0
    score += 0.35 if degree_equal else 0.0
    score += 0.25 if triangle_equal else 0.0
    score += 0.20 if component_equal else 0.0
    score += 0.15 if wl_hash_equal else 0.0
    score += 0.05 * np.exp(-spectral_distance)

    return float(min(score, 1.0))


def _build_node_match(
    node_attributes: Optional[Tuple[str, ...]],
):
    """
    Builds exact categorical node matching.

    None means node attributes are ignored.
    """
    if node_attributes is None:
        return None

    def node_match(attributes_1, attributes_2):
        return all(
            attributes_1.get(key) == attributes_2.get(key)
            for key in node_attributes
        )

    return node_match


def _build_edge_match(
    edge_attributes: Optional[Tuple[str, ...]],
):
    """
    Builds exact categorical edge matching.

    None means edge attributes are ignored.
    """
    if edge_attributes is None:
        return None

    def edge_match(attributes_1, attributes_2):
        return all(
            attributes_1.get(key) == attributes_2.get(key)
            for key in edge_attributes
        )

    return edge_match


def _exact_mapping(
    G1: GraphLike,
    G2: GraphLike,
    node_attributes: Optional[Tuple[str, ...]],
    edge_attributes: Optional[Tuple[str, ...]],
):
    """
    Executes exact VF2 isomorphism matching.
    """
    if G1.is_directed() != G2.is_directed():
        return None

    if G1.is_multigraph() != G2.is_multigraph():
        return None

    node_match = _build_node_match(node_attributes)
    edge_match = _build_edge_match(edge_attributes)

    if G1.is_directed():
        if G1.is_multigraph():
            matcher = iso.MultiDiGraphMatcher
        else:
            matcher = iso.DiGraphMatcher
    else:
        if G1.is_multigraph():
            matcher = iso.MultiGraphMatcher
        else:
            matcher = iso.GraphMatcher

    graph_matcher = matcher(
        G1,
        G2,
        node_match=node_match,
        edge_match=edge_match,
    )

    if not graph_matcher.is_isomorphic():
        return None

    return dict(graph_matcher.mapping)


def compare_graphs(
    G1: GraphLike,
    G2: GraphLike,
    *,
    node_attributes: Optional[Tuple[str, ...]] = None,
    edge_attributes: Optional[Tuple[str, ...]] = None,
    wl_iterations: int = 3,
    use_prefilters: bool = True,
    spectral_tolerance: float = 1e-8,
):
    """
    Compare two social-network graphs.

    Parameters
    ----------
    node_attributes:
        Attributes that must be preserved by an isomorphism.

        Example:
            ("role", "department")

        Use None to ignore node attributes.

    edge_attributes:
        Attributes that must be preserved by an isomorphism.

        Example:
            ("relationship",)

        Use None to ignore edge attributes.

    wl_iterations:
        Number of WL refinement rounds used by the hash prefilter.

    use_prefilters:
        If True, cheap invariants reject obviously different graphs before
        running exact VF2.

    spectral_tolerance:
        Spectral mismatches larger than this reject the graphs before VF2.
        Set to None to disable this prefilter.

    Returns
    -------
    IsomorphismResult
    """
    if not isinstance(G1, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
        raise TypeError("G1 must be a NetworkX graph")

    if not isinstance(G2, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
        raise TypeError("G2 must be a NetworkX graph")

    directed_1, multigraph_1 = _graph_kind(G1)
    directed_2, multigraph_2 = _graph_kind(G2)

    if directed_1 != directed_2:
        return IsomorphismResult(
            isomorphic=False,
            mapping=None,
            rejected_by="directedness",
            wl_hash_equal=False,
            spectral_distance=float("inf"),
            similarity=0.0,
        )

    if multigraph_1 != multigraph_2:
        return IsomorphismResult(
            isomorphic=False,
            mapping=None,
            rejected_by="multigraph_type",
            wl_hash_equal=False,
            spectral_distance=float("inf"),
            similarity=0.0,
        )

    if G1.number_of_nodes() != G2.number_of_nodes():
        return IsomorphismResult(
            isomorphic=False,
            mapping=None,
            rejected_by="node_count",
            wl_hash_equal=False,
            spectral_distance=float("inf"),
            similarity=0.0,
        )

    if G1.number_of_edges() != G2.number_of_edges():
        return IsomorphismResult(
            isomorphic=False,
            mapping=None,
            rejected_by="edge_count",
            wl_hash_equal=False,
            spectral_distance=float("inf"),
            similarity=0.0,
        )

    if _node_degree_signature(G1) != _node_degree_signature(G2):
        return IsomorphismResult(
            isomorphic=False,
            mapping=None,
            rejected_by="degree_signature",
            wl_hash_equal=False,
            spectral_distance=float("inf"),
            similarity=0.0,
        )

    if _edge_count_signature(G1) != _edge_count_signature(G2):
        return IsomorphismResult(
            isomorphic=False,
            mapping=None,
            rejected_by="edge_multiplicity",
            wl_hash_equal=False,
            spectral_distance=float("inf"),
            similarity=0.0,
        )

    if _component_signature(G1) != _component_signature(G2):
        return IsomorphismResult(
            isomorphic=False,
            mapping=None,
            rejected_by="component_signature",
            wl_hash_equal=False,
            spectral_distance=float("inf"),
            similarity=0.0,
        )

    if _triangle_signature(G1) != _triangle_signature(G2):
        return IsomorphismResult(
            isomorphic=False,
            mapping=None,
            rejected_by="triangle_signature",
            wl_hash_equal=False,
            spectral_distance=float("inf"),
            similarity=0.0,
        )

    hash_1 = _wl_hash(
        G1,
        node_attributes=node_attributes,
        edge_attributes=edge_attributes,
        iterations=wl_iterations,
    )

    hash_2 = _wl_hash(
        G2,
        node_attributes=node_attributes,
        edge_attributes=edge_attributes,
        iterations=wl_iterations,
    )

    wl_hash_equal = hash_1 == hash_2

    spectral_distance = _normalized_spectral_distance(G1, G2)

    similarity = _numeric_similarity(
        G1,
        G2,
        wl_hash_equal=wl_hash_equal,
        spectral_distance=spectral_distance,
    )

    if use_prefilters and not wl_hash_equal:
        return IsomorphismResult(
            isomorphic=False,
            mapping=None,
            rejected_by="weisfeiler_lehman_hash",
            wl_hash_equal=False,
            spectral_distance=spectral_distance,
            similarity=similarity,
        )

    if (
        use_prefilters
        and spectral_tolerance is not None
        and spectral_distance > spectral_tolerance
    ):
        return IsomorphismResult(
            isomorphic=False,
            mapping=None,
            rejected_by="adjacency_spectrum",
            wl_hash_equal=wl_hash_equal,
            spectral_distance=spectral_distance,
            similarity=similarity,
        )

    mapping = _exact_mapping(
        G1,
        G2,
        node_attributes=node_attributes,
        edge_attributes=edge_attributes,
    )

    return IsomorphismResult(
        isomorphic=mapping is not None,
        mapping=mapping,
        rejected_by=None if mapping is not None else "vf2_exact_match",
        wl_hash_equal=wl_hash_equal,
        spectral_distance=spectral_distance,
        similarity=similarity,
    )


def relabel_graph(G: GraphLike, mapping: Dict[Node, Node]):
    """
    Applies a node mapping to G.

    The returned graph has the same topology and attributes, but uses the
    target node labels from mapping.
    """
    return nx.relabel_nodes(G, mapping, copy=True)


def print_comparison(
    G1: GraphLike,
    G2: GraphLike,
    result: IsomorphismResult,
):
    """
    Human-readable report.
    """
    print(f"Isomorphic: {result.isomorphic}")
    print(f"Similarity: {result.similarity:.6f}")
    print(f"WL hashes equal: {result.wl_hash_equal}")
    print(f"Spectral distance: {result.spectral_distance:.6e}")

    if result.rejected_by is not None:
        print(f"Rejected by: {result.rejected_by}")

    if result.mapping is not None:
        print("Mapping G1 -> G2:")
        for source, target in result.mapping.items():
            print(f"  {source!r} -> {target!r}")


def make_social_graph():
    """
    Example labelled social network.

    Nodes represent people.
    Edges represent relationships.
    """
    G = nx.Graph()

    G.add_nodes_from(
        [
            ("alice", {"role": "admin"}),
            ("bob", {"role": "member"}),
            ("carol", {"role": "member"}),
            ("dave", {"role": "moderator"}),
            ("erin", {"role": "member"}),
        ]
    )

    G.add_edges_from(
        [
            ("alice", "bob", {"relationship": "friend"}),
            ("alice", "carol", {"relationship": "friend"}),
            ("bob", "dave", {"relationship": "follows"}),
            ("carol", "dave", {"relationship": "follows"}),
            ("dave", "erin", {"relationship": "friend"}),
        ]
    )

    return G


def permute_graph_labels(G: GraphLike):
    """
    Creates a structurally identical graph with unrelated node identifiers.
    """
    old_nodes = list(G.nodes())
    new_nodes = [f"user_{index}" for index in range(len(old_nodes))]

    permutation = dict(zip(old_nodes, new_nodes))
    return nx.relabel_nodes(G, permutation, copy=True)


def test_unlabelled_social_isomorphism():
    G1 = make_social_graph()
    G2 = permute_graph_labels(G1)

    result = compare_graphs(
        G1,
        G2,
        node_attributes=None,
        edge_attributes=None,
    )

    print("UNLABELLED STRUCTURAL TEST")
    print_comparison(G1, G2, result)
    print()


def test_labelled_social_isomorphism():
    G1 = make_social_graph()
    G2 = permute_graph_labels(G1)

    result = compare_graphs(
        G1,
        G2,
        node_attributes=("role",),
        edge_attributes=("relationship",),
    )

    print("LABELLED SOCIAL-NETWORK TEST")
    print_comparison(G1, G2, result)
    print()


def test_regular_graph_collision_case():
    """
    Demonstrates why S-matrix spectra alone are insufficient.

    Many regular graphs have identical local degree features. Exact VF2
    matching is still required.
    """
    G1 = nx.cycle_graph(6)
    G2 = nx.complete_bipartite_graph(3, 3)

    print("REGULAR-GRAPH TEST")
    print(f"G1 regular: {len(set(dict(G1.degree()).values())) == 1}")
    print(f"G2 regular: {len(set(dict(G2.degree()).values())) == 1}")

    result = compare_graphs(G1, G2)
    print_comparison(G1, G2, result)
    print()


def test_directed_social_network():
    """
    Directed example where edge orientation matters.
    """
    G1 = nx.DiGraph()
    G1.add_edges_from(
        [
            ("alice", "bob"),
            ("bob", "carol"),
            ("carol", "alice"),
        ]
    )

    G2 = nx.DiGraph()
    G2.add_edges_from(
        [
            (100, 200),
            (200, 300),
            (300, 100),
        ]
    )

    result = compare_graphs(G1, G2)
    print("DIRECTED SOCIAL-NETWORK TEST")
    print_comparison(G1, G2, result)
    print()


if __name__ == "__main__":
    test_unlabelled_social_isomorphism()
    test_labelled_social_isomorphism()
    test_regular_graph_collision_case()
    test_directed_social_network()