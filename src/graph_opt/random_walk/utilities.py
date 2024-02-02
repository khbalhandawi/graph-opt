
# -*- coding: utf-8 -*-
#
# Adopted from stellargraph

import numpy as np
from typing import Union, Tuple, List, Any
from random import Random

from networkx import Graph, DiGraph, MultiGraph, MultiDiGraph

def edge_arrays(
        G:Union[Graph,DiGraph], include_edge_type=False, include_edge_weight=False
    ) -> List[np.ndarray]:
    """
    Obtains the collection of edges in the graph as a tuple of arrays (sources, targets, types, weights).
    ``types`` and ``weights`` will be `None` if the optional parameters are not specified.

    Args:
        include_edge_type (bool): A flag that indicates whether to return edge types.
        include_edge_weight (bool): A flag that indicates whether to return edge weights.
        use_ilocs (bool): if True return :ref:`ilocs for nodes (and edge types) <iloc-explanation>`

    Returns:
        A tuple containing 1D arrays of the source and target nodes (sources, targets, types, weights).
        Setting include_edge_type and/or include_edge_weight to True will include arrays of edge types
        and/or edge weights in this tuple, otherwise they will be set to ``None``.
    """

    edge_data = list(G.edges(data=True))

    types = np.empty(0)
    weights = np.empty(0)
    sources = np.empty(0)
    targets = np.empty(0)
    for edge_tuple in edge_data:
        types = np.append(types, edge_tuple[1] if (include_edge_type and isinstance(G,(MultiGraph,MultiDiGraph))) else None)
        weights = np.append(weights, edge_tuple[-1].get('weight',0.0) if include_edge_weight 
                            else np.array(None,dtype=np.float64))
        sources = np.append(sources, edge_tuple[0])
        targets = np.append(targets, edge_tuple[1])

    return sources, targets, types, weights

def _transform_edges(other_node: np.ndarray, 
                     edges:Union[List[Tuple[int,int,dict]],List[Tuple[int,int,int,dict]]], 
                     include_edge_weight: bool, 
                     filter_edge_types: List[Any]) -> Union[np.ndarray,List[np.ndarray]]:
    
    types = np.array([e[-1].get('type',None) for e in edges])
    if include_edge_weight:
        weights = np.array([e[-1].get('weight',0.0) for e in edges])
    else:
        weights = None
        
    if filter_edge_types is not None:
        correct_type = np.isin(types, filter_edge_types)
        other_node = other_node[correct_type]
        if weights is not None:
            weights = weights[correct_type]

    if weights is not None:
        return other_node, weights

    return other_node

def neighbor_arrays(
        graph:Union[Graph,DiGraph], node: int, include_edge_weight=False, edge_types=None
    ):
        """
        Obtains the collection of neighbouring nodes connected to the given node
        as an array of node_ids. If `include_edge_weight` edge is `True` then
        an array of edges weights is also returned in a tuple of `(neighbor_ids, edge_weights)`.

        Args:
            node (any): The node in question.
            include_edge_weight (bool, default False): If True an array of edge weights is also returned.
            edge_types (list of hashable, optional): If provided, only traverse the graph
                via the provided edge types when collecting neighbours.
            use_ilocs (bool): if True `node` is treated as a :ref:`node iloc <iloc-explanation>`
                (and similarly `edge_types` is treated as a edge type ilocs) and the ilocs of each
                neighbour is returned.

        Returns:
            A numpy array of the neighboring nodes. If `include_edge_weight` is `True` then an array
            of edge weights is also returned in a tuple `(neighbor_array, edge_weight_array)`
        """
        if isinstance(graph,(DiGraph,MultiDiGraph)):
            edges = list(graph.out_edges(node,data=True))
        else:
            edges = list(graph.edges(node,data=True))
        source = np.array([e[0] for e in edges])
        target = np.array([e[1] for e in edges])
        other_node = np.where(source == node, target, source)

        return _transform_edges(
            other_node, edges, include_edge_weight, edge_types
        )

def naive_weighted_choices(rs:Random, weights:List[float], size:int=None):
    """
    Select indices at random, weighted by the iterator `weights` of
    arbitrary (non-negative) floats. That is, `x` will be returned
    with probability `weights[x]/sum(weights)`.

    For doing a single sample with arbitrary weights, this is much (5x
    or more) faster than numpy.random.choice, because the latter
    requires a lot of preprocessing (normalized probabilties), and
    does a lot of conversions/checks/preprocessing internally.
    """
    probs = np.cumsum(weights)
    total = probs[-1]
    if total == 0:
        # all weights were zero (probably), so we shouldn't choose anything
        return None

    thresholds = rs.random() if size is None else rs.random(size)
    idx = np.searchsorted(probs, thresholds * total, side="left")

    return idx
