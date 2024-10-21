"""
General pre-processing utilities for individual graphs.
"""

import numpy as np
import pyximport
import torch
from torch_geometric.data import Data

pyximport.install(setup_args={"include_dirs": np.get_include()})

# TODO(liamhebert): Merge this class into an object so that we can make it
# configurable by flags


def cantor(x1: int, x2: int) -> int:
    """Maps a set of 2 numbers to a unique index with Cantor's pairing function.

    Notably, this function is a modification of Cantor's, which is normally
    position-dependent (ie: [1, 2] != [2, 1]). We instead make it synmetric, to
    allow for reusable embeddings (bi-directional)

    We use these numbers to index into an embedding table.

    Args:
        x1 (int): The first number to map
        x2 (int): The second number to map

    Returns:
        int: A unique index mapping the two numbers.
    """
    values = sorted([x1, x2])
    return int(
        ((values[0] + values[1]) * (values[0] + values[1] + 1)) / 2 + values[0]
    )


def preprocess_item(item: Data) -> Data:
    """Additional pre-processing for a single graph before passing to the model.

    This includes:
    - Computing the in-degree and out-degree of each node
    - Computing the distance function between each node
    - Converting the edge_index property into an adjacency matrix

    Args:
        item (Data): Processed graph object.

    Returns:
        Data: A new graph object with the following new properties
        - attn_bias: All zeroes
        - in_degree: The in-degree of each node
        - out_degree: The out-degree of each node
        - distance: The distance between each node
        - spatial_pos: The spatial position of each node, as indexed by Cantor's
            pairing function
    """

    # TODO(liamhebert): Move this to dataset

    edge_attr, edge_index, x = item.edge_attr, item.edge_index, item.x

    N = x["input_ids"].size(0)

    adj = torch.zeros([N, N], dtype=torch.bool)
    adj[edge_index[0, :], edge_index[1, :]] = True

    res = []
    for i in range(6):
        for k in range(6):
            res += [cantor(i, k)]

    res = list(set(res))
    mapping = {val: i for i, val in enumerate(res)}

    # wtf is happening here.
    spatial = list(
        map(
            lambda x: list(
                map(
                    lambda up_index, down_index: (
                        mapping[cantor(up_index, down_index)]
                        if cantor(up_index, down_index) in mapping
                        else mapping[cantor(5, 5)]
                    ),
                    x,
                )
            ),
            item.distance_matrix,
        )
    )
    distance = list(
        map(lambda x: list(map(lambda k: sum(k), x)), item.distance_matrix)
    )
    attn_bias = torch.zeros(
        [N + 1, N + 1], dtype=torch.float
    )  # with graph token

    # combine
    item.attn_bias = attn_bias
    item.spatial_pos = torch.Tensor(spatial)
    item.distance = torch.Tensor(distance)
    item.in_degree = adj.long().sum(dim=1).view(-1)
    item.out_degree = item.in_degree  # for undirected graph

    return item
