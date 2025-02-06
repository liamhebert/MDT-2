"""Utilities to generate mask mods for flex attention that operate on graphs."""

from typing import Callable

import torch
from torch import Tensor
from torch.nn.attention.flex_attention import _mask_mod_signature
from torch.nn.attention.flex_attention import (
    BlockMask,
    create_block_mask,
    _DEFAULT_SPARSE_BLOCK_SIZE,
)

PADDING_GRAPH_ID = -1

# Signature for mask mod that operates on individual graphs.
# Batch size, num_heads, q_idx, kv_idx, distance
logical_graph_mask_mod_signature = Callable[
    [Tensor, Tensor, Tensor, Tensor, Tensor | None], Tensor
]


def generate_graph_attn_mask_mod(
    graph_ids: torch.Tensor,
    spatial_distance_matrix: torch.Tensor,
    max_spatial_distance: int = 8,
    num_heads: int = 1,
    block_size: int = _DEFAULT_SPARSE_BLOCK_SIZE,
) -> BlockMask:
    """Shortcut to generate a spatial mask mod for flex attention

    See: generate_attn_mask_mod and create_spatial_distance_mask_mod
    """
    num_nodes = graph_ids.shape[0]
    query_len, key_value_len = spatial_distance_matrix.shape[:2]
    assert query_len == key_value_len == num_nodes, (
        "Expecting square spatial distance matrix, got:",
        f"{query_len=}, {key_value_len=}, {num_nodes=}",
    )

    mask_mod = generate_attn_mask_mod(
        create_spatial_distance_mask_mod(max_spatial_distance),
        graph_ids,
        spatial_distance_matrix,
    )

    return create_block_mask(
        mask_mod=mask_mod,
        B=num_nodes,
        H=num_heads,
        Q_LEN=query_len,
        KV_LEN=key_value_len,
        device="cuda" if spatial_distance_matrix.is_cuda else "cpu",
        BLOCK_SIZE=block_size,
    )


def generate_attn_mask_mod(
    inner_mask_mod: logical_graph_mask_mod_signature,
    document_ids: torch.Tensor,
    spatial_distance_matrix: torch.Tensor | None = None,
) -> _mask_mod_signature:
    """Generates mask mods that apply to inputs to flex attention in the
    sequence stacked format.

    Args:
        inner_mask_mod: The mask mod to apply to individual graphs.
        document_ids: Id of the document each token belongs to, where padding
            tokens are assigned the value PADDING_GRAPH_ID.
        spatial_distance_matrix: A (N, N, 2) matrix of consisting of the
            number of up hops and down hops between each node in the graph.

    Note:
        What is the sequence stacked format? When assembling batches of inputs,
        we take multiple sequences and stack them together to form 1 large
        sequence. We then use masking to ensure that the attention scores are
        only applied to tokens within the same document.

    Example:

    - Square mask
      doc_mask         document_ids
      a a b b b P P    a a b b b P P
    a 1 1 0 0 0 0 0
    a 1 1 0 0 0 0 0
    b 0 0 1 1 1 0 0
    b 0 0 1 1 1 0 0
    b 0 0 1 1 1 0 0
    P 0 0 0 0 0 0 0
    P 0 0 0 0 0 0 0

    """
    # Get unique document IDs and their counts
    _, counts = torch.unique_consecutive(document_ids, return_counts=True)

    # Create cumulative counts (offsets)
    offsets = torch.cat(
        [torch.tensor([0], device=document_ids.device), counts.cumsum(0)[:-1]]
    )

    def doc_mask_wrapper(
        batch_size: Tensor, num_heads: Tensor, q_idx: Tensor, kv_idx: Tensor
    ) -> Tensor:
        """The actual mask mod function that will be used in the flex attention.

        Args:
            batch_size (Tensor): The total batch size of the input
            num_heads (Tensor): The number of heads in the attention layer
            q_idx (Tensor): The query index within the batch
            kv_idx (Tensor): The key-value index within the batch

        Returns:
            Tensor: A boolean tensor indicating whether there should be
                attention between the query and key-value pair.
        """
        if not num_heads.is_cuda:
            # This is a stupid workaround to get this to run
            # in eager mode and not crash. This is not a good solution.
            # print("q", q_idx)
            # print("kv", kv_idx)
            ...

        doc_q = document_ids[q_idx]
        doc_kv = document_ids[kv_idx]

        same_doc = doc_q == doc_kv
        valid_doc = (doc_q != PADDING_GRAPH_ID) & (doc_kv != PADDING_GRAPH_ID)

        q_logical = q_idx - offsets[doc_q]
        kv_logical = kv_idx - offsets[doc_kv]
        distance = (
            spatial_distance_matrix[q_idx, kv_idx]
            if spatial_distance_matrix is not None
            else None
        )
        inner_mask = inner_mask_mod(
            batch_size, num_heads, q_logical, kv_logical, distance
        )

        return same_doc & inner_mask & valid_doc
        # return same_doc & valid_doc

    return doc_mask_wrapper


# Now we need to create a mask for distances within a graph.


def create_spatial_distance_mask_mod(
    max_spatial_distance: int,
) -> logical_graph_mask_mod_signature:
    """Mask mod that masks out attention scores between nodes that are too far
    apart.

    Args:
        max_spatial_distance (int): Max distance between nodes to allow for
            attention. Combines up and down hops.

    Returns:
        graph_mask_mod_signature: Mask mod function for generate_graph_mask_mod.
    """

    def spatial_distance_mask_mod(
        batch_size: torch.Tensor,
        num_heads: torch.Tensor,
        q_idx: torch.Tensor,
        kv_idx: torch.Tensor,
        distance: torch.Tensor | None,
    ) -> Tensor:
        """Local mask mod function that masks out attention scores between
        nodes.

        Args:
            batch_size (torch.Tensor): Batch size of the input
            num_heads (torch.Tensor): Number of heads in the attention layer
            q_idx (torch.Tensor): Query index
            kv_idx (torch.Tensor): Key-value index
            distance (torch.Tensor): A 2 element tensor containing the number of
                up hops and down hops between the nodes.

        Returns:
            Tensor: Boolean tensor indicating whether the attention score should
                be masked out (False) or not (True).
        """
        if distance is None:
            # Full attention
            return torch.ones_like(q_idx, dtype=torch.bool)
        else:
            # We merge the up and down hops
            return torch.sum(distance) <= max_spatial_distance

    return spatial_distance_mask_mod
