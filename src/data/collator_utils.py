"""
Collator functions to merge data samples into a batch.
"""

from typing import List

import torch
from torch_geometric.data import Data
from transformers import BatchEncoding

from data.types import GraphFeatures
from data.types import ImageFeatures
from data.types import TextFeatures
from utils.pylogger import RankedLogger

log = RankedLogger(__name__)


def pad_1d_unsqueeze(
    x: torch.Tensor, padlen: int, pad_value: int = 0, shift: bool = True
) -> torch.Tensor:
    """Pads a 1D tensor to padlen with pad_value and unsqueezes it.

    If shift, nudges all non-pad values + 1.

    This is useful to collate a collection of tensors into a batch. Shift is
    useful if the indices are used for indexing into an embedding table, where
    0 is the padding index.

    Args:
        x (torch.Tensor): The tensor to pad, with shape (seq)
        padlen (int): Size to pad to
        pad_value (int, optional): Value to pad with. Defaults to 0.
        shift (bool, optional): Whether to shift values by 1, useful when value
            is used to index into an embedding table. Defaults to True.

    Returns:
        torch.Tensor: Returns x padded to padlen with pad_value, optionally
        shifted by 1, with shape (1, seq).
    """
    assert len(x.shape) == 1

    if (
        pad_value == 0 and shift
    ):  # to avoid existing 0 values being treated as padding
        x = x + 1
    xlen = x.size(0)
    if xlen < padlen:
        new_x = torch.full([padlen], fill_value=pad_value, dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_2d_unsqueeze(
    x: torch.Tensor, padlen: int, pad_value: int = 0, shift: bool = True
) -> torch.Tensor:
    """Pads a 2D tensor to padlen with pad_value and unsqueezes it.

    That is, if padlen is 4, pad_value is 0, and x is
    [[1, 1],
     [1, 1]] (xlen = 2)
    then the output will be
    [[1, 1],
     [1, 1],
     [0, 0],
     [0, 0]]

    NOTE: This function assumes that all x have the same dimension, no matter
    the batch size. This makes it suitable for features that have deterministic
    padding, like text features!

    Args:
        x (torch.Tensor): The tensor to pad, with shape (seq)
        padlen (int): Size to pad to
        pad_value (int, optional): Value to pad with. Defaults to 0.
        shift (bool, optional): Whether to shift values by 1, useful when value
            is used to index into an embedding table. Defaults to True.

    Returns:
        torch.Tensor: Tensor padded to (1, padlen, xdim) with pad_value.
    """
    if (
        pad_value == 0 and shift
    ):  # to avoid existing 0 values being treated as padding
        x = x + 1
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = torch.full([padlen, xdim], fill_value=pad_value, dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_attn_bias_unsqueeze(x: torch.Tensor, padlen: int):
    """Variant of pad_2d_unsqueeze geared towards attention bias tensors.

    In contrast with pad_2d_unsqueeze, we aim to create square tensors. To do
    this, we pad each row to padlen with -inf, and fill up to xlen with 0s.

    For instance, if padlen is 4 and x is
    [[1, 1],
     [1, 1]] (xlen = 2)
    then the output will be
    [[1, 1, -inf, -inf],
     [1, 1, -inf, -inf],
     [0, 0, -inf, -inf],
     [0, 0, -inf, -inf]]

    Args:
        x (torch.Tensor): The tensor to pad, with shape (xlen, xlen)
        padlen (int): Size to pad to

    Returns:
        torch.Tensor: Tensor padded to (1, padlen, padlen) with -inf and 0s as
            described above.
    """
    xlen = x.size(0)
    if xlen < padlen:
        new_x = torch.full(
            [padlen, padlen], fill_value=-torch.inf, dtype=x.dtype
        )
        new_x[:xlen, :xlen] = x
        new_x[xlen:, :xlen] = 0
        x = new_x
    return x.unsqueeze(0)


def pad_edge_type_unsqueeze(x: torch.Tensor, padlen: int) -> torch.Tensor:
    """Pads a 3D tensor to padlen with 0s and unsqueezes it.

    This is used for edge_type features, which we currently do not use. These
    features are in shape of [N, N, edge_feature_dim], where N is the number of
    nodes in the graph. That is, x[i, j] is the feature between node i and node
    j.

    Similar to pad_2d_unsqueeze, this function pads the tensor to padlen, and
    assumes that all x have the same feature dimension, no matter the batch
    size.

    Args:
        x (torch.Tensor): The tensor to pad, with shape (N, N, xdim)
        padlen (int): Size to pad to

    Returns:
        torch.Tensor: Tensor padded to (1, padlen, padlen, xdim) with 0s.
    """
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen, x.size(-1)], dtype=x.dtype)
        new_x[:xlen, :xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_spatial_pos_unsqueeze(x: torch.Tensor, padlen: int):
    """Similar to pad_attn_bias_unsqueeze, but for spatial position tensors.

    Here, we pad the tensor to be a square tensor of size padlen, but instead
    of partially filling to -inf, we fill with 0s. We also shift the values by
    1. This is because the spatial position embeddings are 1-indexed.

    For instance, if padlen is 4 and x is
    [[1, 1],
     [1, 1]] (xlen = 2)
    then the output will be
    [[1, 1, 0, 0],
     [1, 1, 0, 0],
     [0, 0, 0, 0],
     [0, 0, 0, 0]]

    Args:
        x (torch.Tensor): The tensor to pad, with shape (xlen, xlen)
        padlen (int): The size to pad to

    Returns:
        torch.Tensor: Tensor padded to (1, padlen, padlen) with 0s and shifted
    """
    x = x + 1  # to avoid existing 0 values being treated as padding
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)


InputFeatures = dict[str, List[torch.Tensor]]


def generic_collator(
    graph_features: InputFeatures,
    text_features: InputFeatures,
    image_features: InputFeatures,
    spatial_pos_max: int = 10,
) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
    """Collate function to merge data samples of various sizes into a batch.

    Individual data samples are comprised of the following attributes:
    Graph Features:
        - edge_index (Tensor): The edge index of the graph, with shape
            [2, num_edges].
        - image_mask (Tensor): A mask indicating whether a node has an image or
            not, with shape [num_nodes].
        - distance (Tensor): The relative distance between each node in the
            graph, with shape [num_nodes, num_nodes, 2].
        - attn_bias (Tensor): The initial attention bias for the graph, with
            shape [num_nodes, num_nodes]. By default this is all zeros.
        - in_degree (Tensor): The in-degree of each node in the graph, with
            shape [num_nodes]. Since we treat the graph as bidirectional, this
            is the same as the out-degree.
        - out_degree (Tensor): The out-degree of each node in the graph, with
            shape [num_nodes]. Since we treat the graph as bidirectional, this
            is the same as the in-degree.
        - distance_index (Tensor): A flattened version of the distance tensor,
            mapping each distance to a unique index, with shape
            [num_nodes, num_nodes].
    Text Features:
        - input_ids (Tensor): The tokenized text of the graph as produced by the
            text_tokenizer, with shape [num_nodes, max_text_length].
        - attention_mask (Tensor): The attention mask of the text, with shape
            [num_nodes, max_text_length].
        - token_type_ids (Tensor): The token type ids of the text, with shape
            [num_nodes, max_text_length].
    Image Features:
        - pixel_values (Tensor): The pixel values of the images, with shape
            [num_images, ...].

    NOTE: This collator function *DOES NOT* handle y features. That is the
    responsibility of task specific collator functions.
    See ContrastiveTaskDataset and NodeBatchedDataset for examples.

    Args:
        graph_features: list of graph specific features making up a batch, with
            the attributes described above
        text_features: list of text specific features making up a batch, with
            the attributes described above
        image_features: list of image specific features making up a batch, with
            the attributes described above
        spatial_pos_max: maximum spatial position to attend to when computing
            attention. Any node farther away then this distance is masked.

    Returns:
        A collated patch of data samples where each item is padded to the
        largest size in the batch.

        Each output dictionary contains the following keys:
        - attn_bias (torch.Tensor): batched attention biases for each node.
            Note that we set the attn_bias for items with a distance larger
            than spatial_pos_max to -inf, effectively masking them out.
            Shape (batch_size, N, N).
        - spatial_pos (torch.Tensor): batched spatial positions,
            corresponding to indices to fetch spatial position embeddings.
            Padded with 0s and shifted by 1. Shape (batch_size, N).
        - in_degree (torch.Tensor): batched in-degrees, corresponding to the
            in-degree of each node in the graph. Padded with 0s and shifted
            by 1. Shape (batch_size, N).
        - out_degree (torch.Tensor): batched out-degrees, corresponding to
            the out-degree of each node in the graph. Padded with 0s and
            shifted by 1. Shape (batch_size, N).
        - node_mask (torch.Tensor): batched token mask, indicating which nodes
            are not padding. Shape (batch_size, N).
        - input_ids (torch.Tensor): batched tokenized text ids, with shape
            (batch_size, N, T)
        - token_type_ids (torch.Tensor): batched token type ids, with shape
            (batch_size, N, T)
        - attention_mask (torch.Tensor): batched text attention mask, with shape
            (batch_size, N, T), where 1 indicates a token that should be
            attended to and 0 indicates padding.
        - images (torch.Tensor): batched image features, with shape
            (batch_size, num_images, D)
        - image_indexes (torch.Tensor): batched boolean tensor indicating
            which nodes have images, where the order of True values corresponds
            to the order of images in the images tensor. Shape (batch_size, N).
    """

    attn_biases: list[torch.Tensor] = graph_features[GraphFeatures.AttnBias]
    in_degrees: list[torch.Tensor] = graph_features[GraphFeatures.InDegree]
    image_masks: list[torch.Tensor] = graph_features[GraphFeatures.ImageMask]
    distances: list[torch.Tensor] = graph_features[GraphFeatures.Distance]
    distance_indices: list[torch.Tensor] = graph_features[
        GraphFeatures.DistanceIndex
    ]

    # assert that each property is a list of torch tensors
    assert all(isinstance(i, torch.Tensor) for i in attn_biases)
    assert all(isinstance(i, torch.Tensor) for i in in_degrees)
    assert all(isinstance(i, torch.Tensor) for i in image_masks)
    assert all(isinstance(i, torch.Tensor) for i in distances)
    assert all(isinstance(i, torch.Tensor) for i in distance_indices)

    assert all(
        all(isinstance(i, torch.Tensor) for i in value)
        for value in text_features.values()
    )
    assert all(
        all(isinstance(i, torch.Tensor) for i in value)
        for value in image_features.values()
    )

    # Clip attention bias to -inf for nodes that are farther then spatial_pos_max
    # setting to -inf sets the attention value to 0, removing them from inference

    for idx, _ in enumerate(attn_biases):
        # TODO(liamhebert): Consider never masking direct parents, only children.
        attn_biases[idx][distances[idx].sum(dim=1) >= spatial_pos_max] = (
            -torch.inf
        )

    max_node_num = max(i.size(0) for i in in_degrees)

    collated_text_features: dict[str, torch.Tensor] = {}

    # As we try out newer models, we may need to change this as we verify them.
    known_keys = [
        TextFeatures.InputIds,
        TextFeatures.TokenTypeIds,
        TextFeatures.AttentionMask,
    ]
    for key, value in text_features.items():
        # Attention masks are padded with 0s (0 indicates no attention, 1 means
        # attention).
        # Token type ids technically doesn't matter, can be 0s.
        # Padding tokens are 0s.
        # As a result, we can set all padding to 0s.
        if key not in known_keys:
            log.warning(
                f"Unknown key {key} in text_features. Padding to 0s, but this "
                "may not be correct."
            )
        collated_text_features[key] = torch.cat(
            [
                pad_2d_unsqueeze(i, max_node_num, pad_value=0, shift=False)
                for i in value
            ]
        )

    # TODO(liamhebert): We shouldn't need this if we are using the attention mask.
    # We also use this to index which nodes in the graph are padding.
    node_mask = torch.cat(
        [
            pad_1d_unsqueeze(
                torch.ones(i.size(0)),
                shift=False,
                pad_value=False,
                padlen=max_node_num,
            )
            for i in in_degrees
        ]
    )

    # Remove placeholder images
    # TODO(liamhebert): This should be static, with mask, versus dynamic sizes
    # of images. To make this more efficient, we should set a maximum number of
    # images and always pad to that size, versus for all nodes.
    # This should operate on image_features.
    image_pixels: list[torch.Tensor] = image_features[ImageFeatures.PixelValues]
    filtered_images: list[torch.Tensor] = [
        x for x in image_pixels if not torch.all(x.eq(0))
    ]
    if len(image_pixels) != 0:
        filtered_image = torch.cat(filtered_images)
    else:
        filtered_image = torch.Tensor([])
    image_padding = torch.cat(
        [
            pad_1d_unsqueeze(
                z, max_node_num, pad_value=False, shift=False
            ).squeeze(0)
            for z in image_masks
        ]
    ).bool()

    attn_bias = torch.cat(
        [pad_attn_bias_unsqueeze(i, max_node_num + 1) for i in attn_biases]
    )
    spatial_pos = torch.cat(
        [pad_spatial_pos_unsqueeze(i, max_node_num) for i in spatial_poses]
    ).int()
    in_degree = torch.cat(
        [pad_1d_unsqueeze(i, max_node_num) for i in in_degrees]
    )

    return {
        "attn_bias": attn_bias,
        "spatial_pos": spatial_pos,
        # Since we are using undirected graph, in_degree == out_degree
        "in_degree": in_degree,
        "out_degree": in_degree,
        "node_mask": node_mask,
        "text_input": collated_text_features,
        "image_inputs": {"pixel_values": filtered_image},
        "image_padding_mask": image_padding,
    }
