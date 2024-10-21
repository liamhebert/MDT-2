"""
Collator functions to merge data samples into a batch.
"""

from typing import List

import torch


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


def collator(
    items: List[List[torch.Tensor]], spatial_pos_max: int = 10
) -> dict[str, torch.Tensor]:
    """Collate function to merge data samples of various sizes into a batch.

    Individual data samples are comprised of the following attributes:
    - idxs: (int) list of unique indices from 0 to batch_size for each item
    - attn_biases: (List[float]) list of attention biases values for each node
        in the graph.
    - spatial_poses: (List[int]) list of spatial indexes for each node in the
        graph. Used to fetch spatial position embeddings
    - in_degrees: (List[int]) list of the in-degree for each node in the graph.
        Used to fetch degree embeddings
    - x_text: (List[Dict[str, torch.Tensor]]) list of text input data for each
        node in the graph. Each input is a dictionary with pre-tokenized text
        tokens
    - x_image_indexes: (List[torch.Tensor]) list of boolean tensors indicating
        which nodes have images
    - x_images: (List[torch.Tensor]) list of image features for each node in the
        graph
    - distance: (List[torch.Tensor]) list of exact spatial distance between
        nodes, used to clip attention bias
    - ys: (List[torch.Tensor]) list of target labels for each node in the graph
        or a single label per graph

    Args:
        items: list of data samples making up a match, with the attributes
            described above
        spatial_pos_max: maximum spatial position to attend to when computing
            attention. Any node farther away then this distance is masked.

    Returns:
        A collated patch of data samples where each item is padded to the
        largest size in the batch.

        Each output dictionary contains the following keys:
        - idx (torch.Tensor): batched indices corresponding to the data samples
            making up the batch, with shape (batch_size).
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
        - y (torch.Tensor): batched target labels for each node, with shape
            (batch_size, N).
    """

    zipped_items = zip(*items)
    assert all(len(item) == len(items[0]) for item in zipped_items)
    (
        idxs,
        attn_biases,
        spatial_poses,
        in_degrees,
        text_inputs,
        image_indices,
        images,
        distance,
        ys,
    ) = zipped_items
    # assert that each property is a list of torch tensors
    assert all(isinstance(i, torch.Tensor) for i in idxs)
    assert all(isinstance(i, torch.Tensor) for i in attn_biases)
    assert all(isinstance(i, torch.Tensor) for i in spatial_poses)
    assert all(isinstance(i, torch.Tensor) for i in in_degrees)
    assert all(isinstance(i, dict) for i in text_inputs)
    assert all(isinstance(i, torch.Tensor) for i in image_indices)
    assert all(isinstance(i, torch.Tensor) for i in images)
    assert all(isinstance(i, torch.Tensor) for i in distance)
    assert all(isinstance(i, torch.Tensor) for i in ys)

    # Clip attention bias to -inf for nodes that are farther then spatial_pos_max
    # setting to -inf sets the attention value to 0, removing them from inference
    for idx, _ in enumerate(attn_biases):
        # [1: , 1:] to avoid setting the global token to -inf
        # TODO(liamhebert): Check if this works when using cantor pairing values.
        # TODO(liamhebert): Consider never masking direct parents, only children.
        attn_biases[idx][1:, 1:][distance[idx] >= spatial_pos_max] = float(
            "-inf"
        )

    max_node_num = max(i["input_ids"].size(0) for i in text_inputs)

    y = torch.cat(ys)

    text_input: dict[str, torch.Tensor] = {}
    # currently in the format of [tokens, size]
    for key in ["input_ids", "token_type_ids", "attention_mask"]:
        # TODO(liamhebert): Consider checking that attention_mask can be padded
        # with 0s or if it should be 1s.
        # TODO(liamhebert): Likewise for token_type ids.
        text_input[key] = torch.cat(
            [
                pad_2d_unsqueeze(a[key], max_node_num, pad_value=0, shift=False)
                for a in text_inputs
            ]
        )
    # TODO(liamhebert): We shouldn't need this if we are using the attention mask.
    # We also use this to index which nodes in the graph are padding.
    node_mask = ~text_input["input_ids"].eq(0).all(dim=2)

    # Remove placeholder images
    # TODO(liamhebert): This should be static, with mask, versus dynamic sizes
    # of images.
    # To make this more efficient, we should set a maximum number of images and
    # always pad to that size, versus for all nodes.

    filtered_images: list[torch.Tensor] = [
        x for x in images if not torch.all(x.eq(0))
    ]
    if len(images) != 0:
        filtered_image = torch.cat(filtered_images)
    else:
        filtered_image = torch.Tensor([])
    image_index = torch.cat(
        [
            pad_1d_unsqueeze(
                z, max_node_num, pad_value=False, shift=False
            ).squeeze(0)
            for z in image_indices
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
        "idx": torch.LongTensor(idxs),
        "attn_bias": attn_bias,
        "spatial_pos": spatial_pos,
        # Since we are using undirected graph, in_degree == out_degree
        "in_degree": in_degree,
        "out_degree": in_degree,
        "node_mask": node_mask,
        "input_ids": text_input["input_ids"],
        "token_type_ids": text_input["token_type_ids"],
        "attention_mask": text_input["attention_mask"],
        "images": filtered_image,
        "image_indices": image_index,
        "y": y,
    }
