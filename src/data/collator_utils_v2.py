"""
Collator functions to merge data samples into a batch.
"""

from typing import List

import torch

from data.types import GraphFeatures
from data.types import ImageFeatures
from data.types import TextFeatures
from utils.pylogger import RankedLogger
from components.v2.graph_attention_mask import PADDING_GRAPH_ID
from torch.nn.attention.flex_attention import _DEFAULT_SPARSE_BLOCK_SIZE
from components.v2.graph_attention_mask import (
    generate_graph_attn_mask_tensor,
)

log = RankedLogger(__name__)


InputFeatures = dict[str, List[torch.Tensor]]


def generic_collator(
    graph_features: InputFeatures,
    text_features: InputFeatures,
    image_features: InputFeatures,
    block_size: int = _DEFAULT_SPARSE_BLOCK_SIZE * 2,
) -> dict[str, torch.Tensor | dict[str, torch.Tensor] | int]:
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
        - spatial_pos (torch.Tensor): batched spatial positions,
            corresponding to indices to fetch spatial position embeddings.
            Padded with 0s and shifted by 1. Shape (batch_size * N,
            batch_size * N).
        - graph_ids (torch.Tensor): batched graph ids, corresponding to the
            graph id of each node in the graph. Padded with -1s. Shape
            (batch_size * N).
        - in_degree (torch.Tensor): batched in-degrees, corresponding to the
            in-degree of each node in the graph. Padded with 0s and shifted
            by 1. Shape (batch_size * N).
        - out_degree (torch.Tensor): batched out-degrees, corresponding to
            the out-degree of each node in the graph. Padded with 0s and
            shifted by 1. Shape (batch_size * N).
        - text_input (dict): batched text inputs, with the following keys:
            - input_ids (torch.Tensor): batched tokenized text ids, with shape
                (batch_size * N, T)
            - token_type_ids (torch.Tensor): batched token type ids, with shape
                (batch_size * N, T)
            - attention_mask (torch.Tensor): batched text attention mask, with
                shape (batch_size * N, T), where 1 indicates a token that should
                be attended to and 0 indicates padding.
        - image_inputs: batched image inputs, with the following keys:
            - pixel_values (torch.Tensor): batched image features, with shape
                (batch_size * num_images, D)
        - image_padding_mask (torch.Tensor): batched boolean tensor indicating
            which nodes have images, where the order of True values corresponds
            to the order of images in the images tensor. Shape (batch_size * N).
    """

    # block_size = 32

    in_degrees: list[torch.Tensor] = graph_features[GraphFeatures.InDegree]
    image_masks: list[torch.Tensor] = graph_features[GraphFeatures.ImageMask]
    distances: list[torch.Tensor] = graph_features[GraphFeatures.Distance]
    distance_indices: list[torch.Tensor] = graph_features[
        GraphFeatures.DistanceIndex
    ]
    rotary_poses: list[torch.Tensor] = graph_features[GraphFeatures.RotaryPos]

    # assert that each property is a list of torch tensors
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
    num_graphs = len(in_degrees)  # We have one list for each graph
    total_num_nodes = sum([len(i) for i in in_degrees])
    # We add num_graphs because of the bonus tokens we add
    num_padding = block_size - ((total_num_nodes + num_graphs) % block_size)
    if num_padding == block_size:
        num_padding = 0

    collated_text_features: dict[str, torch.Tensor] = {}

    # As we try out newer models, we may need to change this as we verify them.
    known_keys = [
        TextFeatures.InputIds,
        TextFeatures.TokenTypeIds,
        TextFeatures.AttentionMask,
        TextFeatures.PositionIds,
    ]
    text_size = text_features[TextFeatures.InputIds][0].size(1)
    padding = torch.zeros((num_padding, text_size), dtype=torch.long)
    for key, value in text_features.items():
        if key not in known_keys:
            log.warning(
                f"Unknown key {key} in text_features. Padding to 0s, but this "
                "may not be correct."
            )
        # All tensors should already be padded to the same sequence length
        # NOTE: Not that it matters, but some models have EOS at the end of the
        # sequence, so padding with 0s is not always correct.
        # We should be fine here.
        collated_text_features[key] = torch.cat(value, dim=0)
        collated_text_features[key] = torch.cat(
            [collated_text_features[key], padding]
        )

    # add graph ids for flat masking later
    # The process works as follows:
    # Given a list of graphs with different sizes ex: (2, 3, 4)
    # First create the indices that we need (ex: (0, 1, 2), since we have 3
    # graphs)

    graph_indices = torch.arange(num_graphs)

    # Then compute the sizes of each graph (ex: (2, 3, 4))
    graph_sizes = torch.tensor([len(i) for i in in_degrees])

    # Then repeat the indices by the sizes (ex: (0, 0, 1, 1, 1, 2, 2, 2, 2))
    graph_ids = torch.repeat_interleave(graph_indices, graph_sizes)

    # Remove placeholder images
    # TODO(liamhebert): This should be static, with mask, versus dynamic sizes
    # of images. To make this more efficient, we should set a maximum number of
    # images and always pad to that size, versus for all nodes.
    # This should operate on image_features.
    image_pixels: list[torch.Tensor] = image_features[ImageFeatures.PixelValues]

    filtered_images: list[torch.Tensor] = [x for x in image_pixels if not None]
    if len(image_pixels) != 0:
        # filtered_image = torch.cat(
        #     [
        #         pad_2d_unsqueeze(x, max_node_num, pad_value=0, shift=False)
        #         for x in filtered_images
        #     ]
        # )
        filtered_image = torch.cat(filtered_images)
    else:
        filtered_image = torch.Tensor([])

    image_padding = torch.cat(image_masks)

    # Add padding to the in_degrees, image_masks, distances, and distance_indices
    virtual_distance_pad = torch.zeros((num_padding, 2), dtype=torch.long)
    virtual_distance_graph = torch.zeros((num_graphs, 2), dtype=torch.long)
    rotary_pos = torch.cat(rotary_poses)
    rotary_pos = torch.cat(
        [virtual_distance_graph, rotary_pos, virtual_distance_pad], dim=0
    )

    spatial_pos = torch.block_diag(*[x + 1 for x in distance_indices])
    in_degree = torch.cat(in_degrees) + 1

    padding_1d = torch.zeros((num_padding,), dtype=torch.long)
    in_degree = torch.cat([in_degree, padding_1d])
    image_padding = torch.cat([image_padding, padding_1d.bool()])

    padding_graph_index = torch.full(
        (num_padding,), PADDING_GRAPH_ID, dtype=torch.long
    )
    graph_ids = torch.cat([graph_ids, padding_graph_index])

    total_num_graphs = len(in_degrees)

    # We add distance to the spatial pos for the padding nodes (end) and the
    # graph tokens (start)

    # add on the key dimension
    virtual_distance_pad = torch.zeros_like(spatial_pos[0, :]).expand(
        (num_padding, -1)
    )
    virtual_distance_graph = torch.zeros_like(spatial_pos[0, :]).expand(
        (total_num_graphs, -1)
    )
    spatial_pos = torch.cat(
        (virtual_distance_graph, spatial_pos, virtual_distance_pad), dim=0
    )

    # add on the query dimension
    virtual_distance_pad = torch.zeros_like(spatial_pos[:, 0]).expand(
        (num_padding, -1)
    )
    virtual_distance_graph = torch.zeros_like(spatial_pos[:, 0]).expand(
        (total_num_graphs, -1)
    )
    spatial_pos = torch.cat(
        (virtual_distance_graph.T, spatial_pos, virtual_distance_pad.T), dim=1
    )

    num_graphs = len(in_degrees)
    graph_ids = torch.cat((torch.arange(0, num_graphs), graph_ids), dim=0)

    flex_block_mask = generate_graph_attn_mask_tensor(
        graph_ids,
        spatial_pos,
        max_spatial_distance=10,
        block_size=block_size,
    )

    return {
        # Since we are using undirected graphs, in_degree == out_degree
        "out_degree": in_degree,
        "text_input": collated_text_features,
        "image_inputs": {"pixel_values": filtered_image},
        "image_padding_mask": image_padding,
        "num_total_graphs": len(in_degrees),
        "rotary_pos": rotary_pos,
        "graph_mask": flex_block_mask,
    }
