from enum import auto
from enum import IntEnum

import torch
from transformers import BatchEncoding

from data import collator_utils as cut
from data.types import ImageFeatures
from data.types import TextFeatures
from components.v2.graph_attention_mask import (
    PADDING_GRAPH_ID,
    generate_graph_attn_mask_tensor,
)
from pytest import mark


class DummyValues(IntEnum):
    """Unique values for each feature in the sample input."""

    ATTN_BIAS = auto()
    DISTANCE = auto()
    DISTANCE_INDEX = auto()
    OUT_DEGREE = auto()
    INPUT_IDS = auto()
    TOKEN_TYPE_IDS = auto()
    ATTENTION_MASK = auto()
    IMAGES = auto()
    NODE_MASK = auto()
    Y = auto()
    ROTARY_POS = auto()


def create_sample_input(
    num_nodes: int, text_length: int, num_images: int, image_length: int
) -> dict:
    """Utility function to create a sample input for testing the collator."""

    data = dict(
        attn_bias=torch.full(
            (num_nodes, num_nodes), DummyValues.ATTN_BIAS * num_nodes
        ),
        out_degree=torch.full((num_nodes,), DummyValues.OUT_DEGREE * num_nodes),
        image_mask=torch.tensor(
            [True] * num_images + [False] * (num_nodes - num_images)
        ),
        distance=torch.full(
            (num_nodes, num_nodes, 2), DummyValues.DISTANCE * num_nodes
        ),
        text=BatchEncoding(
            {
                TextFeatures.InputIds: torch.full(
                    (num_nodes, text_length), DummyValues.INPUT_IDS * num_nodes
                ),
                TextFeatures.TokenTypeIds: torch.full(
                    (num_nodes, text_length),
                    DummyValues.TOKEN_TYPE_IDS * num_nodes,
                ),
                TextFeatures.AttentionMask: torch.full(
                    (num_nodes, text_length),
                    DummyValues.ATTENTION_MASK * num_nodes,
                ),
            }
        ),
        images=BatchEncoding(
            {
                ImageFeatures.PixelValues: torch.full(
                    (num_images, 3, image_length, image_length),
                    DummyValues.IMAGES * num_nodes,
                )
            }
        ),
        rotary_position=torch.full(
            (num_nodes, 2), DummyValues.ROTARY_POS * num_nodes
        ),
    )
    return data


@mark.parametrize("with_token_type_ids", [True, False])
def test_collator_v2(with_token_type_ids: bool):
    """Testing the extract_and_merge and generic_collator functions.

    This is effectively an end-to-end test for the v2 collator, which uses
    (batch * nodes, features) flattened format for the input data.
    """
    items = [
        create_sample_input(
            num_nodes=3, text_length=5, num_images=2, image_length=3
        ),
        create_sample_input(
            num_nodes=2, text_length=5, num_images=1, image_length=3
        ),
    ]

    if not with_token_type_ids:
        for item in items:
            item["text"].pop(TextFeatures.TokenTypeIds)

    graph_features, text_features, image_features = (
        cut.extract_and_merge_features(items)
    )

    result = cut.generic_collator(
        graph_features, text_features, image_features, block_size=4
    )
    non_block_spatial_pos = [
        torch.full((2, 2), 0),  # Graph Token distance
        torch.full((3, 3), DummyValues.DISTANCE_INDEX * 3 + 1),
        torch.full((2, 2), DummyValues.DISTANCE_INDEX * 2 + 1),
        torch.full((1, 1), 0),  # Padding
    ]
    spatial_pos = torch.block_diag(*non_block_spatial_pos)
    graph_ids = torch.tensor([0, 1, 0, 0, 0, 1, 1, PADDING_GRAPH_ID])
    graph_mask = generate_graph_attn_mask_tensor(
        graph_ids=graph_ids,
        spatial_distance_matrix=spatial_pos,
        max_spatial_distance=20,
        block_size=4,
    )
    expected = {
        "graph_mask": graph_mask,
        "graph_ids": torch.tensor([0, 1, 0, 0, 0, 1, 1, -1]),
        "out_degree": torch.tensor(
            [
                DummyValues.OUT_DEGREE * 3 + 1,
                DummyValues.OUT_DEGREE * 3 + 1,
                DummyValues.OUT_DEGREE * 3 + 1,
                DummyValues.OUT_DEGREE * 2 + 1,
                DummyValues.OUT_DEGREE * 2 + 1,
                0,
            ]
        ),
        "text_input": {
            "input_ids": torch.cat(
                [
                    torch.full((3, 5), DummyValues.INPUT_IDS * 3),
                    torch.full((2, 5), DummyValues.INPUT_IDS * 2),
                    torch.full((1, 5), 0),
                ]
            ),
            "token_type_ids": torch.cat(
                [
                    torch.full((3, 5), DummyValues.TOKEN_TYPE_IDS * 3),
                    torch.full((2, 5), DummyValues.TOKEN_TYPE_IDS * 2),
                    torch.full((1, 5), 0),
                ]
            ),
            "attention_mask": torch.cat(
                [
                    torch.full((3, 5), DummyValues.ATTENTION_MASK * 3),
                    torch.full((2, 5), DummyValues.ATTENTION_MASK * 2),
                    torch.full((1, 5), 0),
                ]
            ),
        },
        "image_input": {
            "pixel_values": torch.cat(
                [
                    torch.full((2, 3, 3, 3), DummyValues.IMAGES * 3),
                    torch.full((1, 3, 3, 3), DummyValues.IMAGES * 2),
                ]
            )
        },
        "image_padding_mask": torch.tensor(
            [True, True, False, True, False, False]
        ),
    }

    if not with_token_type_ids:
        expected["text_input"].pop("token_type_ids")

    for key, value in expected.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(
                result[key], expected[key]
            ), f"Expected {expected[key]} for key {key}, but got {result[key]}"
        elif isinstance(value, dict):
            for subkey, subvalue in value.items():
                assert torch.equal(
                    result[key][subkey],
                    subvalue,
                ), (
                    f"Expected {subvalue} for key {key}-{subkey}, but"
                    "got {result[key][subkey]}"
                )
