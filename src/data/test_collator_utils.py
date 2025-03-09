from enum import auto
from enum import IntEnum

import torch
from transformers import BatchEncoding

from data import collator_utils as cut
from data import collator_utils_v2 as cut_v2
from data.types import ImageFeatures
from data.types import TextFeatures
from components.v2.graph_attention_mask import (
    PADDING_GRAPH_ID,
    generate_graph_attn_mask_tensor,
)
from pytest import mark


def test_pad_1d_unsqueeze():
    """Testing the pad_1d_unsqueeze function."""
    # Test case 1: pad with shift
    x = torch.tensor([1, 2, 3])
    padlen = 5
    pad_value = 0
    shift = True

    result = cut.pad_1d_unsqueeze(x, padlen, pad_value, shift)
    expected = torch.tensor([[2, 3, 4, 0, 0]])

    assert torch.equal(
        result, expected
    ), f"Expected {expected}, but got {result}"

    # Test case 2: pad without shift
    shift = False
    result = cut.pad_1d_unsqueeze(x, padlen, pad_value, shift)
    expected = torch.tensor([[1, 2, 3, 0, 0]])

    assert torch.equal(
        result, expected
    ), f"Expected {expected}, but got {result}"

    # Test case 3: pad with different pad_value
    pad_value = -1
    result = cut.pad_1d_unsqueeze(x, padlen, pad_value, shift)
    expected = torch.tensor([[1, 2, 3, -1, -1]])

    assert torch.equal(
        result, expected
    ), f"Expected {expected}, but got {result}"

    # Test case 4: no padding needed
    x = torch.tensor([1, 2, 3, 4, 5])
    padlen = 3
    result = cut.pad_1d_unsqueeze(x, padlen, pad_value, shift)
    expected = torch.tensor([[1, 2, 3, 4, 5]])

    assert torch.equal(
        result, expected
    ), f"Expected {expected}, but got {result}"


def test_pad_2d_unsqueeze():
    """Testing the pad_2d_unsqueeze function."""
    # Test case 1: pad with shift
    x = torch.tensor([[1, 2], [3, 4]])
    padlen = 4
    pad_value = 0
    shift = True

    result = cut.pad_2d_unsqueeze(x, padlen, pad_value, shift)
    expected = torch.tensor([[[2, 3], [4, 5], [0, 0], [0, 0]]])

    assert torch.equal(
        result, expected
    ), f"Expected {expected}, but got {result}"

    # Test case 2: pad without shift
    shift = False
    result = cut.pad_2d_unsqueeze(x, padlen, pad_value, shift)
    expected = torch.tensor([[[1, 2], [3, 4], [0, 0], [0, 0]]])

    assert torch.equal(
        result, expected
    ), f"Expected {expected}, but got {result}"

    # Test case 3: pad with different pad_value
    pad_value = -1
    result = cut.pad_2d_unsqueeze(x, padlen, pad_value, shift)
    expected = torch.tensor([[[1, 2], [3, 4], [-1, -1], [-1, -1]]])

    assert torch.equal(
        result, expected
    ), f"Expected {expected}, but got {result}"

    # Test case 4: no padding needed
    x = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
    padlen = 2
    result = cut.pad_2d_unsqueeze(x, padlen, pad_value, shift)
    expected = torch.tensor([[[1, 2], [3, 4], [5, 6], [7, 8]]])

    assert torch.equal(
        result, expected
    ), f"Expected {expected}, but got {result}"


def test_pad_attn_bias_unsqueeze():
    """Testing the pad_attn_bias_unsqueeze function."""
    # Test case: pad attention bias
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    padlen = 4

    result = cut.pad_attn_bias_unsqueeze(x, padlen)
    expected = torch.tensor(
        [
            [
                [1.0, 2.0, -torch.inf, -torch.inf],
                [3.0, 4.0, -torch.inf, -torch.inf],
                [0.0, 0.0, -torch.inf, -torch.inf],
                [0.0, 0.0, -torch.inf, -torch.inf],
            ]
        ]
    )

    assert torch.equal(
        result, expected
    ), f"Expected {expected}, but got {result}"


def test_pad_edge_type_unsqueeze():
    """Testing the pad_edge_type_unsqueeze function."""
    # Test case: pad edge type
    x = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    padlen = 4

    result = cut.pad_edge_type_unsqueeze(x, padlen)
    expected = torch.tensor(
        [
            [
                [[1, 2], [3, 4], [0, 0], [0, 0]],
                [[5, 6], [7, 8], [0, 0], [0, 0]],
                [[0, 0], [0, 0], [0, 0], [0, 0]],
                [[0, 0], [0, 0], [0, 0], [0, 0]],
            ]
        ]
    )

    assert torch.equal(
        result, expected
    ), f"Expected {expected}, but got {result}"


def test_pad_spatial_pos_unsqueeze():
    """Testing the pad_spatial_pos function."""
    # Test case: pad spatial position
    x = torch.tensor([[1, 2], [3, 4]])
    padlen = 4

    result = cut.pad_spatial_pos_unsqueeze(x, padlen)
    expected = torch.tensor(
        [[[2, 3, 0, 0], [4, 5, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]
    )

    assert torch.equal(
        result, expected
    ), f"Expected {expected}, but got {result}"


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


@mark.skip("We dont use the v1 collator")
def test_collator_v1():
    """Testing the extract_and_merge and generic_collator functions.

    This is effectively an end-to-end test for the v1 collator, which uses
    (batch, nodes, features) format for the input data, rather then flattened as
    is in the v2 collator.
    """
    items = [
        create_sample_input(
            num_nodes=3, text_length=5, num_images=2, image_length=3
        ),
        create_sample_input(
            num_nodes=2, text_length=5, num_images=1, image_length=3
        ),
    ]

    graph_features, text_features, image_features = (
        cut.extract_and_merge_features(items)
    )

    result = cut.generic_collator(
        graph_features, text_features, image_features, spatial_pos_max=100000
    )
    expected = {
        "attn_bias": torch.tensor(
            [
                [
                    [
                        DummyValues.ATTN_BIAS * 3,
                        DummyValues.ATTN_BIAS * 3,
                        DummyValues.ATTN_BIAS * 3,
                    ],
                    [
                        DummyValues.ATTN_BIAS * 3,
                        DummyValues.ATTN_BIAS * 3,
                        DummyValues.ATTN_BIAS * 3,
                    ],
                    [
                        DummyValues.ATTN_BIAS * 3,
                        DummyValues.ATTN_BIAS * 3,
                        DummyValues.ATTN_BIAS * 3,
                    ],
                ],
                [
                    [
                        DummyValues.ATTN_BIAS * 2,
                        DummyValues.ATTN_BIAS * 2,
                        -torch.inf,
                    ],
                    [
                        DummyValues.ATTN_BIAS * 2,
                        DummyValues.ATTN_BIAS * 2,
                        -torch.inf,
                    ],
                    [
                        0,
                        0,
                        -torch.inf,
                    ],
                ],
            ]
        ),
        "spatial_pos": torch.tensor(
            [
                [
                    [
                        DummyValues.DISTANCE_INDEX * 3 + 1,
                        DummyValues.DISTANCE_INDEX * 3 + 1,
                        DummyValues.DISTANCE_INDEX * 3 + 1,
                    ],
                    [
                        DummyValues.DISTANCE_INDEX * 3 + 1,
                        DummyValues.DISTANCE_INDEX * 3 + 1,
                        DummyValues.DISTANCE_INDEX * 3 + 1,
                    ],
                    [
                        DummyValues.DISTANCE_INDEX * 3 + 1,
                        DummyValues.DISTANCE_INDEX * 3 + 1,
                        DummyValues.DISTANCE_INDEX * 3 + 1,
                    ],
                ],
                [
                    [
                        DummyValues.DISTANCE_INDEX * 2 + 1,
                        DummyValues.DISTANCE_INDEX * 2 + 1,
                        0,
                    ],
                    [
                        DummyValues.DISTANCE_INDEX * 2 + 1,
                        DummyValues.DISTANCE_INDEX * 2 + 1,
                        0,
                    ],
                    [
                        0,
                        0,
                        0,
                    ],
                ],
            ]
        ),
        "out_degree": torch.tensor(
            [
                [
                    DummyValues.OUT_DEGREE * 3 + 1,
                    DummyValues.OUT_DEGREE * 3 + 1,
                    DummyValues.OUT_DEGREE * 3 + 1,
                ],
                [
                    DummyValues.OUT_DEGREE * 2 + 1,
                    DummyValues.OUT_DEGREE * 2 + 1,
                    0,
                ],
            ]
        ),
        "node_mask": torch.tensor([[True, True, True], [True, True, False]]),
        "text_input": {
            "input_ids": torch.tensor(
                [
                    [[DummyValues.INPUT_IDS * 3] * 5] * 3,
                    [[DummyValues.INPUT_IDS * 2] * 5] * 2 + [[0] * 5],
                ]
            ),
            "token_type_ids": torch.tensor(
                [
                    [[DummyValues.TOKEN_TYPE_IDS * 3] * 5] * 3,
                    [[DummyValues.TOKEN_TYPE_IDS * 2] * 5] * 2 + [[0] * 5],
                ]
            ),
            "attention_mask": torch.tensor(
                [
                    [[DummyValues.ATTENTION_MASK * 3] * 5] * 3,
                    [[DummyValues.ATTENTION_MASK * 2] * 5] * 2 + [[0] * 5],
                ]
            ),
        },
        "image_inputs": {
            "pixel_values": torch.cat(
                [
                    torch.full((2, 3, 3, 3), DummyValues.IMAGES * 3),
                    torch.full((1, 3, 3, 3), DummyValues.IMAGES * 2),
                ]
            )
        },
        "image_padding_mask": torch.tensor(
            [[True, True, False], [True, False, False]]
        ),
        "graph_ids": torch.tensor([[0, 0, 0], [1, 1, -1]]),
    }

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
                    f"Expected {subvalue} for key {key}-{subkey}, but got "
                    f"{result[key][subkey]}"
                )


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

    result = cut_v2.generic_collator(
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
        "image_inputs": {
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
