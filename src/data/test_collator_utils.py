from enum import auto
from enum import IntEnum

import torch

from data import collator_utils as cut
from data.types import GraphFeatures
from data.types import ImageFeatures
from data.types import TextFeatures


def test_pad_1d_unsqueeze():
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
    # Test case: pad attention bias
    x = torch.tensor([[1, 2], [3, 4]])
    padlen = 4

    result = cut.pad_attn_bias_unsqueeze(x, padlen)
    expected = torch.tensor(
        [
            [
                [1, 2, -float("inf"), -float("inf")],
                [3, 4, -float("inf"), -float("inf")],
                [0, 0, -float("inf"), -float("inf")],
                [0, 0, -float("inf"), -float("inf")],
            ]
        ]
    )

    assert torch.equal(
        result, expected
    ), f"Expected {expected}, but got {result}"


def test_pad_edge_type_unsqueeze():
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
    ATTN_BIAS = auto()
    DISTANCE = auto()
    DISTANCE_INDEX = auto()
    IN_DEGREE = auto()
    INPUT_IDS = auto()
    TOKEN_TYPE_IDS = auto()
    ATTENTION_MASK = auto()
    IMAGES = auto()
    NODE_MASK = auto()
    Y = auto()


def create_sample_input(
    num_nodes: int, text_length: int, num_images: int, image_length: int
):
    graph_features = {
        GraphFeatures.AttnBias: torch.tensor(
            [[DummyValues.ATTN_BIAS] * num_nodes] * num_nodes
        ),
        GraphFeatures.SpatialPos: torch.tensor(
            [[DummyValues.DISTANCE] * num_nodes] * num_nodes
        ),
        GraphFeatures.InDegree: torch.tensor(
            [DummyValues.IN_DEGREE] * num_nodes
        ),
        GraphFeatures.ImageMask: torch.tensor(
            [True] * num_images + [False] * (num_nodes - num_images)
        ),
        GraphFeatures.Distance: torch.tensor(
            [[DummyValues.DISTANCE] * num_nodes] * num_nodes
        ),  # dummy value
        GraphFeatures.DistanceIndex: torch.tensor(
            [[0] * num_nodes] * num_nodes
        ),  # dummy value
    }
    return {}
    return [
        torch.tensor([num_nodes]),  # idxs
        torch.tensor(
            [[ATTN_BIAS * num_nodes] * num_nodes] * num_nodes
        ),  # attn_bias
        torch.tensor(
            [[SPATIAL_POS * num_nodes] * num_nodes] * num_nodes
        ),  # spatial_pos
        torch.tensor([IN_DEGREE * num_nodes] * num_nodes),  # in_degree
        {
            "input_ids": torch.tensor(
                [[INPUT_IDS * num_nodes] * text_length] * num_nodes
            ),
            "token_type_ids": torch.tensor(
                [[TOKEN_TYPE_IDS * num_nodes] * text_length] * num_nodes
            ),
            "attention_mask": torch.tensor(
                [[ATTENTION_MASK * num_nodes] * text_length] * num_nodes
            ),
        },  # text_inputs
        torch.tensor(
            [True] * num_images + [False] * (num_nodes - num_images)
        ),  # image_indices
        torch.tensor(
            [[IMAGES * num_nodes] * image_length] * num_images
        ),  # images
        torch.tensor([NODE_MASK * num_nodes] * num_nodes),  # node_mask
        torch.tensor([Y * num_nodes] * num_nodes),  # y
    ]


def test_collator():
    items = [
        create_sample_input(
            num_nodes=3, text_length=2, num_images=2, image_length=2
        ),
        create_sample_input(
            num_nodes=2, text_length=5, num_images=1, image_length=3
        ),
    ]

    result = cut.generic_collator(items)
    expected = {
        "idx": torch.tensor([3, 2]),
        "attn_bias": torch.tensor(
            [
                [
                    [ATTN_BIAS, ATTN_BIAS, ATTN_BIAS],
                    [ATTN_BIAS, ATTN_BIAS, ATTN_BIAS],
                    [ATTN_BIAS, ATTN_BIAS, ATTN_BIAS],
                ],
                [
                    [ATTN_BIAS, ATTN_BIAS, -float("inf")],
                    [ATTN_BIAS, ATTN_BIAS, -float("inf")],
                    [ATTN_BIAS, ATTN_BIAS, -float("inf")],
                ],
            ]
        ),
        "spatial_pos": torch.tensor(
            [
                [[2, 3, 0], [4, 5, 0], [0, 0, 0]],
                [[6, 7, 0], [8, 9, 0], [0, 0, 0]],
            ]
        ),
        "in_degree": torch.tensor([[2, 3, 0], [4, 5, 0]]),
        "out_degree": torch.tensor([[2, 3, 0], [4, 5, 0]]),
        "node_mask": torch.tensor([[True, True, False], [True, True, False]]),
        "input_ids": torch.tensor(
            [[[1, 2], [3, 4], [0, 0]], [[5, 6], [7, 8], [0, 0]]]
        ),
        "token_type_ids": torch.tensor(
            [[[1, 1], [1, 1], [0, 0]], [[1, 1], [1, 1], [0, 0]]]
        ),
        "attention_mask": torch.tensor(
            [[[1, 1], [1, 1], [0, 0]], [[1, 1], [1, 1], [0, 0]]]
        ),
        "images": torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
        "image_indices": torch.tensor(
            [[True, False, False], [False, True, False]]
        ),
        "y": torch.tensor([1, 2, 3, 4]),
    }

    for key in expected:
        assert torch.equal(
            result[key], expected[key]
        ), f"Expected {expected[key]} for key {key}, but got {result[key]}"
