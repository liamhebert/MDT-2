from enum import auto
from enum import IntEnum

import torch

from data import collator_utils as cut
from data.types import GraphFeatures
from data.types import ImageFeatures
from data.types import TextFeatures
from torch_geometric.data import Data
from transformers import BatchEncoding


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
                [1, 2, -torch.inf, -torch.inf],
                [3, 4, -torch.inf, -torch.inf],
                [0, 0, -torch.inf, -torch.inf],
                [0, 0, -torch.inf, -torch.inf],
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
) -> Data:

    data = Data(
        attn_bias=torch.tensor(
            [[DummyValues.ATTN_BIAS * num_nodes] * num_nodes] * num_nodes
        ),
        in_degree=torch.tensor([DummyValues.IN_DEGREE * num_nodes] * num_nodes),
        image_mask=torch.tensor(
            [True] * num_images + [False] * (num_nodes - num_images)
        ),
        distance=torch.tensor(
            [[DummyValues.DISTANCE * num_nodes] * num_nodes] * num_nodes
        ),  # dummy value
        distance_index=torch.tensor(
            [[DummyValues.DISTANCE_INDEX * num_nodes] * num_nodes] * num_nodes
        ),  # dummy value
        text=BatchEncoding(
            {
                TextFeatures.InputIds: torch.tensor(
                    [[DummyValues.INPUT_IDS * num_nodes] * text_length]
                    * num_nodes
                ),
                TextFeatures.TokenTypeIds: torch.tensor(
                    [[DummyValues.TOKEN_TYPE_IDS * num_nodes] * text_length]
                    * num_nodes
                ),
                TextFeatures.AttentionMask: torch.tensor(
                    [[DummyValues.ATTENTION_MASK * num_nodes] * text_length]
                    * num_nodes
                ),
            }
        ),
        image=BatchEncoding(
            {
                ImageFeatures.PixelValues: torch.tensor(
                    [[DummyValues.IMAGES * num_nodes] * image_length]
                    * num_images
                    + [[0] * image_length] * (num_nodes - num_images)
                ),
            }
        ),
    )
    return data


def test_collator():
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
        "in_degree": torch.tensor(
            [
                [
                    DummyValues.IN_DEGREE * 3 + 1,
                    DummyValues.IN_DEGREE * 3 + 1,
                    DummyValues.IN_DEGREE * 3 + 1,
                ],
                [
                    DummyValues.IN_DEGREE * 2 + 1,
                    DummyValues.IN_DEGREE * 2 + 1,
                    0,
                ],
            ]
        ),
        "out_degree": torch.tensor(
            [
                [
                    DummyValues.IN_DEGREE * 3 + 1,
                    DummyValues.IN_DEGREE * 3 + 1,
                    DummyValues.IN_DEGREE * 3 + 1,
                ],
                [
                    DummyValues.IN_DEGREE * 2 + 1,
                    DummyValues.IN_DEGREE * 2 + 1,
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
        # TODO(liamhebert): This input is unrealistic and should be fixed.
        "image_inputs": {
            "pixel_values": torch.tensor(
                [
                    [[DummyValues.IMAGES * 3] * 3] * 2
                    + [[DummyValues.IMAGES * 2] * 3],
                ]
            )
        },
        "image_padding_mask": torch.tensor(
            [[True, True, False], [True, False, False]]
        ),
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
                ), f"Expected {subvalue} for key {key}-{subkey}, but got {result[key][subkey]}"
