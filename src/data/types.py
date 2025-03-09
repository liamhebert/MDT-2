"""Enum classes for the different types of features and labels in the dataset.

This is particularly useful for dataset classes which relate to each
other, such as tasks/dataset, task specific dataset classes, and collator_utils.
"""

from enum import StrEnum


class GraphFeatures(StrEnum):
    """Graph feature names used as inputs."""

    AttnBias = "attn_biases"
    OutDegree = "out_degree"
    ImageMask = "image_mask"
    Distance = "distance"
    DistanceIndex = "distance_index"
    RotaryPos = "rotary_pos"


class TextFeatures(StrEnum):
    """Text feature names used as inputs."""

    InputIds = "input_ids"
    AttentionMask = "attention_mask"
    TokenTypeIds = "token_type_ids"
    PositionIds = "position_ids"


class ImageFeatures(StrEnum):
    """Image feature names used as inputs."""

    PixelValues = "pixel_values"


class Labels(StrEnum):
    """Label names used as targets for node-level tasks"""

    Ys = "ys"


class ContrastiveLabels(StrEnum):
    """Label names used as targets for graph-level contrastive tasks"""

    Ys = "ys"
    HardYs = "hard_ys"
