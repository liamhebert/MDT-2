from enum import StrEnum


class GraphFeatures(StrEnum):
    AttnBias = "attn_biases"
    SpatialPos = "spatial_poses"
    InDegree = "in_degree"
    ImageMask = "image_mask"
    Distance = "distance"
    DistanceIndex = "distance_index"


class TextFeatures(StrEnum):
    InputIds = "input_ids"
    AttentionMask = "attention_mask"
    TokenTypeIds = "token_type_ids"


class ImageFeatures(StrEnum):
    PixelValues = "pixel_values"


class Labels(StrEnum):
    Ys = "ys"
    YMask = "y_mask"


class ContrastiveLabels(Labels):
    HardYs = "hard_ys"
