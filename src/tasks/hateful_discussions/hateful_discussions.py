"""Task dataset for hate speech detection, as used in mDT.

Hebert, L., Sahu, G., Guo, Y., Sreenivas, N. K., Golab, L., & Cohen, R. (2024).
Multi-Modal Discussion Transformer: Integrating Text, Images and Graph
Transformers to Detect Hate Speech on Social Media.
Proceedings of the AAAI Conference on Artificial Intelligence,
38(20), 22096-22104. https://doi.org/10.1609/aaai.v38i20.30213
"""

import logging
from typing import Any

from data.collated_datasets import NodeBatchedDataDataset
from data.types import Labels
import numpy as np


class HatefulDiscussions(NodeBatchedDataDataset):
    """
    Task dataset for HatefulDiscussions.
    """

    tag: str = "hateful_discussion2"

    hate_labels: list[str] = [
        "DEG",
        "True",
        "IdentityDirectedAbuse",
        "AffiliationDirectedAbuse",
        "PersonDirectedAbuse",
        "Slur",
    ]
    # not_hate_labels: list[str] = ["Neutral", "lti_normal", "NDG", "HOM"]

    keep_ids: set

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.keep_ids = set(
            np.load(
                self.root + "/" + "keep_ids.npy", allow_pickle=True
            ).flatten()
        )

    def retrieve_label(self, data: dict[str, Any]) -> dict[str, bool | int]:
        """Retrieves the label for the comment corresponding to
        HatefulDiscussions.

        Args:
            data (dict[str, Any]): The metadata of the comment.

        Returns:
            tuple[int, bool]: _description_
        """
        assert "label" in data, (
            (
                '"label" key not found in data, please check that the data'
                f" format is compatible with HatefulDiscussions.{data.keys()=}"
            ),
        )
        if data["id"] not in self.keep_ids:
            return {
                Labels.Ys: -100,
            }
        self.keep_ids.remove(data["id"])

        label = data["label"]

        if label in self.hate_labels:

            return {
                Labels.Ys: 1,
            }
        elif label != "NA":
            return {
                Labels.Ys: 0,
            }
        else:
            if label != "NA":
                logging.info("Encountered unknown label: %s", label)
            return {
                Labels.Ys: -100,
            }
