import logging
from typing import Any

from data.collated_datasets import NodeBatchedDataDataset
from data.types import Labels


class HatefulDiscussions(NodeBatchedDataDataset):
    """
    Task dataset for HatefulDiscussions.
    """

    hate_labels: list[str] = [
        "DEG",
        "lti_hate",
        "IdentityDirectedAbuse",
        "AffiliationDirectedAbuse",
    ]
    not_hate_labels: list[str] = ["Neutral", "lti_normal", "NDG", "HOM"]

    def has_node_labels(self) -> bool:
        return True

    def retrieve_label(self, data: dict[str, Any]) -> dict[str, bool | int]:
        """Retrieves the label for the comment corresponding to
        HatefulDiscussions.

        Args:
            data (dict[str, Any]): The metadata of the comment.

        Returns:
            tuple[int, bool]: _description_
        """
        assert (
            "label" in data
        ), '"label" key not found in data, please check that the data format is compatible with HatefulDiscussions.'

        label = data["label"]

        if label in self.hate_labels:
            return {
                Labels.Ys: 1,
                Labels.YMask: True,
            }
        elif label in self.not_hate_labels:
            return {
                Labels.Ys: 0,
                Labels.YMask: True,
            }
        else:
            if label != "NA":
                logging.info("Encountered unknown label: %s", label)
            return {
                Labels.Ys: -1,
                Labels.YMask: False,
            }
