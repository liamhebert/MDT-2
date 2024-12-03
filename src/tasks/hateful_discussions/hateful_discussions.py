import logging
from typing import Any

from tasks.dataset import TaskDataset


class HatefulDiscussions(TaskDataset):
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

    def retrieve_label(self, data: dict[str, Any]) -> tuple[int, bool]:
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
            return 1, True
        elif label in self.not_hate_labels:
            return 0, True
        else:
            if label != "NA":
                logging.info("Encountered unknown label: %s", label)
            return -1, False
