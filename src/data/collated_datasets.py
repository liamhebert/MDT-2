from tasks.dataset import TaskDataset
from torch_geometric.data import Data
from abc import abstractmethod
import torch
from data import collator_utils
from data.types import ContrastiveLabels, Labels


class CollatedDataset(TaskDataset):

    @abstractmethod
    def label_collate_fn(
        self,
        batch: list[Data],
        batch_size: int,
        max_nodes: int,
    ) -> dict[str, torch.Tensor]: ...

    def collate_fn(self, batch: list[Data]) -> Data:
        """Collate function to merge data samples of various sizes into a batch.

        Individual data samples must contain the following attributes:
        - text (dict[str, Tensor]): The tokenized text of the graph as produced
            by the text_tokenizer, with keys "input_ids", "attention_mask", and
            "token_type_ids", each with shape [num_nodes, max_text_length].
        - edge_index (Tensor): The edge index of the graph, with shape
            [2, num_edges].
        - image_mask (Tensor): A mask indicating whether a node has an image or
            not, with shape [num_nodes].
        - images (dict[str, Tensor]): The tokenized images of the graph as
            produced by the image_tokenizer ("pixel_values"), with shape
            [num_images, ...].
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

        And auxiliary label information required by label_collate_fn.

        Args:
            batch: list of data samples to collate.

        Returns:
            A collated patch of data samples where each item is padded to the
            largest size in the batch.

            Each output dictionary must contains the following keys:
            - attn_bias: (torch.Tensor) batched attention biases, with shape
                [batch_size * num_nodes, batch_size * num_nodes].
            - segment_ids: (torch.Tensor) batched segment ids, indicating which
                graph a given node belongs to, with shape [batch_size * num_nodes].
            - spatial_pos: (torch.Tensor) batched spatial positions, with shape
                [batch_size * num_nodes, 2].
            - in_degree: (torch.Tensor) batched in-degrees, with shape
                [batch_size * num_nodes].
            - out_degree: (torch.Tensor) batched out-degrees, with shape
                [batch_size * num_nodes].
            - text: (dict[str, torch.Tensor]) all features for the text model
                batched. The keys should include "input_ids", "attention_mask",
                and "token_type_ids". Each tensor should have the shape
                [batch_size * num_nodes, max_text_length, ...].
            - node_attention_mask: (torch.Tensor) batched attention mask, with
                shape [batch_size * num_nodes]
            - x_images: (dict[str, torch.Tensor]) all features for the image
                model batched. The keys should include "pixel_values". Each
                tensor should have the shape [batch_size * num_images, ...].
            - x_image_mask: (torch.Tensor) batched image mask, where 1 indicates
                the presence of an image and 0 indicates the absence, with shape
                [batch_size * num_nodes].
            - y: (torch.Tensor) batched target labels. The shape should be either
                [batch_size * num_nodes] or [batch_size], depending on the task.

            Additional features for task specific loss functions may also be
            added but they are not needed for general processing.
        """
        graphs = [item for item in batch if item is not None]

        graph_features, text_features, image_features = (
            collator_utils.extract_and_merge_features(graphs)
        )

        collated_output = collator_utils.generic_collator(
            graph_features, text_features, image_features, self.spatial_pos_max
        )
        # TODO(liamhebert): If we swap to a flattened batch, this will break.
        batch_size, num_nodes = collated_output["node_mask"].shape

        # TODO(liamhebert): Consider making this generic to collate extra
        # features beyond just the labels.
        collated_output["y"] = self.label_collate_fn(
            graphs, batch_size, num_nodes
        )
        return collated_output


class ContrastiveTaskDataset(CollatedDataset):
    """
    Dataset with contrastive learning specific collate function.
    """

    def label_collate_fn(
        self, batch: list[Data], batch_size: int, max_nodes: int
    ) -> dict[str, torch.Tensor]:
        """Collate function specific to contrastive learning tasks.

        Each item follows the data structure as the general collate function but
        with each item having the following version specific attributes:
        - hard_y: (torch.Tensor) tensor of labels of the polar opposite
            communities
        - y: (torch.Tensor) tensor of labels of which topic the community
            belongs to
        """
        # TODO(liamhebert): Update docstring

        out = {
            key: torch.cat([item.y[key] for item in batch]).flatten()
            for key in [
                ContrastiveLabels.Ys,
                ContrastiveLabels.HardYs,
            ]
        }
        out[ContrastiveLabels.YMask] = torch.cat(
            [item.y_mask for item in batch]
        ).flatten()

        for key in [
            ContrastiveLabels.HardYs,
            ContrastiveLabels.Ys,
            ContrastiveLabels.YMask,
        ]:
            assert out[key].shape == (
                batch_size,
            ), f"{key}: {out[key].shape} != {(batch_size,)}"
        return out


class NodeBatchedDataDataset(CollatedDataset):
    """
    Dataset with Node learning specific collate function.
    """

    def label_collate_fn(
        self, batch: list[Data], batch_size: int, max_nodes: int
    ) -> dict[str, torch.Tensor]:
        """Collate function specific to node prediction tasks.

        Each item follows the data structure as the general collate function but
        with each item having the following version specific attributes:
        - y_mask: (torch.Tensor) boolean tensor for each node in the graph
            indicating if it has a label
        - y: (torch.Tensor) tensor of labels for each node in the graph.
            If a node does not have a label, it is padded with 0
        """
        # TODO(liamhebert): Update docstring

        out: dict[str, torch.Tensor] = {
            Labels.Ys: torch.cat(
                [
                    collator_utils.pad_1d_unsqueeze(
                        item.y[Labels.Ys], max_nodes, -1, False
                    )
                    for item in batch
                ]
            ),
            Labels.YMask: torch.cat(
                [
                    collator_utils.pad_1d_unsqueeze(
                        item.y_mask, max_nodes, False, False
                    )
                    for item in batch
                ]
            ),
        }

        for key in [Labels.Ys, Labels.YMask]:
            assert out[key].shape == (
                batch_size,
                max_nodes,
            ), f"{key}: {out[key].shape} != {(batch_size, max_nodes)}"

        return out
