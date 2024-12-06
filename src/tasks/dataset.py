from abc import ABC
from abc import abstractmethod
import copy
from glob import glob
import json
import math
import os
from typing import Any

from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from torch_geometric import utils as pyg_utils
from torch_geometric.data import Data
from tqdm import tqdm
from transformers import AutoImageProcessor
from transformers import AutoTokenizer
from transformers import BatchEncoding

from data.collator_utils import InputFeatures
import data.collator_utils as collator_utils
from data.types import ContrastiveLabels
from data.types import GraphFeatures
from data.types import ImageFeatures
from data.types import Labels
from data.types import TextFeatures
import tasks.dataset_utils as dut
from utils.pylogger import RankedLogger

log = RankedLogger(__name__)


class TaskDataset(Dataset, ABC):
    """Base class for all task-specific datasets.

    This class takes care of loading in and preparing raw data into a compatible
    format for downstream models. Subclasses should implement the retrieve_label
    function to extract the label from the raw data into the desired task label.
    """

    split_graphs: bool = False
    raw_graph_path: str | list[str]
    output_graph_path: str
    image_path: str
    image_tokenizer_key: str
    text_tokenizer_key: str
    max_distance_length: int

    split_path: str | None
    train_size: int | float
    valid_size: int | float
    test_size: int | float

    spatial_pos_max: int = 100

    def __init__(
        self,
        raw_graph_path: str | list[str],
        image_path: str,
        output_graph_path: str,
        root: str,
        split_path: str | None = None,
        train_size: int | float = 0.8,
        valid_size: int | float = 0.1,
        test_size: int | float = 0.1,
        split_seed: int = 42,
        image_tokenizer_key: str = "google/vit-base-patch16-224",
        text_tokenizer_key: str = "bert-base-uncased",
        split_graphs: bool = False,
        strict: bool = False,
        max_distance_length: int = 6,
        force_reload: bool = False,
    ):
        super().__init__()

        self.raw_graph_path = raw_graph_path
        self.image_path = image_path
        self.output_graph_path = output_graph_path
        self.root = root

        self.image_tokenizer_key = image_tokenizer_key
        self.text_tokenizer_key = text_tokenizer_key

        self._image_tokenizer = None
        self._text_tokenizer = None
        self.split_graphs = split_graphs
        self.strict = strict

        self.max_distance_length = max_distance_length
        # 0 is reserved for self connection

        self.train_size = train_size
        self.valid_size = valid_size
        self.test_size = test_size
        self.split_seed = split_seed

        self.force_reload = force_reload

        self._graph_paths: dict[int, str] = {}
        self._idx_mapping: dict[int, list[int]] = {}
        self._split_path = split_path
        self._splits: dict[str, list[int]] | None = None

    @property
    def data_splits(self) -> dict[str, list[int]]:
        if self._splits is not None:
            return self._splits

        if self._split_path is not None:
            log.info(f"Loading splits from split path {self._split_path=}")
            with open(self._split_path, "r") as f:
                self._splits = json.load(f)
                assert isinstance(self._splits, dict)

            keys = ["train_idx", "test_idx"]
            if "valid_idx" in self._splits:
                keys += ["valid_idx"]
            else:
                self._splits["valid_idx"] = []

            for key in keys:
                updated_paths = []
                for val in self._splits[key]:
                    if val in self._idx_mapping:
                        updated_paths += self._idx_mapping[val]
                self._splits[key] = updated_paths
        else:
            # If we don't have a split path, we will generate the splits using
            # a train_test_split
            log.info(
                (
                    "No split path provided, generating splits using"
                    "train_test_split of"
                    f"{self.train_size=}, {self.valid_size=}, {self.test_size=}"
                )
            )
            all_indices = list(range(self.__len__()))
            train_idx, test_valid_idx = train_test_split(
                all_indices,
                train_size=self.train_size,
                random_state=self.split_seed,
            )
            if self.valid_size != 0:
                valid_idx, test_idx = train_test_split(
                    test_valid_idx,
                    test_size=self.test_size
                    / (self.test_size + self.valid_size),
                    random_state=self.split_seed,
                )
            else:
                valid_idx = []
                test_idx = test_valid_idx

            self._splits = {
                "train_idx": train_idx,
                "valid_idx": valid_idx,
                "test_idx": test_idx,
            }
        return self._splits

    @property
    def train_idx(self) -> list[int]:
        """
        Data indices to use for training.
        """
        return self.data_splits["train_idx"]

    @property
    def valid_idx(self) -> list[int]:
        """
        Data indices to use for validation.
        """
        return self.data_splits["valid_idx"]

    @property
    def test_idx(self) -> list[int]:
        """
        Data indices to use for test.
        """
        return self.data_splits["test_idx"]

    @property
    def raw_paths(self) -> list[str]:
        if isinstance(self.raw_graph_path, str):
            paths = [self.raw_graph_path]
        else:
            paths = self.raw_graph_path
        final_paths = []
        for path in paths:
            path = self.root + "/" + path
            final_paths.extend(list(glob(os.path.expandvars(path))))
        return final_paths

    @property
    def processed_file_names(self) -> list[str]:
        path = os.path.expandvars(f"{self.output_graph_path}/processed")
        paths = []
        for i, graph in enumerate(glob(f"{path}/graph-*.pt")):
            # make sure graphs are appropriately sorted
            parts = os.path.basename(graph).split("-")
            assert len(parts) in [2, 3], f"Invalid graph path: {graph}, {parts}"
            if len(parts) == 3:
                idx = int(parts[1])
            else:
                idx = int(parts[1].removesuffix(".pt"))

            if idx not in self._idx_mapping:
                self._idx_mapping[idx] = []
            self._idx_mapping[idx] += [i]

            paths += [graph]
        return paths

    def __len__(self) -> int:
        return len(self.processed_file_names)

    @property
    def text_tokenizer(self) -> AutoTokenizer:
        if self._text_tokenizer is None:
            self._text_tokenizer = AutoTokenizer.from_pretrained(
                self.text_tokenizer_key
            )

        return self._text_tokenizer

    @property
    def image_tokenizer(self) -> AutoImageProcessor:
        if self._image_tokenizer is None:
            self._image_tokenizer = AutoImageProcessor.from_pretrained(
                self.image_tokenizer_key
            )

        return self._image_tokenizer

    @abstractmethod
    def retrieve_label(self, data: dict) -> tuple[int, bool]: ...

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
        batch_size, num_nodes, _ = collated_output["input_ids"].shape

        # TODO(liamhebert): Consider making this generic to collate extra
        # features beyond just the labels.
        collated_output["y"] = self.label_collate_fn(
            graphs, batch_size, num_nodes
        )
        return collated_output

    def process(self):
        """Processes all the raw graphs as pointed to by self.raw_paths into
        processed torch_geometric Data objects.

        If self.split_graphs is True, then graphs containing multiple labels
        will be split into multiple graphs, each containing a single label.

        All output graphs are saved to the self.output_graph_path/processed
        directory.
        """
        # TODO(liamhebert): Add support to process multiple files, and not just
        # the first one.
        processed_files = (
            list(glob(f"{self.output_graph_path}/processed/*.pt"))
            if not self.force_reload
            else []
        )
        corrected_files = []
        for file in processed_files:
            file_name = os.path.basename(file)
            file_name = file_name.removesuffix(".pt")
            # keep only the {file_name}-{idx} part
            file_name = "-".join(file_name.split("-")[:2])
            corrected_files.append(file_name)

        for file in self.raw_paths:
            file_name = os.path.basename(file)
            log.info(f"Processing file {file_name}")
            file_name = file_name.removesuffix(".json")
            with open(file, "r") as f:
                for idx, line in tqdm(enumerate(f)):
                    if f"graph-{file_name}-{idx}" in corrected_files:
                        continue
                    json_data = json.loads(line)
                    data = self.process_graph(json_data)

                    if self.split_graphs:
                        assert isinstance(data.y_mask, torch.BoolTensor)
                        for i in data.y_mask.nonzero(as_tuple=True)[0]:
                            new_data = copy.deepcopy(data)
                            new_data.y_mask = torch.zeros_like(data.y_mask)
                            new_data.y_mask[i] = 1

                            torch.save(
                                new_data,
                                f"{self.output_graph_path}/processed/graph-{file_name}-{idx}-{i}.pt",
                            )
                    else:
                        torch.save(
                            data,
                            f"{self.output_graph_path}/processed/graph-{file_name}-{idx}.pt",
                        )

    def flatten_graph(
        self, tree: dict[Any, Any]
    ) -> dict[str, list[bool | str | int | dict[str, int] | None]]:
        """Converts a nested graph into a flattened representation.

        This function will also retrieve the label and validity mask for each
        comment by calling the retrieve_label function with the "data" field
        of the node.

        Args:
            tree (dict[Any, Any]): The nested graph to flatten.

        Returns:
            dict[str, list[bool | str | int | None]]: The flattened graph,
                containing the following fields:
                - images (list[str | None]): The image paths for each node. If
                    there is no image for a given node, the value is None.
                - distances (list[dict[str, int]]): The distances between each
                    node and its parent.
                - id (list[str]): The unique identifier for each node.
                - parent_id (list[str]): The unique identifier for the parent of
                    each node.
                - is_root (list[bool]): Whether each node is the root or not.
                - y (list[int | str]): The label for each node.
                - y_mask (list[bool]): Whether the label is valid or not.
                - text (list[str]): The text for each node
        """
        result: dict[str, list[bool | str | int | dict[str, int] | None]] = {
            "images": [],
            "distances": [],
            "id": [],
            "parent_id": [],
            "is_root": [],
            "y": [],
            "y_mask": [],
            "text": [],
        }

        def traverse(node: dict[Any, Any], parent_id: str | None = None):
            """Recursively processes the data for a single node and adds it to
            the result dictionary.

            Args:
                node (dict[Any, Any]): The data for a single node, which may
                    contain children nodes in the "tree" field.
                parent_id (str | None, optional): The id of the parent node of
                    this node. If None, assumes that the node is the top-level
                    node and Defaults to None.
            """
            is_root = parent_id is None
            if is_root:
                parent_id = node["id"]

            assert parent_id is not None

            if len(node["images"]) != 0:
                node["images"] = node["images"][0]
            else:
                node["images"] = None

            result["images"].append(node["images"])
            result["distances"].append(node["distances"])
            result["id"].append(node["id"])
            result["parent_id"].append(parent_id)
            result["is_root"].append(is_root)

            y, y_mask = self.retrieve_label(node["data"])
            result["y"].append(y)
            # Whether it is a valid label or not
            result["y_mask"].append(y_mask)

            if is_root:
                text = (
                    f"Title: {node['data']['title']}\n"
                    f"Body: {node['data']['body']}"
                )
            else:
                text = f"Comment: {node['data']['body']}"
            result["text"].append(dut.clean_text(text))

            for child in node["tree"]:
                traverse(child, node["id"])

        traverse(tree)
        return result

    def process_graph(self, json_data: dict[str, Any]) -> Data:
        """Processes a single raw json graph into a processed torch_geometric
        Data object.

        Notably, this function will:
        - Tokenize the text and images
        - Compute the distance between all nodes in the graph
        - Flatten the graph into a [num_nodes, ...] format.

        Each nested dictionary in the json_data should contain the following
        - id (str | int): The unique identifier for the node
        - images (list[str]): A list of image paths nested under self.root. If
            there are no images, the list should be empty.
        - data (dict): A dictionary containing auxilary data for the node. The
            only required fields within data are "body", and if the root node,
            "title".

            The entire data dictionary will be passed to the
            self.retrieve_label function, which task_specific datasets should
            implement to extract the label and mask. See those datasets for
            additional requirements.
        - tree (list[dict]): A list of nested dictionaries representing the
            children of the current node. If there are no children, the list
            should be empty.

        After processing, the data graph will contain the following fields:
        - text (dict[str, Tensor]): The tokenized text of the graph as produced
            by the text_tokenizer, with keys "input_ids", "attention_mask", and
            "token_type_ids", each with shape [num_nodes, max_text_length].
        - edge_index (Tensor): The edge index of the graph, with shape
            [2, num_edges].
        - y (Tensor): The label for each node in the graph, with shape
            [num_nodes] or [] in the case of graph_level tasks.
        - y_mask (Tensor): A mask indicating whether labels are valid or not,
            with shape [num_nodes].
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

        Args:
            json_data (dict[str, Any]): The graph data to convert into a torch
                Data object.

        Raises:
            ValueError: If self.strict is True, we raise a ValueError if a graph
                has no valid labels. Otherwise, we log a warning and suppress.

        Returns:
            Data: The processed torch_geometric Data object, ready to be used
                by the model.
        """
        dut.compute_relative_distance(json_data)
        flattened_graph = self.flatten_graph(json_data)
        if all([x == False for x in flattened_graph["y_mask"]]):
            link_id = json_data["data"]["id"]
            if self.strict:
                raise ValueError("No valid labels in graph with {link_id=}")
            else:
                log.warning(
                    f"No valid labels in graph with {link_id=}, suppressing due "
                    f"to {self.strict=}. Note that if {self.split_graphs=} is"
                    " True, then the graph will be dropped."
                )

        id_map = {j: int(i) for i, j in enumerate(flattened_graph["id"])}
        mapped_ids = [id_map[x] for x in flattened_graph["id"]]
        mapped_parent_ids = [id_map[x] for x in flattened_graph["parent_id"]]

        flattened_edges = [list(x) for x in zip(mapped_ids, mapped_parent_ids)]
        edges = torch.tensor(flattened_edges).T

        tokenized_text: BatchEncoding = self.text_tokenizer(
            flattened_graph["text"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=256,
        )

        assert isinstance(flattened_graph["images"], list)
        assert all(
            isinstance(x, str) or x is None for x in flattened_graph["images"]
        )
        image_mask = torch.tensor(
            [x is not None for x in flattened_graph["images"]]
        )
        images = [
            Image.open(self.root + "/" + x).convert(mode="RGB")  # type: ignore
            for x in flattened_graph["images"]
            if x is not None
        ]
        if len(images) != 0:
            tokenized_images = self.image_tokenizer(images, return_tensors="pt")
        else:
            tokenized_images = None

        combined_distance = []
        for distance in flattened_graph["distances"]:
            assert isinstance(distance, dict)
            mapped_distance = [
                (id_map[key], dist) for key, dist in distance.items()
            ]
            mapped_distance = sorted(mapped_distance, key=lambda x: x[0])
            combined_distance.append([x[1] for x in mapped_distance])

        distance_tensor = torch.tensor(combined_distance)

        # Since each node only has one directional edge, we know that the number
        # of nodes is the same as the number of edges.
        num_nodes = edges.shape[1]
        # NOTE: Leaving the global token attn_bias to be added in the model code
        attn_bias = torch.zeros((num_nodes, num_nodes))

        data = Data(
            text=tokenized_text,
            edge_index=edges,
            y=torch.tensor(flattened_graph["y"]),
            y_mask=torch.tensor(flattened_graph["y_mask"]),
            image_mask=image_mask,
            images=tokenized_images,
            distance=distance_tensor,
            attn_bias=attn_bias,
        )
        # We do [1:] to remove the self-loop from the root node because it is it's
        # own parent.
        degree_out = pyg_utils.degree(
            data.edge_index[0][1:], num_nodes=num_nodes, dtype=torch.long
        )
        degree_in = pyg_utils.degree(
            data.edge_index[1][1:], num_nodes=num_nodes, dtype=torch.long
        )
        # We currently treat the graph as bidirectional
        degree = degree_out + degree_in
        data.in_degree = degree
        data.out_degree = degree
        distance_clamped = torch.clamp(
            distance_tensor,
            min=0,
            max=self.max_distance_length,
        )

        # Slightly faster then indexing a table, esp if we use a gpu.
        data.distance_index = (
            distance_clamped[..., 0] * self.max_distance_length
            + distance_clamped[..., 1]
        )
        assert data.distance_index.shape == (
            num_nodes,
            num_nodes,
        ), data.distance_index.shape
        return data

    def __getitem__(self, idx: int) -> Data:
        """Loads the processed graph from disk at the given index.

        Args:
            idx (int): The index of the graph to load.

        Raises:
            FileNotFoundError: If a graph does not exist, we raise an exception.

        Returns:
            Data: The processed torch_geometric Data object at the given index.
        """
        data = torch.load(self.processed_file_names[idx], weights_only=False)
        if data is None:
            raise FileNotFoundError("Loaded in None graph")
        return data


class ContrastiveTaskDataset(TaskDataset):
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

        out: dict[str, list[torch.Tensor]] = {
            ContrastiveLabels.Ys: [],
            ContrastiveLabels.HardYs: [],
            ContrastiveLabels.YMask: [],
        }
        for item in batch:
            out[ContrastiveLabels.Ys].append(item.y)
            out[ContrastiveLabels.HardYs].append(item.hard_y)
            out[ContrastiveLabels.YMask].append(item.y_mask)

        tensor_out: dict[str, torch.Tensor] = {
            key: torch.cat(val).flatten() for key, val in out.items()
        }

        assert tensor_out[ContrastiveLabels.Ys].shape == (
            batch_size,
        ), f"{tensor_out[ContrastiveLabels.Ys].shape} != {(batch_size,)}"

        return tensor_out


class NodeBatchedDataDataset(TaskDataset):
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

        out: dict[str, list[torch.Tensor]] = {
            Labels.Ys: [],
            Labels.YMask: [],
        }
        for item in batch:
            out[Labels.Ys].append(item.y)
            out[Labels.YMask].append(item.y_mask)

        tensor_out: dict[str, torch.Tensor] = {
            key: torch.cat(val).flatten() for key, val in out.items()
        }
        assert tensor_out[Labels.Ys].shape == (
            batch_size * max_nodes,
        ), f"{tensor_out[Labels.Ys].shape} != {(batch_size * max_nodes,)}"

        return tensor_out
