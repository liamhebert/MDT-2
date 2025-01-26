from abc import ABC
from abc import abstractmethod
import copy
from glob import glob
import json
import os
import pprint
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
from joblib import Parallel, delayed
from data.types import Labels, TextFeatures
import tasks.dataset_utils as dut
from utils.pylogger import RankedLogger

log = RankedLogger(__name__)


class TaskDataset(Dataset, ABC):
    """Base class for all task-specific datasets.

    This class takes care of loading in and preparing raw data into a compatible
    format for downstream models. Subclasses should implement the retrieve_label,
    has_graph_label and has_node_label functions to extract the label from the raw
    data into the desired task label.
    """

    split_graphs: bool = False
    raw_graph_path: str | list[str]
    output_graph_path: str
    image_path: str
    image_tokenizer_key: str
    text_tokenizer_key: str
    max_distance_length: int

    train_size: int | float | None
    valid_size: int | float | None
    test_size: int | float | None

    spatial_pos_max: int = 100

    max_text_length: int = 256

    _splits: dict[str, list[int]] | None

    def __init__(
        self,
        root: str,
        raw_graph_path: str | list[str],
        output_graph_path: str,
        train_size: int | float | None = 0.8,
        valid_size: int | float | None = 0.1,
        test_size: int | float | None = 0.1,
        split_seed: int = 42,
        image_tokenizer_key: str = "google/vit-base-patch16-224",
        text_config: dict[str, str | bool] | None = None,
        split_graphs: bool = False,
        strict: bool = False,
        max_distance_length: int = 10,
        force_reload: bool = False,
        debug: bool = False,
        max_text_length: int = 256,
    ):
        """Initializes the TaskDataset.

        Args:
            root (str): Root data directory for the dataset.
            raw_graph_path (str | list[str]): Path(s) to the raw graph data files.
            output_graph_path (str): Path to save the processed graph data files.
            train_size (int | float | None, optional): Proportion or number of
                samples to use for training. Defaults to 0.8.
            valid_size (int | float | None, optional): Proportion or number of
                samples to use for validation. Defaults to 0.1.
            test_size (int | float | None, optional): Proportion or number of
                samples to use for testing. Defaults to 0.1.
            split_seed (int, optional): Random seed for splitting the dataset.
                Defaults to 42.
            image_tokenizer_key (str, optional): Pretrained model key for image
                tokenization. Defaults to "google/vit-base-patch16-224".
            text_tokenizer_key (str, optional): Pretrained model key for text
                tokenization. Defaults to "bert-base-uncased".
            split_graphs (bool, optional): Whether to split graphs containing
                multiple labels into multiple graphs. Defaults to False.
            strict (bool, optional): Whether to raise an error if a graph has no
                valid labels. Defaults to False.
            max_distance_length (int, optional): Maximum distance length for
                relative distance computation. Values beyond that will be clamped
                to the maximum value. Defaults to 10.
            force_reload (bool, optional): Whether to force reloading and
                processing of raw data. Defaults to False.
            debug (bool, optional): Whether to enable debug mode, which only loads
                the first 10 graphs from each file. Defaults to False.
        """
        super().__init__()

        self.raw_graph_path = raw_graph_path
        self.output_graph_path = output_graph_path
        self.root = root

        self.image_tokenizer_key = image_tokenizer_key
        if text_config is None:
            log.warning(
                "No text config provided, using default bert-base-uncased. "
                "If this is prod, you should fix this!!"
            )
            text_config = {
                "text_model_name": "bert-base-uncased",
                "has_token_type_ids": True,
                "add_position_ids": False,
            }
        self.text_config = text_config

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
        self._splits = None

        self.force_reload = force_reload
        self.debug = debug

        self.max_text_length = max_text_length

    def get_idx_mapping(self) -> dict[str, dict[int, list[int]]]:
        idx_mapping: dict[str, dict[int, list[int]]] = {}
        # Assume that dataset does not have ".json" in it.
        graphs = self.processed_file_names
        # each graph path is of the form
        # {output_graph_path}/processed/{dataset}/graph-{idx}.pt
        for i, graph in enumerate(graphs):
            path_parts = graph.split("/")
            dataset = path_parts[-2]

            if dataset not in idx_mapping:
                idx_mapping[dataset] = {}

            # make sure graphs are appropriately sorted
            parts = os.path.basename(graph).split("-")

            assert len(parts) in [2, 3], f"Invalid graph path: {graph}, {parts}"

            idx = int(parts[1].removesuffix(".pt"))

            if idx not in idx_mapping[dataset]:
                idx_mapping[dataset][idx] = []
            idx_mapping[dataset][idx] += [i]

        return idx_mapping

    @property
    def data_splits(self) -> dict[str, list[int]]:
        if self._splits is not None:
            return self._splits

        assert self.raw_paths, "No raw paths found"

        dataset_splits: dict[str, dict[str, list[int]]] = {}
        dataset_mappings = self.get_idx_mapping()

        for path in self.raw_paths:
            current_dataset_splits: dict[str, list[int]] = {
                "train_idx": [],
                "valid_idx": [],
                "test_idx": [],
            }
            dataset_name = os.path.basename(path).removesuffix("-data.json")
            current_dataset_mapping = dataset_mappings[dataset_name]
            split_path = path.removesuffix("-data.json") + "-split.json"

            if os.path.exists(split_path) and not self.debug:
                log.info(
                    f"Dataset {dataset_name}: Loading splits from split "
                    f"path {split_path=}"
                )

                # First we load in the original values
                with open(split_path, "r") as f:
                    loaded_splits = json.load(f)
                    assert isinstance(loaded_splits, dict)
                    for key in ["train_idx", "test_idx"]:
                        current_dataset_splits[key] = loaded_splits[key]
                    if "valid_idx" not in loaded_splits:
                        log.warning(
                            f"Dataset {dataset_name}: No validation split found,"
                            "using test set for validation"
                        )
                        current_dataset_splits["valid_idx"] = loaded_splits[
                            "test_idx"
                        ]
                    else:
                        current_dataset_splits["valid_idx"] = loaded_splits[
                            "valid_idx"
                        ]

                # Then we correct them to match the flattened structure of the
                # dataset
                for key, val in current_dataset_splits.items():
                    corrected_paths = []
                    for idx in val:
                        if idx in current_dataset_mapping:
                            corrected_paths += current_dataset_mapping[idx]
                    # If we have valid paths, we should have a list of corrected
                    # paths.
                    assert (not val) or corrected_paths, (
                        f"No paths after correction found for {key=},"
                        f"{corrected_paths=}, {val=}, {current_dataset_mapping=}"
                    )
                    current_dataset_splits[key] = corrected_paths
            else:
                # If we don't have a split path, we will generate the splits using
                # a train_test_split
                if self.debug:
                    log.warning("Forcing auto split for debug mode")
                log.info(
                    (
                        f"Dataset {dataset_name}: No split path provided,"
                        "generating splits using train_test_split of"
                        f"{self.train_size=}, {self.valid_size=}, "
                        f"{self.test_size=}"
                    )
                )

                all_indices_for_dataset = []
                for indices in current_dataset_mapping.values():
                    all_indices_for_dataset += indices

                assert self.train_size is not None
                assert self.test_size is not None

                train_idx, test_valid_idx = train_test_split(
                    all_indices_for_dataset,
                    train_size=self.train_size,
                    random_state=self.split_seed,
                )
                if (
                    self.valid_size is None
                    or self.valid_size == 0
                    or len(test_valid_idx) < 2
                ):
                    log.warning(
                        f"Dataset {dataset_name}: Since {self.valid_size=}, "
                        "using test set for validation"
                    )
                    valid_idx = test_valid_idx
                    test_idx = test_valid_idx
                else:
                    valid_idx, test_idx = train_test_split(
                        test_valid_idx,
                        test_size=self.test_size
                        / (self.test_size + self.valid_size),
                        random_state=self.split_seed,
                    )

                current_dataset_splits["train_idx"] = train_idx
                current_dataset_splits["valid_idx"] = valid_idx
                current_dataset_splits["test_idx"] = test_idx

            dataset_splits[dataset_name] = current_dataset_splits

        # now that we are done, lets pretty print the size of each split
        dataset_sizes = {
            dataset: {split: len(val) for split, val in split_dict.items()}
            for dataset, split_dict in dataset_splits.items()
        }
        log.info("dataset sizes:")
        log.info(pprint.pformat(dataset_sizes))

        # Now we need to flatten dataset_splits into a canonical train, valid,
        # test split
        self._splits: dict[str, list[int]] = {
            "train_idx": [],
            "valid_idx": [],
            "test_idx": [],
        }
        for split_dict in dataset_splits.values():
            self._splits["train_idx"] += split_dict["train_idx"]
            self._splits["valid_idx"] += split_dict["valid_idx"]
            self._splits["test_idx"] += split_dict["test_idx"]

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
        """Returns the set of raw graph file paths in the dataset.

        Notably, this function will expand any glob patterns in the
        raw_graph_path. Each raw_graph_path should point to a single path such
        as "test1", of which there exists a "test1-data.json" and
        "test1-split.json" in the same folder.
        """
        # a raw_graph_path can be a single path -- "data"
        # or it can be a glob pattern -- "data-*"
        # or it can be a list of paths -- ["data", "data2"]
        #   and those paths can also be glob patterns.
        # it should not contain "-data.json" or "-split.json" as we will add
        # those ourselves.

        if isinstance(self.raw_graph_path, str):
            paths = [self.raw_graph_path]
        else:
            paths = self.raw_graph_path
        final_paths: list[str] = []

        for path in paths:
            path = self.root + "/" + path
            if (not path.endswith("*.json")) or ".json" not in path:
                path += "*.json"
            else:
                raise ValueError(
                    f"Raw path globs should end with *.json or not contain .json."
                    f"Got {path=}"
                )

            # This will include -data.json and -split.json
            found_paths = list(glob(os.path.expandvars(path)))

            found_data_paths = [x for x in found_paths if "-data.json" in x]
            for path in found_data_paths:
                assert os.path.exists(
                    path.removesuffix("-data.json") + "-split.json"
                ), f"Missing split file for {path=}"

            assert len(found_data_paths) > 0, f"No files found for {path=}"
            final_paths.extend(found_data_paths)

        return final_paths

    @property
    def processed_file_names(self) -> list[str]:
        """Computes the list of processed graph file names in the dataset."""
        path = os.path.expandvars(f"{self.output_graph_path}/processed")
        return list(glob(f"{path}/*/graph-*.pt"))

    def __len__(self) -> int:
        """Returns the number of processed graphs in the dataset."""
        return len(self.processed_file_names)

    @property
    def text_tokenizer(self) -> AutoTokenizer:
        """Returns the lazily initialized text tokenizer for the dataset."""
        if self._text_tokenizer is None:
            self._text_tokenizer = AutoTokenizer.from_pretrained(
                self.text_config["text_model_name"],
                clean_up_tokenization_spaces=True,
            )

        return self._text_tokenizer

    @property
    def image_tokenizer(self) -> AutoImageProcessor:
        """Returns the lazily initialized image tokenizer for the dataset."""
        if self._image_tokenizer is None:
            self._image_tokenizer = AutoImageProcessor.from_pretrained(
                self.image_tokenizer_key
            )

        return self._image_tokenizer

    @property
    def has_node_labels(self) -> bool:
        """Indicates whether the dataset has node labels.

        Unless overridden, False.
        """
        return False

    @property
    def has_graph_labels(self) -> bool:
        """Indicates whether the dataset has graph labels.

        Unless overridden, False.
        """
        return False

    @abstractmethod
    def retrieve_label(self, data: dict) -> dict[str, bool | int]:
        """Retrieves the label for a given node using metadata information.

        This function is called with the "data" field of the node, which contains
        whatever auxiliary information is available for the node.

        If has_node_labels is True, then this function is called on every node in
        a graph. Otherwise, if has_graph_labels is True, then this function is
        only called on the root node of the graph.

        Labels are returned as a dictionary which must contain "Ys" and "YMask"
        as keys, where Ys is the label and YMask is a boolean indicating whether
        the label is valid or not.

        Args:
            data (dict[str, Any]): The metadata of the comment.

        Returns:
            dict[str, bool | int]: The label information for the node, which must
                contain "Ys" and "YMask" as keys.
        """
        ...

    def process(self):
        """Processes all the raw graphs as pointed to by self.raw_paths into
        processed torch_geometric Data objects.

        If self.split_graphs is True, then graphs containing multiple labels
        will be split into multiple graphs, each containing a single label.

        All output graphs are saved to the self.output_graph_path/processed
        directory.
        """

        assert self.has_node_labels or self.has_graph_labels, (
            f"Either has_node_labels or has_graph_labels must be True - "
            f"{self.has_node_labels=} {self.has_graph_labels=}"
        )
        assert not (
            self.has_node_labels and self.has_graph_labels
        ), "Only one of has_node_labels or has_graph_labels can be True"

        if self.force_reload:
            log.warning("Force reloading data, deleting all processed data")
            os.system(f"rm -rf {self.output_graph_path}/processed")

        processed_folders = (
            list(glob(f"{self.output_graph_path}/processed/*"))
            if not self.force_reload
            else []
        )
        corrected_files = {}
        for folder in processed_folders:
            folder = os.path.basename(folder)
            corrected_files[folder] = []
            for file in glob(f"{self.output_graph_path}/processed/{folder}/*"):
                file_name = os.path.basename(file)
                file_name = file_name.removesuffix(".pt")
                # keep only the -{idx} part
                if len(file_name.split("-")) > 2:
                    file_name = "-".join(file_name.split("-")[:1])
                corrected_files[folder].append(file_name)

        def process_file(file):
            file_name = os.path.basename(file)
            log.info(f"Processing file {file_name}")
            if self.debug:
                log.warning(
                    "Debug mode enabled, only processing first 10 graphs"
                )
            file_name = file_name.removesuffix("-data.json")
            os.makedirs(
                f"{self.output_graph_path}/processed/{file_name}",
                exist_ok=True,
            )

            with open(file, "r") as f:
                for idx, line in tqdm(enumerate(f)):
                    if self.debug and idx == 10:
                        break

                    if (
                        file_name in corrected_files
                        and f"graph-{idx}" in corrected_files[file_name]
                    ):
                        continue
                    json_data = json.loads(line)
                    data = self.process_graph(json_data)

                    if self.split_graphs:
                        assert not self.has_graph_labels, (
                            "Trying to split graphs with graph labels,"
                            "this is pointless you silly goose!"
                        )
                        mask = data.y[Labels.Ys] != -100

                        for i in mask.nonzero(as_tuple=True)[0]:
                            new_data = copy.deepcopy(data)
                            # We negate all other labels
                            ys = new_data.y[Labels.Ys]
                            y_mask = torch.ones_like(ys).bool()
                            # But keep the one we want
                            y_mask[i] = False
                            new_data.y[Labels.Ys][y_mask] = -100

                            torch.save(
                                new_data,
                                f"{self.output_graph_path}/processed/{file_name}/"
                                f"graph-{idx}-{i}.pt",
                            )
                    else:
                        torch.save(
                            data,
                            f"{self.output_graph_path}/processed/{file_name}/"
                            f"graph-{idx}.pt",
                        )

        Parallel(n_jobs=8)(
            delayed(process_file)(file)
            for file in tqdm(self.raw_paths, desc="Files")
        )

    def flatten_graph(
        self, tree: dict[Any, Any]
    ) -> dict[str, list[bool | str | int | dict[str, int | bool] | None]]:
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
                - text (list[str]): The text for each node
        """
        result: dict[
            str, list[bool | str | int | dict[str, int | bool] | None]
        ] = {
            "images": [],
            "distances": [],
            "id": [],
            "parent_id": [],
            "is_root": [],
            "y": [],
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

            if node["id"] not in result["id"]:
                assert parent_id is not None

                if len(node["images"]) != 0:
                    node["images"] = node["images"][0]
                else:
                    node["images"] = None

                result["images"].append(node["images"])
                result["distances"].append(node["distances"])
                if node["id"] in result["id"]:
                    raise ValueError(
                        f"Duplicate id found {node['id']} \n {result=} \n new node \n {node=}"
                    )
                result["id"].append(node["id"])
                result["parent_id"].append(parent_id)
                result["is_root"].append(is_root)

                if self.has_node_labels:
                    label = self.retrieve_label(node["data"])
                    result["y"].append(label)

                if is_root:
                    text = (
                        f"Title: {node['data']['title']}\n"
                        f"Body: {node['data']['body']}"
                    )
                else:
                    text = f"Comment: {node['data']['body']}"
                result["text"].append(dut.clean_text(text))
            else:
                print("Duplicate id found, skipping ", node["id"])

            for child in node["tree"]:
                traverse(child, node["id"])

        traverse(tree)
        assert len(result["id"]) == len(
            result["distances"][0]
        ), f"Distance mismatch, {result=}, \n {set(result['id']) - set(result['distances'][0].keys())=}"

        if self.has_graph_labels:
            label = self.retrieve_label(tree["data"])
            result["y"].append(label)
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
        - data (dict): A dictionary containing auxiliary data for the node. The
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
        if all(x == -100 for x in flattened_graph["y"]):
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
            # max_length=self.max_text_length,
            return_attention_mask=True,
            return_token_type_ids=self.text_config["has_token_type_ids"],
        )
        for feature in [TextFeatures.InputIds, TextFeatures.AttentionMask] + (
            [TextFeatures.TokenTypeIds]
            if self.text_config["has_token_type_ids"]
            else []
        ):
            assert (
                feature in tokenized_text
            ), f"Missing {feature=}, {tokenized_text.keys()=}"

        if self.text_config["add_position_ids"]:
            tokenized_text[TextFeatures.PositionIds] = torch.tile(
                torch.arange(tokenized_text[TextFeatures.InputIds].shape[1]),
                (tokenized_text[TextFeatures.InputIds].shape[0], 1),
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

        assert distance_tensor.shape == (
            num_nodes,
            num_nodes,
            2,
        ), distance_tensor.shape

        # NOTE: Leaving the global token attn_bias to be added in the model code
        # TODO(liamhebert): Technically this is no-op since we don't ever set
        # dataset specific attn_biases. The only spot where we do is in the
        # collator to mask out nodes according to distance. We should probably
        # remove this.
        attn_bias = torch.zeros((num_nodes, num_nodes))

        assert all(isinstance(x, dict) for x in flattened_graph["y"])
        ys: list[dict[str, int | bool]] = flattened_graph["y"]  # type: ignore
        y = {key: torch.tensor([x[key] for x in ys]) for key in ys[0].keys()}

        distance_clamped = torch.clamp(
            distance_tensor,
            min=0,
            max=self.max_distance_length,
        )

        # Slightly faster then indexing a table, esp if we use a gpu.
        distance_index = (
            distance_clamped[..., 0] * self.max_distance_length
            + distance_clamped[..., 1]
        )

        data = Data(
            text=tokenized_text,
            edge_index=edges,
            y=y,
            image_mask=image_mask,
            images=tokenized_images,
            distance=distance_tensor,  # NOTE: this distance is not clamped
            distance_index=distance_index,
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
