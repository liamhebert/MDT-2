from abc import ABC, abstractmethod
import copy
from glob import glob
import orjson
import os
import pprint
from typing import Any, Dict, List

from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from torch_geometric import utils as pyg_utils
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoTokenizer
from joblib import Parallel, delayed
from data.types import Labels
import tasks.dataset_utils as dut
from utils.pylogger import RankedLogger
import h5py
import numpy as np
from itertools import chain
import random

log = RankedLogger(__name__)


class TaskDataset(Dataset, ABC):
    """Base class for task-specific datasets storing data in HDF5 format."""

    split_graphs: bool = False
    raw_graph_path: str | list[str]
    output_graph_path: str
    image_tokenizer_key: str
    text_tokenizer_key: str
    max_distance_length: int

    train_size: int | float | None
    valid_size: int | float | None
    test_size: int | float | None

    spatial_pos_max: int = 100

    _splits: dict[str, list[int]] | None
    _hdf5_file: h5py.File | None = None
    _hdf5_filename: str | None = None

    _splits = None
    _flattened_data = None
    _graph_sizes = None

    _idx_mapping = None

    # This tag will be used to differentiate between datasets
    tag: str = "default_dataset"

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
        debug: int | bool | None = None,
        group_size: int = 1,
        skip_invalid_label_graphs: bool = False,
        force_tag: str | None = None,
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
        if force_tag:
            log.warning(f"Overriding tag {self.tag=} with {force_tag=}")
            self.tag = force_tag

        self.image_tokenizer_key = image_tokenizer_key
        self.text_config = text_config or {
            "text_model_name": "bert-base-uncased",
            "has_token_type_ids": True,
            "add_position_ids": False,
            "max_length": None,
        }
        if text_config is None:
            log.warning("Using default text config.")

        self._image_tokenizer = None
        self._text_tokenizer = None
        self.split_graphs = split_graphs
        self.strict = strict
        self.max_distance_length = max_distance_length

        self.train_size = train_size
        self.valid_size = valid_size
        self.test_size = test_size
        self.split_seed = split_seed

        self.group_size = group_size

        self.force_reload = force_reload
        self.debug = debug
        is_prod = os.getenv("IS_PROD", False) == "1"
        if is_prod:
            log.warning("Force skipping invalid graphs in PROD mode")
            self._skip_invalid_label_graphs = True
        else:
            if skip_invalid_label_graphs:
                log.warning(
                    "Skipping invalid label graphs in non-PROD mode. Make sure"
                    " you know what you are doing!"
                )
            self._skip_invalid_label_graphs = skip_invalid_label_graphs

        if debug is True:
            self.debug = 10

        if self.debug:
            self.output_graph_path += "_debug"
            self.force_reload = True

        os.makedirs(self.output_graph_path, exist_ok=True)

        self._hdf5_filename = os.path.join(
            self.output_graph_path, self.tag + "-processed_graphs.hdf5"
        )
        log.info(f"Output graph path: {self._hdf5_filename}")

    @property
    def data_splits(self) -> dict[str, list[int]]:
        if self._splits is not None:
            return self._splits

        assert self.raw_paths, "No raw paths found"
        dataset_splits = {}
        dataset_file = self.hdf5_file

        for path in self.raw_paths:
            dataset_name = os.path.basename(path).removesuffix("-data.json")
            if dataset_name not in dataset_file:
                log.warning(
                    f"Dataset {dataset_name} not found in hdf5 dataset."
                    " Skipping."
                )
                continue

            split_path = path.removesuffix("-data.json") + "-split.json"
            current_dataset_splits = {
                "train_idx": [],
                "valid_idx": [],
                "test_idx": [],
            }
            if self._idx_mapping is None:
                self._dumb_idx_mapping()

            dataset_mapping = self._idx_mapping[dataset_name]

            if os.path.exists(split_path) and not self.debug:
                log.info(f"Loading splits from {split_path=}")
                with open(split_path, "r") as f:
                    loaded_splits = orjson.loads(f.read())
                    for key in ["train_idx", "test_idx"]:
                        current_dataset_splits[key] = loaded_splits[key]
                    current_dataset_splits["valid_idx"] = loaded_splits.get(
                        "valid_idx", loaded_splits["test_idx"]
                    )  # Use test if valid not present

                for key, val in current_dataset_splits.items():
                    corrected_paths = []
                    for idx in val:
                        if idx not in dataset_mapping:
                            log.warning(
                                f"Index {idx} not found in {dataset_name=}"
                            )
                            log.warning(f"current_dataset_splits: {val=}")
                            log.warning(f"dataset_mapping: {dataset_mapping=}")
                        corrected_paths += dataset_mapping[idx]
                    assert (
                        not val
                    ) or corrected_paths, f"No corrected paths for {key=}"
                    current_dataset_splits[key] = corrected_paths
            else:
                if self.debug:
                    log.warning("Forcing auto split in debug mode")
                log.info(
                    f"Generating splits: {self.train_size=},"
                    f" {self.valid_size=}, {self.test_size=}"
                )
                all_indices_for_dataset = list(chain(*dataset_mapping.values()))

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
                    log.warning("Using test set for validation")
                    valid_idx = test_valid_idx
                    test_idx = test_valid_idx
                else:
                    assert self.test_size and self.valid_size
                    valid_idx, test_idx = train_test_split(
                        test_valid_idx,
                        test_size=self.test_size
                        / (self.test_size + self.valid_size),
                        random_state=self.split_seed,
                    )

                current_dataset_splits["train_idx"] = train_idx
                current_dataset_splits["valid_idx"] = valid_idx
                current_dataset_splits["test_idx"] = test_idx

            def group_indices(
                indices: list[str],
            ) -> list[tuple[str, list[str]]]:
                # Group sets of indices such that each group consists of unique
                # indices only.

                groups = []
                current_group = set()
                remaining_ids = list(indices)  # Create a copy to modify
                random.shuffle(remaining_ids)
                total_dups = 0
                while remaining_ids:
                    if len(current_group) < self.group_size:
                        item = remaining_ids.pop(0)  # Take the first item
                        if item in current_group:
                            total_dups += 1
                        current_group.add(item)
                    else:
                        groups.append((dataset_name, list(current_group)))
                        current_group = set()  # Start a new group

                if len(current_group) == self.group_size:
                    groups.append((dataset_name, list(current_group)))
                return groups

            # TODO(liamhebert): This only groups once, meaning graphs will
            # always have the same positive. Would be nice to recreate this
            # at the end of every epoch maybe?
            current_dataset_splits = {
                key: group_indices(val)
                for key, val in current_dataset_splits.items()
            }

            dataset_splits[dataset_name] = current_dataset_splits

        dataset_sizes = {
            dataset: {split: len(val) for split, val in split_dict.items()}
            for dataset, split_dict in dataset_splits.items()
        }
        log.info(f"Dataset sizes: {pprint.pformat(dataset_sizes)}")

        self._splits = {"train_idx": [], "valid_idx": [], "test_idx": []}
        for split_dict in dataset_splits.values():
            for split in ["train_idx", "valid_idx", "test_idx"]:
                self._splits[split] += split_dict[split]

        self._flattened_data: List[tuple[str, list[str]]] = list(
            chain(
                self._splits["train_idx"],
                self._splits["valid_idx"],
                self._splits["test_idx"],
            )
        )

        offset_valid = len(self._splits["train_idx"])
        offset_test = len(self._splits["valid_idx"]) + offset_valid

        self._splits["train_idx"] = list(range(len(self._splits["train_idx"])))
        self._splits["valid_idx"] = [
            x + offset_valid for x in range(len(self._splits["valid_idx"]))
        ]

        self._splits["test_idx"] = [
            x + offset_test for x in range(len(self._splits["test_idx"]))
        ]

        random.shuffle(self._splits["train_idx"])
        random.shuffle(self._splits["valid_idx"])
        random.shuffle(self._splits["test_idx"])

        return self._splits

    @property
    def train_idx(self) -> list[int]:
        return self.data_splits["train_idx"]

    @property
    def valid_idx(self) -> list[int]:
        return self.data_splits["valid_idx"]

    @property
    def test_idx(self) -> list[int]:
        return self.data_splits["test_idx"]

    @property
    def raw_paths(self) -> list[str]:
        paths = (
            self.raw_graph_path
            if isinstance(self.raw_graph_path, list)
            else [self.raw_graph_path]
        )
        final_paths = []
        for path in paths:
            path = os.path.join(self.root, path)
            if not path.endswith("*.json") and ".json" not in path:
                path += "*.json"
            elif ".json" in path and not path.endswith("*.json"):
                raise ValueError(
                    "Raw path globs should end with *.json or not contain"
                    f" .json. Got {path=}"
                )

            found_paths = glob(os.path.expandvars(path))
            found_data_paths = [p for p in found_paths if "-data.json" in p]
            assert found_data_paths, f"No files found for {path=}"
            final_paths.extend(found_data_paths)
        return final_paths

    def __len__(self) -> int:
        total_size = 0
        for dataset in self.hdf5_file.keys():
            total_size += len(self.hdf5_file[dataset].keys())
        return total_size

    @property
    def text_tokenizer(self) -> AutoTokenizer:
        if self._text_tokenizer is None:
            self._text_tokenizer = AutoTokenizer.from_pretrained(
                self.text_config["text_model_name"],
                clean_up_tokenization_spaces=True,
                use_fast=True,
            )
        return self._text_tokenizer

    @property
    def image_tokenizer(self) -> AutoImageProcessor:
        if self._image_tokenizer is None:
            self._image_tokenizer = AutoImageProcessor.from_pretrained(
                self.image_tokenizer_key, use_fast=True
            )
        return self._image_tokenizer

    @property
    def has_node_labels(self) -> bool:
        return False

    @property
    def has_graph_labels(self) -> bool:
        return False

    @abstractmethod
    def retrieve_label(self, data: dict) -> dict[str, bool | int]: ...

    def process(self):
        if not (self.has_node_labels or self.has_graph_labels):
            raise ValueError(
                "Either has_node_labels or has_graph_labels must be True"
            )
        if self.has_node_labels and self.has_graph_labels:
            raise ValueError(
                "Only one of has_node_labels or has_graph_labels can be True"
            )

        if self.force_reload:
            log.warning("Force reloading data, deleting HDF5 file")
            os.system(f"rm -rf {self._hdf5_filename}")

        is_prod = os.getenv("IS_PROD", False) == "1" and not self.debug

        if is_prod:
            log.warning("In PROD mode, opening HDF5 file in read-only mode")
            self._hdf5_file = h5py.File(self._hdf5_filename, "r")
        else:
            log.warning("NOT in PROD mode, opening HDF5 file in write mode")
            self._hdf5_file = h5py.File(self._hdf5_filename, "w")

        def process_file(file):
            dataset_name = os.path.basename(file).removesuffix("-data.json")
            assert self._hdf5_file, "HDF5 file not open"
            if is_prod and dataset_name not in self._hdf5_file:
                log.warning(
                    f"Skipping empty group {dataset_name=} due to PROD mode"
                )
                return (dataset_name, None)

            dataset_group = self._hdf5_file.require_group(dataset_name)
            log.info(f"Processing file: {dataset_name}")
            if self.debug:
                log.warning(f"Debug mode: processing first {self.debug} graphs")
            index_mapping = {}
            had_errors = 0
            with open(file, "r") as f:
                for original_idx, line in tqdm(enumerate(f)):
                    if self.debug and original_idx == self.debug:
                        break
                    # TODO(liamhebert): This currently only works in graph-mode
                    # not for split nodes.
                    if f"graph_{original_idx}" in dataset_group:
                        index_mapping[original_idx] = [f"graph_{original_idx}"]
                        continue

                    # if (
                    #     processed_count in processed_indices
                    #     and self.processed_file_names_dataset_names[
                    #         processed_count
                    #     ]
                    #     == file_name
                    #     and self.processed_file_names_original_indices[
                    #         processed_count
                    #     ]
                    #     == original_idx
                    # ):
                    #     processed_count += 1
                    #     continue

                    json_data = orjson.loads(line)
                    try:
                        data = self.process_graph(json_data)
                    except ValueError as e:
                        # Graphs that raise label errors are skipped.
                        had_errors += 1
                        continue

                    if (
                        self.split_graphs and self.has_node_labels
                    ):  # Split graphs only for node-level tasks
                        index_mapping[original_idx] = []

                        mask = data["y"][Labels.Ys] != -100
                        for i in mask.nonzero(as_tuple=True)[0]:
                            new_data = copy.deepcopy(data)
                            ys = new_data["y"][Labels.Ys]
                            y_mask = torch.ones_like(ys).bool()
                            y_mask[i] = False
                            new_data["y"][Labels.Ys][y_mask] = -100
                            graph_name = self._write_graph_to_hdf5(
                                dataset_group,
                                new_data,
                                original_idx,
                                label_index=i,
                            )
                            index_mapping[original_idx].append(graph_name)
                    else:
                        graph_name = self._write_graph_to_hdf5(
                            dataset_group,
                            data,
                            original_idx,
                        )
                        index_mapping[original_idx] = [graph_name]

            log.info(
                f"Had {had_errors} errors and {len(index_mapping)} success"
                f" while processing {dataset_name}"
            )
            if len(index_mapping) == 0:
                log.warning(f"No graphs processed for {dataset_name}")
                del self._hdf5_file[dataset_name]
                return (dataset_name, None)
            return (dataset_name, index_mapping)

        dataset_mappings = Parallel(n_jobs=1)(
            delayed(process_file)(file)
            for file in tqdm(self.raw_paths, desc="Files")
        )
        assert dataset_mappings, "No data processed"

        # TODO(liamhebert): This is a side-effect of process, which is REALLY bad.
        # Ideally, there should be no side-effects and we should instead generate
        # this live.
        self._idx_mapping = {
            dataset_name: index_mapping
            for dataset_name, index_mapping in dataset_mappings
            if index_mapping  # drop empty mappings (skipped graphs)
        }
        self._hdf5_file = self._hdf5_file.close()

    def _dumb_idx_mapping(self):
        """
        This only works for no-split indexing, but we need this to recreate
        self._idx_mapping, which is only created in the master process.
        """
        self._idx_mapping = {}
        log.warning(
            "Creating dumb idx mapping. NOTE: This will break when using split"
            " graphs."
        )
        dataset_file = self.hdf5_file
        log.info(dataset_file.keys())
        for file in self.raw_paths:
            dataset_name = os.path.basename(file).removesuffix("-data.json")
            if dataset_name not in dataset_file:
                log.warning(f"Dataset {dataset_name} not found in HDF5 file")
                continue
            with open(file, "r") as f:
                index_mapping = {}
                for original_idx, line in enumerate(f):
                    if self.debug and original_idx == self.debug:
                        break
                    index_mapping[original_idx] = [f"graph_{original_idx}"]
            self._idx_mapping[dataset_name] = index_mapping

    def _write_graph_to_hdf5(
        self, dataset_group, data, original_index, label_index=None
    ) -> str:
        if label_index is not None:
            graph_name = f"graph_{original_index}_label_{label_index}"
        else:
            graph_name = f"graph_{original_index}"
        group = dataset_group.create_group(graph_name)
        group.attrs["size"] = len(data["out_degree"])

        text_group = group.create_group("text")
        for key, tensor in data["text"].items():
            text_group.create_dataset(key, data=tensor.numpy())
        y_group = group.create_group("y")
        for key, tensor in data["y"].items():
            y_group.create_dataset(key, data=tensor.numpy())

        group.create_dataset("image_mask", data=data["image_mask"].numpy())
        if data["images"] is not None:
            images_group = group.create_group("images")
            for key, tensor in data["images"].items():
                images_group.create_dataset(key, data=tensor.numpy())
        group.create_dataset("distance", data=data["distance"].numpy())
        group.create_dataset(
            "rotary_position", data=data["rotary_position"].numpy()
        )
        group.create_dataset("out_degree", data=data["out_degree"].numpy())

        return graph_name

    def flatten_graph(self, tree) -> dict:
        result = {
            "images": [],
            "distances": [],
            "rotary_position": [],
            "id": [],
            "parent_id": [],
            "is_root": [],
            "y": [],
            "text": [],
        }
        if self.has_graph_labels:
            label = self.retrieve_label(tree)
            if self._skip_invalid_label_graphs and all(
                x == -100 for x in label.values()
            ):
                raise ValueError("Invalid graph")
            result["y"].append(label)

        def traverse(node, parent_id=None):
            is_root = parent_id is None
            if is_root:
                parent_id = node["id"]

            if node["id"] not in result["id"]:
                node["images"] = node["images"][0] if node["images"] else None

                result["images"].append(node["images"])
                result["distances"].append(node["distances"])
                result["rotary_position"].append(node["rotary_position"])
                result["id"].append(node["id"])
                result["parent_id"].append(parent_id)
                result["is_root"].append(is_root)

                if self.has_node_labels:
                    result["y"].append(self.retrieve_label(node))

                text = (
                    f"Title: {node['title']}\nBody: {node['body']}"
                    if is_root
                    else f"Comment: {node['body']}"
                )
                result["text"].append(dut.clean_text(text))

            for child in node["tree"]:
                traverse(child, node["id"])

        traverse(tree)
        assert len(result["id"]) == len(
            result["distances"][0]
        ), "Distance mismatch"

        return result

    def process_graph(self, json_data):
        dut.compute_relative_distance(json_data)
        flattened_graph = self.flatten_graph(json_data)
        if len(flattened_graph["id"]) > 49:
            raise ValueError("Graph too large")

        if all(
            y == -100 for y in flattened_graph["y"]
        ):  # Check for empty labels
            if self.strict:
                raise ValueError(
                    f"No valid labels in graph: {json_data.get('id')=}"
                )
            else:
                log.warning(
                    "No valid labels, skipping graph:"
                    f" {json_data.get('id')=}, strict={self.strict=},"
                    f" split_graphs={self.split_graphs=}"
                )

        id_map = {node_id: i for i, node_id in enumerate(flattened_graph["id"])}
        mapped_parent_ids = [
            id_map[parent_id] for parent_id in flattened_graph["parent_id"]
        ]
        mapped_ids = list(
            range(len(flattened_graph["id"]))
        )  # Indices are already in order

        edges = torch.tensor([mapped_ids, mapped_parent_ids])

        tokenized_text = self.text_tokenizer(
            flattened_graph["text"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=self.text_config.get("max_length", None),
            return_attention_mask=True,
            return_token_type_ids=self.text_config.get(
                "has_token_type_ids", False
            ),
        )

        image_mask = []
        images = []
        for img in flattened_graph["images"]:
            if img:
                try:
                    img = Image.open(os.path.join(self.root, img)).convert(
                        "RGB"
                    )
                except Exception as e:
                    log.warning(
                        f"Error opening image {img} in {self.root}: {e}"
                    )
                    img = None

            if img is not None:
                images.append(img)
                image_mask.append(True)
            else:
                image_mask.append(False)

        image_mask = torch.tensor(image_mask, dtype=torch.bool)

        # image_mask = torch.tensor(
        #    [img is not None for img in flattened_graph["images"]],
        #    dtype=torch.bool,
        # )
        # images = [
        #    Image.open(os.path.join(self.root, img)).convert("RGB")
        #    for img in flattened_graph["images"]
        #    if img
        # ]
        tokenized_images = (
            self.image_tokenizer(images, return_tensors="pt")
            if images
            else None
        )

        combined_distance = [
            sorted(
                [(id_map[key], dist) for key, dist in distance.items()],
                key=lambda x: x[0],
            )
            for distance in flattened_graph["distances"]
        ]
        distance_tensor = torch.tensor(
            [
                [dist for _, dist in distances]
                for distances in combined_distance
            ],
            dtype=torch.uint8,
        )
        rotary_pos = torch.tensor(
            flattened_graph["rotary_position"], dtype=torch.uint8
        )
        degree = pyg_utils.degree(
            edges[1][1:], num_nodes=edges.shape[1], dtype=torch.long
        )  # Degree excluding self-loop

        y = {
            key: torch.tensor(
                [label[key] for label in flattened_graph["y"]], dtype=torch.int8
            )
            for key in flattened_graph["y"][0].keys()
        }

        return {
            "text": tokenized_text,
            "y": y,
            "image_mask": image_mask,
            "images": tokenized_images,
            "distance": distance_tensor,
            "rotary_position": rotary_pos,
            "out_degree": degree,
        }

    @property
    def hdf5_file(self) -> h5py.File:
        if self._hdf5_file is None or self._hdf5_file.id is None:
            self._hdf5_file = h5py.File(self._hdf5_filename, "r")
        return self._hdf5_file

    def get_sizes(self, indices: list[int]) -> list[int]:
        assert self._flattened_data, "Flattened data not loaded"
        mapped_indices = [self._flattened_data[i] for i in indices]

        result = []
        for dataset, graph_indices in tqdm(
            mapped_indices, desc="Getting sizes"
        ):
            total_size = 0
            for graph_index in graph_indices:
                group = self.hdf5_file.get(f"{dataset}/{graph_index}")
                assert isinstance(
                    group, h5py.Group
                ), f"Invalid group: {dataset}/{graph_index}"
                size = group.attrs["size"]
                assert isinstance(size, np.number), type(size)
                total_size = total_size + size

            result += [total_size]

        return result

    def _load_graph_from_hdf5(
        self, dataset_name: str, graph_index: str
    ) -> Dict[str, Any]:
        """Loads a processed graph from the HDF5 file as a dictionary."""
        group = self.hdf5_file.get(f"{dataset_name}/{graph_index}")
        assert isinstance(
            group, h5py.Group
        ), f"Invalid dataset group {group=}, {dataset_name=}, {graph_index=}"
        text_data = {
            key: torch.from_numpy(np_array[()])
            for key, np_array in group["text"].items()
        }
        y_data = {
            key: torch.from_numpy(np_array[()])
            for key, np_array in group["y"].items()
        }
        image_mask = torch.from_numpy(group["image_mask"][()])
        images_data = (
            {
                key: torch.from_numpy(np_array[()])
                for key, np_array in group["images"].items()
            }
            if "images" in group
            else None
        )
        distance = torch.from_numpy(group["distance"][()])
        rotary_position = torch.from_numpy(group["rotary_position"][()])
        out_degree = torch.from_numpy(group["out_degree"][()])

        return {  # Return a dictionary now, not Data object
            "text": text_data,
            "y": y_data,
            "image_mask": image_mask,
            "images": images_data,
            "distance": distance,
            "rotary_position": rotary_position,
            "out_degree": out_degree,
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Loads the processed graph from HDF5 at index as a dictionary."""

        if self._flattened_data is None:
            # This creates flattened data, kinda gross, but helps for tests.
            _ = self.data_splits

        assert self._flattened_data
        try:
            dataset_name, graph_indices = self._flattened_data[idx]
        except Exception as e:
            log.warning(f"Index {idx} not found in {self._flattened_data=}")
            raise e

        data = [
            self._load_graph_from_hdf5(dataset_name, graph_index)
            for graph_index in graph_indices
        ]

        return data
