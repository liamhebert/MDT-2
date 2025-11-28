"""Abstract base class for handling graph-structured data.

The `TaskDataset` class provides a comprehensive framework for processing,
storing, and accessing graph-based datasets, particularly for use in deep
learning models. It is designed to handle datasets where each sample is a graph
(e.g., a discussion tree), which may contain text, images, and structural
information.

Key functionalities of this module include:
- **Data Processing:** Reading raw graph data from JSON files, processing it
  into a structured format, and computing graph-specific features like
  relative distances and rotary positions.
- **Data Storage:** Efficiently storing and retrieving processed graphs using
  the HDF5 file format, which is well-suited for large, heterogeneous datasets.
- **Tokenization:** On-the-fly tokenization of text and images using pre-trained
  models from the `transformers` library.
- **Data Splitting:** Splitting the dataset into training, validation, and test
  sets, with support for both pre-defined splits and random splitting.
- **Label Handling:** Abstract methods for retrieving task-specific labels,
  allowing for both graph-level and node-level classification tasks.
- **Graph Grouping:** A mechanism to group related graphs, which is particularly
  useful for contrastive learning tasks.
- **Flexibility:** The class is designed to be extended for specific tasks by
  implementing abstract methods like `retrieve_label` and `group_hint`.

The module relies on several external libraries, including `torch`, `h5py`,
`numpy`, `scikit-learn`, and `transformers`, to provide its functionality.
"""

from abc import ABC, abstractmethod
import copy
from glob import glob
import orjson
import os
import pprint
from typing import Any, Dict, List, cast

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
    """An abstract base class for creating and managing graph-based datasets.

    This class provides a comprehensive framework for handling datasets composed
    of graphs, such as discussion trees or social networks. It manages the entire
    pipeline from raw data loading to providing processed, tokenized, and batched
    data suitable for training deep learning models. The data is stored
    efficiently in HDF5 files to handle large datasets that may not fit into
    memory.

    Subclasses must implement the `retrieve_label` method to define how labels
    are extracted from the raw data, and can optionally override `group_hint` to
    provide custom logic for grouping similar graphs (e.g., for contrastive
    learning).

    Attributes:
        raw_graph_path (str | list[str]): Path(s) to the raw graph data in JSON
            format.
        output_graph_path (str): Directory to save the processed HDF5 file.
        root (str): The root directory where data is stored.
        tag (str): A unique identifier for the dataset, used in the HDF5 filename.
        image_tokenizer_key (str): The Hugging Face model key for the image
            tokenizer.
        text_config (dict): Configuration for the text tokenizer.
        split_graphs (bool): If True, graphs with multiple node labels are split
            into separate graphs, one for each labeled node.
        strict (bool): If True, raises an error for graphs with no valid labels.
        max_distance_length (int): The maximum distance to consider for relative
            positional encoding.
        max_graph_size (int): The maximum number of nodes allowed in a graph.
        train_size (int | float | None): The proportion or number of samples for
            the training set.
        valid_size (int | float | None): The proportion or number of samples for
            the validation set.
        test_size (int | float | None): The proportion or number of samples for
            the test set.
        split_seed (int): The random seed for data splitting.
        group_size (int): The number of graphs to group together in each sample.
        force_reload (bool): If True, forces the reprocessing of raw data.
        debug (int | bool | None): If set, runs in debug mode, processing only a
            small subset of the data.
    """

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
    max_graph_size: int = 49

    _splits: dict[str, list[list[tuple[str, int]]]] | None = None
    _hdf5_file: h5py.File | None = None
    _hdf5_filename: str | None = None

    _flattened_data: (
        list[list[tuple[str, str]]] | list[list[tuple[str, int]]] | None
    ) = None
    _graph_sizes = None

    _idx_mapping: dict[str, Any] | None = None

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
        max_graph_size: int = 49,
    ):
        """Initializes the TaskDataset.

        This constructor sets up the configuration for the dataset, including
        paths, tokenizers, data splitting parameters, and other processing
        options.

        Args:
            root (str): Root data directory for the dataset.
            raw_graph_path (str | list[str]): Path(s) to the raw graph data files.
            output_graph_path (str): Path to save the processed graph data files.
            train_size (int | float | None, optional): Proportion or number of
                samples for training. Defaults to 0.8.
            valid_size (int | float | None, optional): Proportion or number of
                samples to use for validation. Defaults to 0.1.
            test_size (int | float | None, optional): Proportion or number of
                samples to use for testing. Defaults to 0.1.
            split_seed (int, optional): Random seed for splitting the dataset.
                Defaults to 42.
            image_tokenizer_key (str, optional): Pretrained model key for image
                tokenization. Defaults to "google/vit-base-patch16-224".
            text_config (dict, optional): Configuration for the text tokenizer,
                including model name and other parameters.
            split_graphs (bool, optional): Whether to split graphs containing
                multiple labels into multiple graphs. Defaults to False.
            strict (bool, optional): Whether to raise an error if a graph has no
                valid labels. Defaults to False.
            max_distance_length (int, optional): Maximum distance for relative
                distance computation. Values beyond this are clamped.
                Defaults to 10.
            force_reload (bool, optional): Whether to force reloading and
                processing of raw data. Defaults to False.
            debug (int | bool | None, optional): If set, enables debug mode,
                which processes a small subset of data. Defaults to None.
            group_size (int, optional): The number of graphs to group together
                in a single sample. Defaults to 1.
            skip_invalid_label_graphs (bool, optional): If True, graphs with no
                valid labels are skipped. Defaults to False.
            force_tag (str | None, optional): If provided, overrides the default
                dataset tag. Defaults to None.
            max_graph_size (int, optional): The maximum number of nodes allowed
                in a graph. Defaults to 49.
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
        self.max_graph_size = max_graph_size

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
    def data_splits(self) -> dict[str, list[list[tuple[str, int]]]]:
        """Gets the train, validation, and test splits for the dataset.

        If the splits have not been computed yet, this method will:
        1.  Load or generate an index mapping from original file indices to
            HDF5 graph names.
        2.  Check for pre-defined split files. If they exist, it loads them.
        3.  If no split files are found, it generates new splits based on the
            `train_size`, `valid_size`, and `test_size` parameters.
        4.  Groups the indices based on the `group_hint` and `group_size` for
            use in tasks like contrastive learning.
        5.  Flattens the grouped data and creates final index lists for train,
            validation, and test sets.

        The splits are cached after the first computation.

        Returns:
            A dictionary containing the indices for each split:
            `{"train_idx": [...], "valid_idx": [...], "test_idx": [...]}`.
        """
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

            assert self._idx_mapping is not None
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
                                f"Index {idx} not found in"
                                f" {dataset_name=} {len(dataset_mapping)=}"
                            )
                            # log.warning(f"current_dataset_splits: {val=}")
                            # log.warning(f"dataset_mapping: {dataset_mapping=}")
                        else:
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
                log.info(
                    f"Processing {dataset_name=} with"
                    f" {len(all_indices_for_dataset)=}"
                )
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

            # TODO(liamhebert): This only groups once, meaning graphs will
            # always have the same positive. Would be nice to recreate this
            # at the end of every epoch maybe?

            dataset_splits[dataset_name] = current_dataset_splits

        dataset_sizes = {
            dataset: {split: len(val) for split, val in split_dict.items()}
            for dataset, split_dict in dataset_splits.items()
        }
        log.info(f"Dataset sizes: {pprint.pformat(dataset_sizes)}")

        pre_grouped_splits = {"train_idx": {}, "valid_idx": {}, "test_idx": {}}
        for dataset, split_dict in dataset_splits.items():
            group_hint = self.group_hint(dataset)
            for split in ["train_idx", "valid_idx", "test_idx"]:
                if group_hint not in pre_grouped_splits[split]:
                    pre_grouped_splits[split][group_hint] = []
                pre_grouped_splits[split][group_hint] += [
                    (dataset, x) for x in split_dict[split]
                ]

        def group_indices(
            indices: list[tuple[str, str]],
        ) -> list[list[tuple[str, str]]]:
            # Group sets of indices such that each group consists of unique
            # indices only.

            groups = []
            current_group = set()
            remaining_ids = list(indices)  # Create a copy to modify
            random.shuffle(remaining_ids)
            total_dups = 0
            while remaining_ids:
                if len(current_group) < self.group_size:
                    dataset, item = remaining_ids.pop(0)  # Take the first item
                    if item in current_group:
                        total_dups += 1
                    current_group.add((dataset, item))
                else:
                    groups.append(list(current_group))
                    current_group = set()  # Start a new group

            if len(current_group) == self.group_size:
                groups.append(list(current_group))
            return groups

        self._splits = {}
        for split, groups in pre_grouped_splits.items():
            self._splits[split] = []
            for group in groups.values():
                grouped_idx = group_indices(group)
                self._splits[split].extend(grouped_idx)

        self._flattened_data = list(
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

    def group_hint(self, dataset_name: str) -> int:
        """Returns an integer value indicating how to partition the data into
        groups of similar graphs. Useful for contrastive learning tasks, where
        we can group positive graphs together.

        In the base implementation, all graphs are considered part of the same
        group (group 0). This method should be overridden in subclasses to
        provide task-specific grouping logic. For example, in a classification
        task, this could return the class label, so that graphs with the same
        label are grouped together.

        Args:
            dataset_name (str): The name of the dataset being processed.

        Returns:
            int: An integer representing the group ID for the dataset.
        """
        return 0

    @property
    def train_idx(self) -> list[int]:
        """Returns the list of indices for the training set."""
        return self.data_splits["train_idx"]

    @property
    def valid_idx(self) -> list[int]:
        """Returns the list of indices for the validation set."""
        return self.data_splits["valid_idx"]

    @property
    def test_idx(self) -> list[int]:
        """Returns the list of indices for the test set."""
        return self.data_splits["test_idx"]

    @property
    def raw_paths(self) -> list[str]:
        """Finds and returns the absolute paths to the raw data files.

        This property handles glob patterns in the `raw_graph_path` to find all
        matching JSON data files. It ensures that only files ending with
        "-data.json" are included.

        Returns:
            A list of absolute file paths to the raw data files.

        Raises:
            ValueError: If the provided path glob is malformed.
            AssertionError: If no files are found for the given path.
        """
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
        """Returns the total number of individual graphs in the HDF5 file."""
        total_size = 0
        for dataset in self.hdf5_file.keys():
            total_size += len(cast(h5py.Group, self.hdf5_file[dataset]).keys())
        return total_size

    @property
    def text_tokenizer(self) -> AutoTokenizer:
        """Provides a Hugging Face `AutoTokenizer` instance.

        The tokenizer is initialized lazily on the first access using the
        configuration specified in `self.text_config`.

        Returns:
            An instance of `AutoTokenizer`.
        """
        if self._text_tokenizer is None:
            self._text_tokenizer = AutoTokenizer.from_pretrained(
                self.text_config["text_model_name"],
                clean_up_tokenization_spaces=True,
                use_fast=True,
            )
        return self._text_tokenizer

    @property
    def image_tokenizer(self) -> AutoImageProcessor:
        """Provides a Hugging Face `AutoImageProcessor` instance.

        The image processor is initialized lazily on the first access using the
        key specified in `self.image_tokenizer_key`.

        Returns:
            An instance of `AutoImageProcessor`.
        """
        if self._image_tokenizer is None:
            self._image_tokenizer = AutoImageProcessor.from_pretrained(
                self.image_tokenizer_key, use_fast=True
            )
        return self._image_tokenizer

    @property
    def has_node_labels(self) -> bool:
        """Indicates whether the dataset has node-level labels.

        Subclasses should override this property if they support node labels.
        """
        return False

    @property
    def has_graph_labels(self) -> bool:
        """Indicates whether the dataset has graph-level labels.

        Subclasses should override this property if they support graph labels.
        """
        return False

    @abstractmethod
    def retrieve_label(self, data: dict) -> dict[str, bool | int]:
        """Abstract method to retrieve labels from a raw data dictionary.

        Subclasses must implement this method to define how task-specific
        labels are extracted from a single data point (a node or a graph).

        Args:
            data (dict): A dictionary representing a single raw data point
                (e.g., a node in a graph).

        Returns:
            A dictionary containing the extracted labels. The keys should
            correspond to the label names (e.g., "is_hateful") and the values
            should be the label values.
        """
        ...

    def process(self):
        """Processes the raw data and saves it to an HDF5 file.

        This is the main data processing method. It orchestrates the entire
        workflow:
        1.  Checks if data needs to be reprocessed (`force_reload`).
        2.  Opens the HDF5 file in the appropriate mode (write or read-only).
        3.  Iterates through the raw data files and processes them in parallel.
        4.  For each graph, it calls `process_graph` to perform tokenization,
            feature extraction, and structuring.
        5.  Handles graph splitting for node-level tasks if `split_graphs`
            is True.
        6.  Writes the processed graphs to the HDF5 file using
            `_write_graph_to_hdf5`.
        7.  Creates an index mapping from original file indices to the names of
            the stored graphs in the HDF5 file.

        Raises:
            ValueError: If neither `has_node_labels` nor `has_graph_labels` is
                True, or if both are True.
        """
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
            if self.group_hint(dataset_name) < 0:
                log.warning(
                    f"Skipping dataset {dataset_name} due to group hint < 0"
                )
                return (dataset_name, None)
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
                    test_graph_name = (
                        f"graph_{original_idx}"
                        if self.has_graph_labels or (not self.split_graphs)
                        else f"graph_{original_idx}_label_0"
                    )
                    if test_graph_name in dataset_group:
                        if self.has_graph_labels or (not self.split_graphs):
                            index_mapping[original_idx] = [test_graph_name]
                        else:
                            i = 0
                            index_mapping[original_idx] = []
                            while True:
                                if (
                                    f"graph_{original_idx}_label_{i}"
                                    in dataset_group
                                ):
                                    index_mapping[original_idx].append(
                                        f"graph_{original_idx}_label_{i}"
                                    )
                                    i += 1
                                else:
                                    break
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
                    except Exception as e:
                        log.info(e)
                        # Graphs that raise label errors are skipped.
                        had_errors += 1
                        continue

                    if (
                        self.split_graphs and self.has_node_labels
                    ):  # Split graphs only for node-level tasks
                        index_mapping[original_idx] = []

                        mask: np.ndarray = data["y"][Labels.Ys] != -100
                        for idx, label_index in enumerate(mask.nonzero()[0]):
                            new_data = copy.deepcopy(data)
                            ys = new_data["y"][Labels.Ys]
                            y_mask = np.ones_like(ys).astype(bool)
                            y_mask[label_index] = False
                            new_data["y"][Labels.Ys][y_mask] = -100

                            graph_name = self._write_graph_to_hdf5(
                                dataset_group,
                                new_data,
                                original_idx,
                                label_index=idx,
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
        """Creates a simplified index mapping for the dataset.

        This method is a fallback for when the primary index mapping (which is
        typically created in the main process during data processing) is not
        available, for example, in a worker process. It works by iterating
        through the raw files and the HDF5 file to reconstruct a mapping from
        the original line number in the raw file to the corresponding graph
        name(s) in the HDF5 file.

        Note: This method is less efficient than the primary mapping creation
        and is intended for specific use cases where the mapping needs to be
        recreated. It assumes a direct correspondence that might not hold for
        complex data processing scenarios.
        """
        self._idx_mapping = {}
        log.warning("Creating dumb idx mapping.")
        dataset_file = self.hdf5_file
        log.info(dataset_file.keys())
        # TODO(liamhebert): Instead of recreating indexing mapping, we should
        # just save it in the hdf5 file, and then retrieve it directly every time.
        for file in self.raw_paths:
            dataset_name = os.path.basename(file).removesuffix("-data.json")
            if dataset_name not in dataset_file:
                log.warning(f"Dataset {dataset_name} not found in HDF5 file")
                continue
            dataset_group = dataset_file.require_group(dataset_name)
            with open(file, "r") as f:
                index_mapping = {}
                for original_idx, line in enumerate(f):
                    if self.debug and original_idx == self.debug:
                        break
                    test_graph_name = (
                        f"graph_{original_idx}"
                        if self.has_graph_labels or (not self.split_graphs)
                        else f"graph_{original_idx}_label_0"
                    )
                    if test_graph_name in dataset_group:
                        if self.has_graph_labels or (not self.split_graphs):
                            index_mapping[original_idx] = [test_graph_name]
                        else:
                            i = 0
                            index_mapping[original_idx] = []
                            while True:
                                if (
                                    f"graph_{original_idx}_label_{i}"
                                    in dataset_group
                                ):
                                    index_mapping[original_idx].append(
                                        f"graph_{original_idx}_label_{i}"
                                    )
                                    i += 1
                                else:
                                    break

            self._idx_mapping[dataset_name] = index_mapping

    def flatten_graph(self, tree: dict) -> dict:
        """Flattens a nested graph structure into a dictionary of lists.

        This method traverses a tree-like graph (represented as a nested
        dictionary) and extracts the attributes of each node into a flat
        structure. Each key in the output dictionary corresponds to a node
        attribute (e.g., "images", "text", "id"), and the value is a list
        containing the attribute values for all nodes in the graph.

        Args:
            tree (dict): The root of the graph/tree, represented as a nested
                dictionary.

        Returns:
            A dictionary where keys are attribute names and values are lists
            of those attributes for each node in the graph.

        Raises:
            ValueError: If the graph is invalid (e.g., has no valid labels when
                `_skip_invalid_label_graphs` is True).
        """
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

    def process_graph(self, json_data: dict) -> dict:
        """Processes a single raw graph from a JSON object.

        This method takes a raw graph represented as a dictionary, performs
        several processing steps, and returns a dictionary of tensors and arrays
        ready to be stored in the HDF5 file.

        The processing steps include:
        1.  Computing relative distances between nodes.
        2.  Flattening the graph structure using `flatten_graph`.
        3.  Tokenizing the text content of the nodes.
        4.  Processing and tokenizing images associated with the nodes.
        5.  Creating structural features like edge indices and node degrees.
        6.  Formatting labels.

        Args:
            json_data (dict): A dictionary containing the raw graph data.

        Returns:
            A dictionary containing the processed graph data, with keys like
            "text", "images", "distance", "edges", etc.

        Raises:
            ValueError: If the graph is too large or has no valid labels (and
                `strict` is True).
        """
        dut.compute_relative_distance(json_data)
        flattened_graph = self.flatten_graph(json_data)
        if len(flattened_graph["id"]) > self.max_graph_size:
            raise ValueError(
                f"Graph too large ({len(flattened_graph['id'])} >"
                f" {self.max_graph_size})"
            )

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

        tokenized_text = self.text_tokenizer(
            flattened_graph["text"],
            padding="max_length",
            truncation=True,
            return_tensors="np",
            max_length=self.text_config.get("max_length", None),
            return_attention_mask=True,
            return_token_type_ids=self.text_config.get(
                "has_token_type_ids", False
            ),
        )  # type: ignore

        image_mask = []
        images = []
        for img in flattened_graph["images"]:
            if img:
                try:
                    img = Image.open(os.path.join(self.root, img)).convert(
                        "RGB"
                    )
                except Exception as e:
                    log.warning(f"Error checking image path: {e}")
                    img = None

            if img is not None:
                images.append(img)
                image_mask.append(True)
            else:
                image_mask.append(False)

        image_mask = np.array(image_mask, dtype=bool)

        image_mask = torch.tensor(
            [img is not None for img in flattened_graph["images"]],
            dtype=torch.bool,
        )
        images = [
            Image.open(os.path.join(self.root, img)).convert("RGB")
            for img in flattened_graph["images"]
            if img
        ]
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        tokenized_images = (
            self.image_tokenizer(images, return_tensors="pt")  # type: ignore
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
        distance_tensor = np.array(
            [
                [dist for _, dist in distances]
                for distances in combined_distance
            ],
            dtype=np.int16,
        )
        rotary_pos = np.array(
            flattened_graph["rotary_position"], dtype=np.int16
        )

        mapped_parent_ids = [
            id_map[parent_id] for parent_id in flattened_graph["parent_id"]
        ]
        mapped_ids = list(
            range(len(flattened_graph["id"]))
        )  # Indices are already in order

        edges = torch.tensor([mapped_ids, mapped_parent_ids])
        degree = pyg_utils.degree(
            edges[1][1:], num_nodes=edges.shape[1], dtype=torch.long
        )  # Degree excluding self-loop
        edges = edges.numpy()
        degree = degree.numpy()

        y = {
            key: np.array(
                [label[key] for label in flattened_graph["y"]], dtype=np.int8
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
            "edges": edges,
        }

    @property
    def hdf5_file(self) -> h5py.File:
        """Provides a handle to the HDF5 file.

        This property ensures that the HDF5 file is opened only when needed and
        provides a consistent way to access it. It opens the file in read mode.

        Returns:
            An `h5py.File` object for the dataset's HDF5 file.
        """
        if self._hdf5_file is None or self._hdf5_file.id is None:
            self._hdf5_file = h5py.File(self._hdf5_filename, "r")
        return self._hdf5_file

    def get_sizes(self, indices: list[int]) -> list[int]:
        """Calculates the total number of nodes for a list of graph indices.

        This method is used by the `BucketSampler` to group graphs of similar
        sizes together, which helps in creating more efficient batches. It reads
        the size of each graph from the HDF5 file's attributes.

        Args:
            indices (list[int]): A list of indices corresponding to the items
                in `self._flattened_data`.

        Returns:
            A list of integers, where each integer is the total number of nodes
            in the corresponding group of graphs.
        """
        assert self._flattened_data, "Flattened data not loaded"
        mapped_indices = [self._flattened_data[i] for i in indices]

        result = []
        for grouped_indices in tqdm(mapped_indices, desc="Getting sizes"):
            total_size = 0
            for dataset, graph_index in grouped_indices:
                group = self.hdf5_file.get(f"{dataset}/{graph_index}")
                assert isinstance(
                    group, h5py.Group
                ), f"Invalid group: {dataset}/{graph_index}"
                size = group.attrs["size"]
                assert isinstance(size, np.number), type(size)
                total_size = total_size + size

            result += [total_size]

        return result

    def _write_graph_to_hdf5(
        self,
        dataset_group: h5py.Group,
        data: dict,
        original_index: int,
        label_index=None,
    ) -> str:
        """Writes a single processed graph to the HDF5 file.

        This internal method serializes a processed graph (represented as a
        dictionary of numpy arrays and tensors) into a group within the HDF5
        file. It handles the creation of subgroups for different data modalities
        (text, images, etc.) and stores metadata like the graph size as an
        attribute.

        Args:
            dataset_group (h5py.Group): The HDF5 group for the specific dataset
                (e.g., "giga_pretrain").
            data (dict): The dictionary of processed graph data to be written.
            original_index (int): The original index of the graph in the raw
                JSON file.
            label_index (int, optional): An index used when a graph is split by
                node labels. Defaults to None.

        Returns:
            The name of the created graph group in the HDF5 file.
        """

        is_prod = os.getenv("IS_PROD", False) == "1" and not self.debug
        assert not is_prod, (
            f"Attempted to write graph {dataset_group=}, {original_index=},"
            f" {label_index=} in PROD mode"
        )

        if label_index is not None:
            graph_name = f"graph_{original_index}_label_{label_index}"
        else:
            graph_name = f"graph_{original_index}"
        group = dataset_group.create_group(graph_name)
        group.attrs["size"] = len(data["out_degree"])

        text_group = group.create_group("text")
        for key, tensor in data["text"].items():
            text_group.create_dataset(key, data=tensor)
        y_group = group.create_group("y")
        for key, tensor in data["y"].items():
            y_group.create_dataset(key, data=tensor)

        group.create_dataset("image_mask", data=data["image_mask"])
        if data["images"] is not None:
            images_group = group.create_group("images")
            for key, tensor in data["images"].items():
                images_group.create_dataset(key, data=tensor.cpu().numpy())
        group.create_dataset("distance", data=data["distance"])
        group.create_dataset("rotary_position", data=data["rotary_position"])
        group.create_dataset("out_degree", data=data["out_degree"])
        group.create_dataset("edges", data=data["edges"])

        return graph_name

    def _load_graph_from_hdf5(
        self, dataset_name: str, graph_index: str
    ) -> Dict[str, Any]:
        """Loads a processed graph from the HDF5 file as a dictionary.

        This internal method reads a graph from the specified HDF5 group and
        reconstructs it as a dictionary of PyTorch tensors. It handles loading
        all the different data components, including text, images, labels, and
        structural information.

        Args:
            dataset_name (str): The name of the dataset group in the HDF5 file.
            graph_index (str): The name of the specific graph group to load.

        Returns:
            A dictionary containing the loaded graph data, with values as
            PyTorch tensors.
        """
        group = self.hdf5_file.get(f"{dataset_name}/{graph_index}")
        assert isinstance(
            group, h5py.Group
        ), f"Invalid dataset group {group=}, {dataset_name=}, {graph_index=}"
        text_data = {
            key: torch.from_numpy(np_array[()])
            for key, np_array in cast(h5py.Group, group["text"]).items()
        }
        y_data = {
            key: torch.from_numpy(np_array[()])
            for key, np_array in cast(h5py.Group, group["y"]).items()
        }
        image_mask = torch.from_numpy(
            cast(h5py.Dataset, group["image_mask"])[()]
        )
        images_data = (
            {
                key: torch.from_numpy(np_array[()])
                for key, np_array in cast(h5py.Group, group["images"]).items()
            }
            if "images" in group
            else None
        )
        distance = torch.from_numpy(group["distance"][()])
        rotary_position = torch.from_numpy(group["rotary_position"][()])
        out_degree = torch.from_numpy(group["out_degree"][()])
        edges = torch.from_numpy(group["edges"][()])

        return {  # Return a dictionary now, not Data object
            "text": text_data,
            "y": y_data,
            "image_mask": image_mask,
            "images": images_data,
            "distance": distance,
            "rotary_position": rotary_position,
            "out_degree": out_degree,
            "edges": edges,
        }

    def __getitem__(self, idx: int) -> List[Dict[str, Any]]:
        """Loads and returns a group of processed graphs from the HDF5 file.

        This method is the core of the PyTorch `Dataset` interface. It takes an
        index, retrieves the corresponding group of graph identifiers from the
        flattened data list, and loads each graph from the HDF5 file using
        `_load_graph_from_hdf5`.

        The number of graphs in the returned list is determined by the
        `group_size` parameter.

        Args:
            idx (int): The index of the data sample to retrieve.

        Returns:
            A list of dictionaries, where each dictionary represents a
            processed graph.
        """

        if self._flattened_data is None:
            # This creates flattened data, kinda gross, but helps for tests.
            _ = self.data_splits

        assert self._flattened_data
        try:
            grouped_data = self._flattened_data[idx]
        except Exception as e:
            log.warning(f"Index {idx} not found in {self._flattened_data=}")
            raise e

        data = [
            self._load_graph_from_hdf5(dataset_name, graph_index)
            for dataset_name, graph_index in grouped_data
        ]

        return data
