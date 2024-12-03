"""General wrappers for datasets and dataloaders.

The organization of dataloading functions as follows:
- `pyg_datasets` contain dataloading functions for loading and pre-processing
    PyG objects into general data objects we can use. Includes logic for
    splitting datasets into train, validation, and test sets
- `GraphormerDataset` is a wrapper for PyG datasets and provides a unified
    interface for datasets. This returns individual samples from the dataset
    without batching
- `BatchedDataDataset` interface with `GraphormerDataset` and provide
    task-specific collator functions to batch data for contrastive and node
    prediction tasks
- `EpochShuffleDataset` is a wrapper for `BatchedDataDataset` datasets to
    include shuffling logic for each epoch

As such, the hierarchy of the dataloading process is:
`pyg_datasets`
-> `GraphormerDataset`
-> `BatchedDataDataset`
"""

from abc import ABC
from abc import abstractmethod
import copy
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from data.collator_utils import generic_collator
from tasks.dataset import TaskDataset

DatasetCopy = TypeVar("DatasetCopy", bound="GraphormerDataset")


class GraphormerDataset(Dataset):
    """Dataset class to hold data splits and generic pre-processing.

    Notably, this class holds functionality to split a dataset into train,
    validation, and test sets. This is done by creating copies of itself with
    the desired data pointed by indices or subsets.
    """

    dataset: TaskDataset
    train_idx: torch.Tensor | None
    valid_idx: torch.Tensor | None
    test_idx: torch.Tensor | None
    train_data: TaskDataset | None
    valid_data: TaskDataset | None
    test_data: TaskDataset | None

    seed: int

    def __init__(
        self,
        dataset: TaskDataset,
        seed: int = 0,
        train_idx: NDArray[np.int_] | None = None,
        valid_idx: NDArray[np.int_] | None = None,
        test_idx: NDArray[np.int_] | None = None,
        train_set: TaskDataset = None,
        valid_set: TaskDataset = None,
        test_set: TaskDataset = None,
    ):
        """Creates the dataset object from either precomputed splits or indices.

        The dataset can be created in three different ways:
        - Random train, valid, and test splits: If no indices or splits are
            provided, the dataset will be split into 80% train, 10% valid, and
            10% test.
        - Precomputed splits: If train_set, valid_set, and test_set sets are
            provided, the dataset will comprise of these splits.
        - Precomputed indices: If train_idx, valid_idx, and test_idx are
            provided, we will create splits of dataset based on these indices.

        Args:
            dataset (Dataset): The primary dataset object which contains the
                entire dataset.
            seed (int, optional): Random seed to use for shuffling indices and
                creating train-test-splits. Defaults to 0.
            train_idx (NDArray[np.int] | None, optional): Numpy array of unique
                dataset indices which comprise the train set. Defaults to None.
            valid_idx (NDArray[np.int] | None, optional): Numpy array of unique
                dataset indices which comprise the validation set. Defaults to
                None.
            test_idx (NDArray[np.int] | None, optional): Numpy array of unique
                dataset indices which comprise the test set. Defaults to None.
            train_set (Dataset, optional): Pre-computed Dataset object
                comprising the train set. Defaults to None.
            valid_set (Dataset, optional): Pre-computed Dataset object
                comprising the validation set. Defaults to None.
            test_set (Dataset, optional): Pre-computed Dataset object comprising
                the test set. Defaults to None.
        """
        self.dataset = dataset
        self.seed = seed

        if train_idx is None and train_set is None:
            self.num_data = len(self.dataset)
            train_idx, test_valid_idx = train_test_split(
                np.arange(self.num_data),
                test_size=self.num_data // 5,
                random_state=seed,
            )
            test_idx, valid_idx = train_test_split(
                test_valid_idx, test_size=self.num_data // 10, random_state=seed
            )
            self.train_idx = torch.from_numpy(train_idx)
            self.valid_idx = torch.from_numpy(valid_idx)
            self.test_idx = torch.from_numpy(test_idx)
            self.train_data = self.index_select(self.train_idx)
            self.valid_data = self.index_select(self.valid_idx)
            self.test_data = self.index_select(self.test_idx)

        has_precomputed_splits = (
            train_set is not None
            and valid_set is not None
            and test_set is not None
        )
        has_indices = (
            train_idx is not None
            and valid_idx is not None
            and test_idx is not None
        )

        assert has_precomputed_splits != has_indices, (
            "Must provide either precomputed splits or indices, and not both."
            f"{has_precomputed_splits=}, {has_indices=}"
        )

        if has_precomputed_splits:
            self.num_data = len(train_set) + len(valid_set) + len(test_set)
            assert self.num_data <= len(dataset), "Splits exceed dataset size."
            self.train_data = self.create_subset(train_set)
            self.valid_data = self.create_subset(valid_set)
            self.test_data = self.create_subset(test_set)
            self.train_idx = None
            self.valid_idx = None
            self.test_idx = None
        elif (
            train_idx is not None
            and valid_idx is not None
            and test_idx is not None
        ):
            # mypy has a hard time parsing that we do have values for
            # train_idx, valid_idx, and test_idx when using has_indices, but
            # oddly is fine with has_precomputed_splits. This is a workaround.
            assert train_idx.max() < self.num_data
            assert valid_idx.max() < self.num_data
            assert test_idx.max() < self.num_data
            assert train_idx.min() >= 0
            assert valid_idx.min() >= 0
            assert test_idx.min() >= 0

            self.num_data = len(train_idx) + len(valid_idx) + len(test_idx)

            assert self.num_data == len(
                set(train_idx) | set(valid_idx) | set(test_idx)
            ), "Indices must be unique."
            assert self.num_data <= len(dataset), "Indices exceed dataset size."

            rng = np.random.RandomState(seed)
            rng.shuffle(train_idx)
            rng.shuffle(valid_idx)
            rng.shuffle(test_idx)
            self.train_idx = torch.from_numpy(train_idx)
            self.valid_idx = torch.from_numpy(valid_idx)
            self.test_idx = torch.from_numpy(test_idx)
            self.train_data = self.index_select(self.train_idx)
            self.valid_data = self.index_select(self.valid_idx)
            self.test_data = self.index_select(self.test_idx)
        else:
            raise ValueError(
                (
                    "Must provide either precomputed splits or indices."
                    f"Splits: {train_set=}, {valid_set=}, {test_set=}."
                    f"Indices: {train_idx=}, {valid_idx=}, {test_idx=}"
                )
            )

        self.__indices__ = None

    def index_select(self, idx: torch.Tensor) -> DatasetCopy:
        """Creates a copy of this dataset with just the data referenced by idx.

        This is useful for creating a train, valid, and test set, where you can
        create copies by calling

        ```
        dataset = GraphormerPYGDataset(full_dataset)
        train = dataset.index_select(dataset.train_idx)
        test = dataset.index_select(dataset.test_idx)
        ```

        or some custom variant by specifying a list of indices. This differs
        from create_subset, as it does not require a dataset object to be passed
        in.

        NOTE: This should be used to query data, as it sets the "dataset"
        property to the passed in subset, which is used by the __getitem__
        method. Without this, the default __getitem__ method will query from the
        whole dataset without splits.

        TODO(liamhebert): This is kinda wasteful, since it requires a full copy
        before creating the subset. Instead, we should just access the
        properties directly, adding the pre-processing to the lower-level
        GraphormerDataset in dataloaders.py.

        Args:
            idx (list[int]): List of indices to create a subset from.

        Returns:
            DatasetCopy: _description_
        """
        dataset = copy.copy(self)
        dataset.dataset = self.dataset.index_select(idx)

        if isinstance(idx, torch.Tensor):
            dataset.num_data = idx.size(0)
        else:
            dataset.num_data = idx.shape[0]

        dataset.__indices__ = idx
        dataset.train_data = None
        dataset.valid_data = None
        dataset.test_data = None
        dataset.train_idx = None
        dataset.valid_idx = None
        dataset.test_idx = None
        return dataset

    def create_subset(self, subset: Dataset) -> DatasetCopy:
        """Creates a copy of this dataset with just the data within the subset.

        This is useful for creating a train, valid, and test set, where you can
        create copies by calling
        ```
        dataset = GraphormerPYGDataset(full_dataset)
        train = dataset.create_subset(dataset.train_data)
        test = dataset.create_subset(dataset.test_data)
        ```

        NOTE: This should be used to query data, as it sets the "dataset"
        property to the passed in subset, which is used by the __getitem__
        method. Without this, the default __getitem__ method will query from the
        whole dataset without splits.

        TODO(liamhebert): This is kinda wasteful, since it requires a full copy
        before creating the subset. Instead, we should just access the
        properties directly, adding the pre-processing to the lower-level
        GraphormerDataset in dataloaders.py.

        Args:
            subset (Dataset): Dataset object to set

        Returns:
            DatasetCopy: _description_
        """
        dataset = copy.copy(self)
        dataset.dataset = subset
        dataset.num_data = len(subset)

        dataset.__indices__ = None
        dataset.train_data = None
        dataset.valid_data = None
        dataset.test_data = None
        dataset.train_idx = None
        dataset.valid_idx = None
        dataset.test_idx = None
        return dataset

    def __getitem__(self, idx: int) -> Data:
        """Returns a graph at index idx.

        Args:
            idx (int): Index of the graph to fetch

        Raises:
            TypeError: Raises if index is not an integer, for some reason.

        Returns:
            Data: PyG Graph object processed and ready to be collated.
        """
        if isinstance(idx, int):
            item = self.dataset[idx]
            item.idx = idx
            item.y = item.y.reshape(-1)
            return preprocess_item(item)
        else:
            raise TypeError(
                "index to a GraphormerPYGDataset can only be an integer."
            )

    def __len__(self):
        """
        Returns the number of examples in the dataset.
        """
        return self.num_data


class BatchedDataDataset(ABC, GraphormerDataset):
    """
    Dataloader to batch examples from `GraphormerDataset` into task-specific
    batches with provided collator functions.
    """

    def __init__(self, spatial_pos_max=1024):
        """
        Args:
            dataset: (GraphormerDataset) dataset to batch
            spatial_pos_max: (int) maximum spatial position to consider. Any
                node farther away then this distance is attention masked
        """
        super().__init__()
        self.spatial_pos_max = spatial_pos_max

    @abstractmethod
    def collater(self, samples):
        """Collate function to merge data samples of various sizes into a batch.

        Individual data samples are comprised of the following attributes:
        - idxs: (int) list of unique indices from 0 to batch_size for each item
        - attn_biases: (List[float]) list of attention biases values for each
            node in the graph
        - spatial_poses: (List[int]) list of spatial indexes for each node in the
            graph. Used to fetch spatial position embeddings
        - in_degrees: (List[int]) list of the in-degree for each node in the
            graph. Used to fetch degree embeddings
        - x_text: (List[Dict[str, torch.Tensor]]) list of text input data for
            each node in the graph. Each input is a dictionary with
            pre-tokenized text tokens
        - x_image_indexes: (List[torch.Tensor]) list of boolean tensors
            indicating which nodes have images
        - x_images: (List[torch.Tensor]) list of image features for each node
            in the graph
        - distance: (List[torch.Tensor]) list of exact spatial distance between
            nodes, used to clip attention bias
        - ys: (List[torch.Tensor]) list of target labels for each node in the
            graph or a single label per graph

        Args:
            items: list of data samples
            spatial_pos_max: maximum spatial pos

        Returns:
            A collated patch of data samples where each item is padded to the
            largest size in the batch.

            Each output dictionary must contains the following keys:
            - idx: (torch.Tensor) batched indices
            - attn_bias: (torch.Tensor) batched attention biases
            - spatial_pos: (torch.Tensor) batched spatial positions
            - in_degree: (torch.Tensor) batched in-degrees
            - out_degree: (torch.Tensor) batched out-degrees
            - x_token_mask: (torch.Tensor) batched token mask
            - x: (torch.Tensor) batched tokenized text input
            - x_token_type_ids: (torch.Tensor) batched token type ids
            - x_attention_mask: (torch.Tensor) batched attention mask
            - x_images: (torch.Tensor) batched image features
            - x_image_indexes: (torch.Tensor) batched image indexes
            - y: (torch.Tensor) batched target labels

            Additional features for task specific loss functions may also be
            added but they are not needed for general processing
        """
        ...


class ContrastiveBatchedDataDataset(BatchedDataDataset):
    """
    Dataset with contrastive learning specific collate function.
    """

    def collater(self, samples):
        """Collate function specific to contrastive learning tasks.

        Each item follows the data structure as the general collate function but
        with each item having the following version specific attributes:
        - hard_y: (torch.Tensor) tensor of labels of the polar opposite
            communities
        - y: (torch.Tensor) tensor of labels of which topic the community
            belongs to
        """
        samples = [item for item in samples if item is not None]
        items = [
            (
                item.idx,
                item.attn_bias,
                item.spatial_pos,
                item.in_degree,
                item.x,
                item.x_image_index,
                item.x_images,
                item.distance,
                item.y,
            )
            for item in samples
        ]
        hard_ys = [item.hard_y for item in samples]
        hard_y = torch.cat(hard_ys)
        collated_output = generic_collator(items, self.spatial_pos_max)
        collated_output["hard_y"] = hard_y
        return collated_output


class NodeBatchedDataDataset(BatchedDataDataset):
    """
    Dataset with Node learning specific collate function.
    """

    def collater(self, samples):
        """Collate function specific to node prediction tasks.

        Each item follows the data structure as the general collate function but
        with each item having the following version specific attributes:
        - y_mask: (torch.Tensor) boolean tensor for each node in the graph
            indicating if it has a label
        - y: (torch.Tensor) tensor of labels for each node in the graph.
            If a node does not have a label, it is padded with 0
        """

        samples = [sample for sample in samples if sample is not None]
        items = [
            (
                item.idx,
                item.attn_bias,
                item.spatial_pos,
                item.in_degree,
                item.x,
                item.x_image_index,
                item.x_images,
                item.distance,
                item.y,
            )
            for item in samples
        ]
        y_masks = [item.y_mask for item in samples]
        y_mask = torch.cat(y_masks).bool()
        collated_output = generic_collator(items, self.spatial_pos_max)
        collated_output["y_mask"] = y_mask
        return collated_output
