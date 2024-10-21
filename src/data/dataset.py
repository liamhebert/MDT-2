"""
Classes related to managing data splits and generic pre-processing.
"""

import copy
from functools import lru_cache
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split
import torch
from torch_geometric.data import Data
from torch_geometric.data import Dataset

from data.pre_processing import preprocess_item

DatasetCopy = TypeVar("DatasetCopy", bound="GraphormerPYGDataset")


class GraphormerPYGDataset(Dataset):
    """Dataset class to hold data splits and generic pre-processing.

    Notably, this class holds functionality to split a dataset into train,
    validation, and test sets. This is done by creating copies of itself with
    the desired data pointed by indices or subsets.
    """

    dataset: Dataset
    train_idx: torch.Tensor | None
    valid_idx: torch.Tensor | None
    test_idx: torch.Tensor | None
    train_data: Dataset | None
    valid_data: Dataset | None
    test_data: Dataset | None

    seed: int

    # TODO(liamhebert): We can probably merge this with GraphormerDataset in
    # dataloaders.py
    def __init__(
        self,
        dataset: Dataset,
        seed: int = 0,
        train_idx: NDArray[np.int_] | None = None,
        valid_idx: NDArray[np.int_] | None = None,
        test_idx: NDArray[np.int_] | None = None,
        train_set: Dataset = None,
        valid_set: Dataset = None,
        test_set: Dataset = None,
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
            idx (list[int]): _description_

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

    @lru_cache(maxsize=32)
    def get(self, idx: int) -> Data:
        """Returns a graph at index idx.

        Args:
            idx (int): Index of the graph to fetch

        Raises:
            TypeError: Raises if index is not an integer, for some reason.

        Returns:
            Data: PyG Graph object processed and ready to be collated.
        """
        # TODO(liamhebert): Do we need this if we have __getitem__?
        if isinstance(idx, int):
            item = self.dataset[idx]
            item.idx = idx
            item.y = item.y.reshape(-1)
            return preprocess_item(item)
        else:
            raise TypeError(
                "index to a GraphormerPYGDataset can only be an integer."
            )

    @lru_cache(maxsize=32)
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

    def len(self):
        """
        Returns the number of examples in the dataset.
        """
        # TODO(liamhebert): Do we need a second __len__ ?
        return self.num_data

    def __len__(self):
        """
        Returns the number of examples in the dataset.
        """
        return self.num_data
