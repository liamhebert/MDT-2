"""Dataset classes and utilities."""

from dataclasses import dataclass
import os.path as osp

import pandas as pd
from pytorch_lightning import LightningDataModule
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm

tqdm.pandas()


class DataModule(LightningDataModule):
    """DataModule containing processed train/val/test dataloaders for our
    dataset.

    This class handles the loading, splitting, and pre-processing of the dataset.

    Params:
        data_dir: The directory containing the raw dataset files
        train_batch_size: The total batch size for training. Must be divisible
            by the number of GPUs.
        test_batch_size: The total batch size for testing. Must be divisible by
            the number of GPUs.
        train_val_test_split: A tuple containing the percentage split between
            train, val, and test datasets.
        num_workers: The number of workers to use for data loading.
        force_remake: Whether to force remake the dataset cache. Relevant if the
            dataset is configured to cache to disk.
        pin_memory: Whether to pin batches in GPU memory in the dataloader. This
            helps with performance on GPU, but can cause issues with large
            datasets.
    """

    # Datasets are loaded in lazily during "setup" to assist with DDP
    _train_dataset: Dataset | None = None
    _val_dataset: Dataset | None = None
    _test_dataset: Dataset | None = None

    _train_device_batch_size: int = 1
    _test_device_batch_size: int = 1

    def __init__(
        self,
        data_dir: str,
        train_batch_size: int,
        test_batch_size: int,
        train_val_test_split: tuple[float, float, float],
        num_workers: int,
        force_remake: bool,
        pin_memory: bool,
    ):
        super()
        assert (
            sum(train_val_test_split) == 1.0
        ), f"Train/val/test split must sum to 1.0. Got {train_val_test_split=}"
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

    def prepare_data(self):
        """Prepare the data for the dataset.

        This is only called once on the rank 0 gpu per run, and results in
        memory are not replicated across gpus. This is useful for downloading.
        """
        # TODO(liamhebert): Implement data preparation logic
        if self.hparams.force_remake is False and osp.exists(self.tensor_dir):
            return
        pass

    def setup(self, stage: str):
        """Load dataset for training/validation/testing.

        NOTE: When using DDP (multiple GPUs), this is run once per GPU.
        As a result, this function should be deterministic and not download
        or have side effects. As a result, all data processing should be done in
        prepare_data and cached to disk, or done prior to training.

        Args:
            stage: either 'fit' (train), 'validate', 'test', or 'predict'
        """

        # We only have access to trainer in setup, so we need to calculate
        # these parameters here.
        if self.trainer is not None and (
            self._train_device_batch_size is None
            or self._test_device_batch_size is None
        ):
            # We test both here to fail quickly if misconfigured
            if (
                self.hparams.train_batch_size % self.trainer.world_size != 0
                or self.hparams.test_batch_size % self.trainer.world_size != 0
            ):
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible"
                    f"by the number of devices ({self.trainer.world_size})."
                )

            self._train_device_batch_size = (
                self.hparams.train_batch_size // self.trainer.world_size
            )
            self._test_device_batch_size = (
                self.hparams.test_batch_size // self.trainer.world_size
            )

        # TODO(liamhebert): Implement dataset setup logic
        # TODO(liamhebert): Implement debug mode
        if stage == "fit" and self._train_dataset is None:
            # make training dataset
            self._train_dataset = Dataset()
        elif stage == "validate" and self._val_dataset is None:
            # make validation dataset
            self._val_dataset = Dataset()
        elif (
            stage == "test" or stage == "predict"
        ) and self._test_dataset is None:
            # Make test dataset
            self._test_dataset = Dataset()
        else:
            raise ValueError(f"Unexpected stage: {stage}")

    def train_dataloader(self) -> DataLoader:
        """Return the training dataloader."""
        return DataLoader(
            self._train_dataset,
            batch_size=self.hparams.train_batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """Return the validation dataloader."""
        return DataLoader(
            self._val_dataset,
            batch_size=self.hparams.train_batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        """Return the test dataloader."""
        return DataLoader(
            self._test_dataset,
            batch_size=self.hparams.test_batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )


class Dataset(Dataset):
    """Dataset instance for a dataloader.

    Params:
        df: The dataframe containing the dataset, used for tracking sizes.
        tensor_dir: The directory containing processed tensors.
    """

    # NOTE: If things are bigger then what can be held in memory, we will need
    # to offload some to disk. This is particularly relevant with distributed
    # training, where each gpu will have a copy of df.
    df: pd.DataFrame

    def __getitem__(self, idx) -> dict[str, torch.Tensor]:
        """Fetch a single item from the dataset indexed by idx.

        Params:
            idx: The index of the item to fetch.

        Returns:
            A dictionary mapping keys to torch tensors. It is expected that the
            tensors have a shape of (batch_size, ...).
        """
        pass

    def __len__(self) -> int:
        """Return the size of the dataset."""
        return len(self.df)
