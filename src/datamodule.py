"""
Dataset classes and utilities.
"""

from lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import Subset
from tqdm import tqdm

from data.collated_datasets import CollatedDataset

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

    _train_device_batch_size: int | None = None
    _test_device_batch_size: int | None = None

    master_dataset: CollatedDataset

    def __init__(
        self,
        dataset: CollatedDataset,
        train_batch_size: int,
        test_batch_size: int,
        num_workers: int,
        pin_memory: bool,
    ):
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.master_dataset = dataset
        self.save_hyperparameters(logger=False)

    def prepare_data(self):
        """Prepare the data for the dataset.

        This is only called once on the rank 0 gpu per run, and results in
        memory are not replicated across gpus. This is useful for downloading.
        """
        self.master_dataset.process()

    def setup(self, stage: str):
        """Load dataset for training/validation/testing.

        NOTE: When using DDP (multiple GPUs), this is run once per GPU.
        As a result, this function should be deterministic and not download
        or have side effects. As a result, all data processing should be done in
        prepare_data and cached to disk, or done prior to training.

        Args:
            stage: either 'fit' (train), 'validate', 'test', or 'predict'
        """

        train_batch_size: int = self.hparams.train_batch_size  # type: ignore
        test_batch_size: int = self.hparams.test_batch_size  # type: ignore

        # We only have access to trainer in setup, so we need to calculate
        # these parameters here.
        if self.trainer is not None and (
            self._train_device_batch_size is None
            or self._test_device_batch_size is None
        ):
            # We test both here to fail quickly if misconfigured
            if (
                train_batch_size % self.trainer.world_size != 0
                or test_batch_size % self.trainer.world_size != 0
            ):
                raise RuntimeError(
                    f"Batch size "
                    f"({train_batch_size}, {test_batch_size})"
                    " is not divisible by the number of devices "
                    f"({self.trainer.world_size})."
                )

            self._train_device_batch_size = (
                train_batch_size // self.trainer.world_size
            )
            self._test_device_batch_size = (
                test_batch_size // self.trainer.world_size
            )
        else:
            self._train_device_batch_size = train_batch_size
            self._test_device_batch_size = test_batch_size

        if self._train_dataset is None:
            # make training dataset
            self._train_dataset = Subset(
                self.master_dataset, self.master_dataset.train_idx
            )
        if self._val_dataset is None:
            # make validation dataset
            self._val_dataset = Subset(
                self.master_dataset, self.master_dataset.valid_idx
            )
        if self._test_dataset is None:
            # Make test dataset
            self._test_dataset = Subset(
                self.master_dataset, self.master_dataset.test_idx
            )

    def train_dataloader(self) -> DataLoader:
        """
        Return the training dataloader.
        """
        assert self._train_dataset is not None, "Train dataset not loaded"

        return DataLoader(
            self._train_dataset,
            batch_size=self._train_device_batch_size,  # type: ignore
            shuffle=True,
            num_workers=self.hparams.num_workers,  # type: ignore
            pin_memory=self.hparams.pin_memory,  # type: ignore
            collate_fn=self.master_dataset.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Return the validation dataloader.
        """
        assert self._val_dataset is not None, "Val dataset not loaded"

        return DataLoader(
            self._val_dataset,
            batch_size=self._test_device_batch_size,  # type: ignore
            shuffle=False,
            num_workers=self.hparams.num_workers,  # type: ignore
            pin_memory=self.hparams.pin_memory,  # type: ignore
            collate_fn=self.master_dataset.collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        """
        Return the test dataloader.
        """
        assert self._test_dataset is not None, "Test dataset not loaded"

        return DataLoader(
            self._test_dataset,
            batch_size=self._test_device_batch_size,  # type: ignore
            shuffle=False,
            num_workers=self.hparams.num_workers,  # type: ignore
            pin_memory=self.hparams.pin_memory,  # type: ignore
            collate_fn=self.master_dataset.collate_fn,
        )
