"""
Dataset classes and utilities.
"""

from lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler
from torch.utils.data import Subset
from torch.utils.data import SequentialSampler
from tqdm import tqdm
from torchdata.stateful_dataloader import StatefulDataLoader
from data.bucket_sampler import LengthGroupedSampler, LengthSubsetDataset
from data.collated_datasets import CollatedDataset
from utils import RankedLogger
from typing import Iterable

tqdm.pandas()
log = RankedLogger(__name__, rank_zero_only=True)


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
        cache_dataset: Whether to cache the dataset to memory. This can help
            speed up training, but can cause memory issues with large datasets.
    """

    # Datasets are loaded in lazily during "setup" to assist with DDP
    _train_dataset: Subset | LengthSubsetDataset | None = None
    _val_dataset: Subset | None = None
    _test_dataset: Subset | None = None

    _train_device_batch_size: int | None = None
    _test_device_batch_size: int | None = None

    batch_sampler: LengthGroupedSampler | None = None
    _val_sampler: DistributedSampler | None = None
    _test_sampler: DistributedSampler | None = None
    _train_sampler: DistributedSampler | None = None

    master_dataset: CollatedDataset

    def __init__(
        self,
        dataset: CollatedDataset,
        train_batch_size: int,
        test_batch_size: int,
        num_workers: int,
        pin_memory: bool,
        cache_dataset: bool,
        use_length_grouped_sampler: bool = True,
        max_total_length: int | None = None,
        max_length_multiplier: float | None = None,
    ):
        super().__init__()
        assert (
            cache_dataset or not use_length_grouped_sampler
        ), "Must cache dataset if using length grouped sampler"

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.master_dataset = dataset
        self.save_hyperparameters(logger=False, ignore="dataset")

    def prepare_data(self):
        """Prepare the data for the dataset.

        This is only called once on the rank 0 gpu per run, and results in
        memory are not replicated across gpus. This is useful for downloading.
        """
        self.master_dataset.process()

    def teardown(self, stage):
        if self.master_dataset._hdf5_file is not None:
            self.master_dataset._hdf5_file.close()

    def build_sampler(self, dataset: Iterable) -> DistributedSampler | None:
        """Build a distributed sampler for the dataset."""
        if self.trainer:
            distributed_kwargs = self.trainer.distributed_sampler_kwargs
            if distributed_kwargs is not None:
                log.info("Loading distributed sampler")
                return DistributedSampler(
                    dataset, **distributed_kwargs  # type: ignore
                )
            else:
                return None
        return None

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
        max_total_length = self.hparams.max_total_length  # type: ignore

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
                    "Batch size "
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
            if max_total_length:
                max_total_length = max_total_length // self.trainer.world_size
            log.info(
                f"TRAIN DEVICE BATCH SIZE: {self._train_device_batch_size}"
            )
            log.info(f"TEST DEVICE BATCH SIZE {self._test_device_batch_size}")
        else:
            self._train_device_batch_size = train_batch_size
            self._test_device_batch_size = test_batch_size

        # if (
        #     self._train_dataset is None
        #     and self._val_dataset is None
        #     and self._test_dataset is None
        # ):
        #     if self.hparams.cache_dataset:  # type: ignore
        #         self.master_dataset.populate_cache(counts_only=False)

        if self._train_dataset is None:
            # make training dataset
            if self.hparams.use_length_grouped_sampler:  # type: ignore
                self._train_dataset = LengthSubsetDataset(
                    self.master_dataset,
                    self.master_dataset.train_idx,
                )
                indices = self._train_dataset.indices
                example_lengths = self.master_dataset.get_sizes(indices)
                assert example_lengths is not None, "Graph sizes not loaded"
                multiplier = self.hparams.max_length_multiplier  # type: ignore
                if multiplier is not None:
                    max_total_length = int(
                        self._train_device_batch_size * multiplier
                    )

                # Since we do distributed training, this can't be shuffled
                self.batch_sampler = LengthGroupedSampler(
                    example_lengths=example_lengths,
                    batch_size=self._train_device_batch_size,  # type: ignore
                    shuffle=True,
                    shuffle_every_epoch=True,
                    drop_last=True,
                    max_total_length=max_total_length,
                )

                sampler = self.build_sampler(range(len(self.batch_sampler)))
                if sampler is None:
                    sampler = SequentialSampler(range(len(self.batch_sampler)))

                self.batch_sampler.set_sampler(sampler)
            else:
                self._train_dataset = Subset(
                    self.master_dataset, self.master_dataset.train_idx
                )
                self._train_sampler = self.build_sampler(
                    self.master_dataset.train_idx
                )

        if self._val_dataset is None:
            # make validation dataset
            self._val_dataset = Subset(
                self.master_dataset, self.master_dataset.valid_idx
            )
            self._val_sampler = self.build_sampler(
                self.master_dataset.valid_idx
            )

        if self._test_dataset is None:
            # Make test dataset
            self._test_dataset = Subset(
                self.master_dataset, self.master_dataset.test_idx
            )
            self._test_sampler = self.build_sampler(
                self.master_dataset.test_idx
            )

        # if flag_cache_later:
        #     log.info("Finally caching dataset into memory...")

        #     # assert self._val_sampler is not None
        #     # assert self._test_sampler is not None

        #     # # Since we are distributed, we selectively load things in cache.
        #     # if self.hparams.use_length_grouped_sampler:  # type: ignore
        #     #     assert self.batch_sampler is not None
        #     #     assert self.batch_sampler.sampler is not None
        #     #     train_idx = set(chain.from_iterable(self.batch_sampler))
        #     # else:
        #     #     assert self._train_sampler is not None
        #     #     train_idx = set(self._train_sampler)

        #     # valid_idx = set(self._val_sampler)
        #     # test_idx = set(self._test_sampler)
        #     # total_indices = train_idx | valid_idx | test_idx
        #     self.master_dataset.populate_cache(counts_only=False)

    def train_dataloader(self) -> DataLoader:
        """
        Return the training dataloader.
        """
        assert self._train_dataset is not None, "Train dataset not loaded"
        return StatefulDataLoader(
            self._train_dataset,
            batch_size=(
                1 if self.batch_sampler else self._train_device_batch_size
            ),
            batch_sampler=self.batch_sampler,
            sampler=None if self.batch_sampler else self._train_sampler,
            shuffle=None if self.batch_sampler else True,
            num_workers=self.hparams.num_workers,  # type: ignore
            pin_memory=self.hparams.pin_memory,  # type: ignore
            collate_fn=self.master_dataset.collate_fn,
            prefetch_factor=20,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Return the validation dataloader.
        """
        assert self._val_dataset is not None, "Val dataset not loaded"

        return StatefulDataLoader(
            self._val_dataset,
            sampler=self._val_sampler,
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

        return StatefulDataLoader(
            self._test_dataset,
            sampler=self._test_sampler,
            batch_size=self._test_device_batch_size,  # type: ignore
            shuffle=False,
            num_workers=self.hparams.num_workers,  # type: ignore
            pin_memory=self.hparams.pin_memory,  # type: ignore
            collate_fn=self.master_dataset.collate_fn,
        )
