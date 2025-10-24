"""Utilities for constructing length-aware batches for PyTorch datasets.

This module provides samplers and dataset helpers that group examples with
similar lengths into fixed-size batches. The goal is to improve computational
efficiency by keeping sequences within a batch close to a target total length
while remaining mindful of distributed training setups. The helper dataset
allows sampling subsequences in a way that plays nicely with the custom
sampler.
"""

from torch.utils.data import BatchSampler, Sampler
import random
from typing import List, Sequence, Sized, cast
from torch.utils.data import Dataset
from tqdm import tqdm
from utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=False)


class LengthGroupedSampler(BatchSampler):
    """
    Sampler that groups dataset examples into a fixed number of batches,
    each with the same number of items, while trying to maximize the length
    of items within each batch close to max_total_length using a modified
    First-Fit Decreasing (FFD) like approach.

    Args:
        example_lengths (list[int]): A list of lengths for each example in the
            dataset, pre-computed.
        batch_size (int): Required batch size.
        shuffle (bool): Whether to shuffle the dataset indices *before* sorting
            by length (affects tie-breaking).
        drop_last (bool): Whether to drop the last incomplete batch if the total
            number of examples is not perfectly divisible by num_batches.
    """

    sampler: Sampler[int] | None = None

    def __init__(
        self,
        example_lengths: list[int],
        batch_size: int = 1,
        shuffle: bool = True,
        shuffle_every_epoch: bool = False,
        drop_last: bool = True,
        sampler: Sampler[int] | None = None,
        max_total_length: int | None = None,
        hard_limit: int | None = None,
    ):
        """Initialize the sampler with precomputed sequence lengths.

        Args:
            example_lengths: Per-example lengths used to sort and bin pack
                items.
            batch_size: Desired number of examples per batch.
            shuffle: Shuffles indices before sorting lengths, randomizing
                batches between runs.
            shuffle_every_epoch: Recreates groupings after each epoch to add
                additional randomness.
            drop_last: Drops incomplete batches when the dataset size is not a
                multiple of the number of batches.
            sampler: Optional sampler that defines the order in which batches
                are yielded.
            max_total_length: Upper bound on the total sequence length per
                batch; default derives from the mean example length.
            hard_limit: Absolute ceiling on batch length; batches exceeding it
                are discarded.
        """

        self.batch_size = batch_size  # Have to do this for lightning
        self.example_lengths = example_lengths
        self.hard_limit = hard_limit

        if max_total_length is None:
            # We take the mean length of examples as the max total length
            mean_total_length = sum(example_lengths) // len(example_lengths)
            max_total_length = mean_total_length * batch_size
            log.info(
                "Since max_total_length is None, we will use the mean "
                f"length of examples ({mean_total_length}) * batch_size "
                f"({batch_size}) as the max total length ({max_total_length})."
            )

        self.shuffle_every_epoch = shuffle_every_epoch
        self.shuffle_every_epoch = False
        if shuffle_every_epoch:
            assert shuffle, (
                "shuffle_every_epoch requires shuffle=True, as it would be"
                " redundant otherwise."
            )
        self.group_args = {
            "batch_size": batch_size,
            "target_length": max_total_length,
            "drop_last": drop_last,
            "shuffle": shuffle,
        }
        # Initial grouping (distributed-aware)
        self.batches = self.group_and_broadcast_batches(
            example_lengths=self.example_lengths,
            batch_size=batch_size,
            target_length=max_total_length,
            drop_last=drop_last,
            shuffle=shuffle,
            hard_limit=self.hard_limit,
        )
        if sampler is not None:
            assert hasattr(sampler, "__len__")
            assert len(sampler) <= len(self.batches)  # type: ignore
            self.sampler = sampler

    def set_sampler(self, sampler: Sampler[int]) -> None:
        """Attach a sampler that controls the order in which batches appear."""
        assert hasattr(sampler, "__len__")
        assert len(sampler) <= len(self.batches)  # type: ignore
        self.sampler = sampler

    @staticmethod
    def _pure_group(
        *,
        example_lengths: list[int],
        batch_size: int,
        target_length: int,
        drop_last: bool,
        shuffle: bool,
        hard_limit: int | None,
    ) -> List[List[int]]:
        """Group indices into batches using a first-fit decreasing heuristic.

        Args:
            example_lengths: Length for every example in the dataset.
            batch_size: Number of items per batch.
            target_length: Soft ceiling for the total length of each batch.
            drop_last: Whether to discard incomplete batches.
            shuffle: Whether to randomly break ties prior to sorting by
                example length.
            hard_limit: Hard cap on batch length; offending batches are
                dropped entirely.

        Returns:
            A list of index lists, each describing one batch.
        """
        num_batches = len(example_lengths) // batch_size
        if num_batches == 0:
            log.warning(
                "Batch size is too large for the dataset, setting num_batches"
                f" to 1. ({len(example_lengths)=}, {batch_size=})"
            )
            num_batches = 1
        log.info(
            f"Will attempt to make {num_batches} batches with a target length"
            f" of {target_length} and a HARD LIMIT of {hard_limit}"
        )
        num_examples = len(example_lengths)
        if drop_last:
            num_keep_examples = (num_examples // num_batches) * num_batches
            if num_keep_examples == 0:
                return []  # No batches can be formed
            indices_to_group = list(range(num_keep_examples))
        else:
            indices_to_group = list(range(num_examples))
            if num_examples < num_batches:
                raise ValueError(
                    f"Number of examples ({num_examples}) is less than the "
                    f"requested number of batches ({num_batches}) and "
                    f"drop_last is False. Cannot create {num_batches} "
                    "batches."
                )

        if shuffle:
            random.shuffle(
                indices_to_group
            )  # Shuffle before sorting to break ties randomly

        # Sort indices by length in descending order
        indexed_lengths = sorted(
            [(i, example_lengths[i]) for i in indices_to_group],
            key=lambda x: x[1],
            reverse=True,
        )
        sorted_indices = [i for i, length in indexed_lengths]

        batches = [[] for _ in range(num_batches)]
        batch_lengths = [0] * num_batches
        items_per_batch = len(sorted_indices) // num_batches

        log.info(
            f"Starting fixed batch grouping of {len(sorted_indices)} examples "
            f"into {num_batches} batches, each with max {items_per_batch} "
            f"items and with a target length of {target_length}."
        )

        item_count_in_batches = [0] * num_batches
        indices_to_check = list(range(num_batches))

        for index in tqdm(
            sorted_indices,
            desc="Grouping examples into fixed batches",
        ):
            example_length = example_lengths[index]
            if hard_limit is not None and example_length > hard_limit:
                log.warning(
                    f"Example {index} with length {example_length} exceeds "
                    f"the target batch length of {hard_limit} and will "
                    "be skipped."
                )
                continue
            added_to_batch = False

            best_batch_idx = -1
            best_i = -1
            min_batch_length = float("inf")
            for i, batch_idx in enumerate(indices_to_check):
                if batch_lengths[batch_idx] < min_batch_length:
                    min_batch_length = batch_lengths[batch_idx]
                    best_batch_idx = batch_idx
                    best_i = i
                if batch_lengths[batch_idx] + example_length <= target_length:
                    batches[batch_idx].append(index)
                    batch_lengths[batch_idx] += example_length
                    item_count_in_batches[batch_idx] += 1
                    added_to_batch = True
                    if item_count_in_batches[batch_idx] == items_per_batch:
                        indices_to_check.pop(i)
                    break  # Placed in the first suitable batch

            if not added_to_batch:
                # If not added to any existing batch (should ideally not happen
                # much if max_total_length is reasonable and num_batches is not
                # excessively large)
                #
                # Fallback strategy: find the batch with the least current length
                # that still has space in terms of item count.
                if best_batch_idx != -1:
                    batches[best_batch_idx].append(index)
                    batch_lengths[best_batch_idx] += example_length
                    item_count_in_batches[best_batch_idx] += 1
                    added_to_batch = True
                    if item_count_in_batches[best_batch_idx] == items_per_batch:
                        indices_to_check.pop(best_i)
                else:
                    # As a last resort, if still not added and all batches are
                    # full in item count (should not happen if drop_last is True
                    # and logic is correct)
                    # Or if no batch has space within max_total_length (which
                    # might indicate max_total_length is too restrictive).
                    # Place in the first batch that is not yet item_per_batch
                    # full, even if it goes over max_total_length
                    # (we are prioritizing fixed batch count and size).
                    first_fit = indices_to_check[0]
                    batches[first_fit].append(index)
                    batch_lengths[first_fit] += example_length
                    item_count_in_batches[first_fit] += 1
                    added_to_batch = True
                    log.warning(
                        f"Item {index} with length"
                        f" {example_length} added to batch"
                        f" {first_fit} potentially exceeding"
                        " max_total_length, or due to fallback when no"
                        " suitable batch was found."
                    )
                    if item_count_in_batches[first_fit] == items_per_batch:
                        indices_to_check.pop(first_fit)

        length_first = item_count_in_batches[0]

        good_batches = []
        good_lengths = []
        for i, batch in enumerate(batches):
            # drop all batches that are above a HARD limit
            if (hard_limit is not None and batch_lengths[i] > hard_limit) or (
                len(batch) != items_per_batch
            ):
                log.warning(
                    f"Dropping batch {i} with length {batch_lengths[i]} and"
                    f" count {len(batch)} which exceeds hard limit of"
                    f" {hard_limit}."
                )
            else:
                good_batches.append(batch)
                good_lengths.append(batch_lengths[i])

        assert all(len(batch) == items_per_batch for batch in good_batches), (
            good_batches,
            length_first,
        )
        average_batch_length = sum(good_lengths) / len(good_lengths)
        min_batch_length = min(good_lengths)
        max_batch_length = max(good_lengths)
        log.info(
            "Done grouping examples into fixed batches. "
            f"Created {len(good_batches)} batches with an average length of "
            f"{average_batch_length:.2f} (target: {target_length}, "
            f"min: {min_batch_length}, max: {max_batch_length})."
        )

        return good_batches

    @classmethod
    def group_and_broadcast_batches(
        cls,
        *,
        example_lengths: list[int],
        batch_size: int,
        target_length: int,
        drop_last: bool,
        shuffle: bool,
        hard_limit: int | None,
    ) -> list[list[int]]:
        """Construct batches and share them across distributed ranks.

        Rank 0 executes the pure grouping logic, then broadcasts the resulting
        batches to every other process. If torch.distributed is unavailable or
        uninitialized, the method transparently falls back to the local
        grouping implementation.

        Args:
            example_lengths: Length metadata for every dataset element.
            batch_size: Number of examples per batch.
            target_length: Soft ceiling for total batch length.
            drop_last: Whether to keep incomplete batches.
            shuffle: Whether to introduce randomness prior to sorting.
            hard_limit: Optional hard cap on batch length.

        Returns:
            A list of batches, identical across every rank.
        """
        try:
            import torch.distributed as dist  # local import

            is_dist = dist.is_available() and dist.is_initialized()
        except Exception:  # pragma: no cover
            dist = None  # type: ignore
            is_dist = False

        if not is_dist:
            return cls._pure_group(
                example_lengths=example_lengths,
                batch_size=batch_size,
                target_length=target_length,
                drop_last=drop_last,
                shuffle=shuffle,
                hard_limit=hard_limit,
            )

        rank = dist.get_rank()  # type: ignore[attr-defined]
        if rank == 0:
            batches = cls._pure_group(
                example_lengths=example_lengths,
                batch_size=batch_size,
                target_length=target_length,
                drop_last=drop_last,
                shuffle=shuffle,
                hard_limit=hard_limit,
            )
        else:
            batches = None  # type: ignore
        obj_list: list[object] = [batches]
        dist.broadcast_object_list(obj_list, src=0)  # type: ignore[attr-defined]
        return obj_list[0]  # type: ignore[return-value]

    def __iter__(self):
        """Yield batches according to the attached sampler's ordering."""
        assert self.sampler is not None
        # Iterate over current epoch's batches in the sampler-defined order.
        for batch_idx in self.sampler:
            yield self.batches[batch_idx]
        if self.shuffle_every_epoch:
            self.batches = self.group_and_broadcast_batches(
                example_lengths=self.example_lengths,
                batch_size=self.group_args["batch_size"],
                target_length=self.group_args["target_length"],
                drop_last=self.group_args["drop_last"],
                shuffle=self.group_args["shuffle"],
                hard_limit=self.hard_limit,
            )

    def __len__(self):
        """Return the number of available batches for the current grouping."""
        if self.sampler is None:
            return len(self.batches)
        # Some sampler types (e.g., torch Sampler base) might not implement
        # __len__
        assert hasattr(self.sampler, "__len__")
        return len(cast(Sized, self.sampler))


class LengthSubsetDataset(Dataset):
    """
    Subset of a dataset at specified indices.

    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    dataset: Dataset
    indices: Sequence[int]

    def __init__(self, dataset: Dataset, indices: Sequence[int]) -> None:
        """Wrap a dataset so it can be indexed using a subset of indices."""
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx: int | list[int]):
        """Fetch a single item or list of items using subset-relative indices."""
        if isinstance(idx, list):
            return [self.dataset[self.indices[i]] for i in idx]
        return self.dataset[self.indices[idx]]

    def __getitems__(self, indices: List[int]):
        """Retrieve multiple items, delegating to the underlying dataset."""
        real_indices = [self.indices[i] for i in indices]
        if callable(getattr(self.dataset, "__getitems__", None)):
            return self.dataset.__getitems__(  # type: ignore[attr-defined]
                real_indices
            )
        else:
            return [self.dataset[idx] for idx in real_indices]

    def __len__(self):
        """Report the number of items exposed through the subset."""
        return len(self.indices)
