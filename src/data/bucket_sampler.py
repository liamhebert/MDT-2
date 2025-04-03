from torch.utils.data import BatchSampler, Sampler
import random
from typing import List, Sequence
from torch.utils.data import Dataset
from tqdm import tqdm
from utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


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
        shuffle (bool): Whether to shuffle the dataset indices *before* sorting by
            length (affects tie-breaking).
        drop_last (bool): Whether to drop the last incomplete batch if the total
            number of examples is not perfectly divisible by num_batches.
    """

    sampler: Sampler[int] | None = None

    def __init__(
        self,
        example_lengths: list[int],
        batch_size: int = 1,
        shuffle: bool = True,
        shuffle_every_epoch: bool = True,
        drop_last: bool = True,
        sampler: Sampler[int] | None = None,
        max_total_length: int | None = None,
    ):

        self.batch_size = batch_size  # Have to do this for lightning
        self.example_lengths = example_lengths
        if max_total_length is None:
            # We take the mean length of examples as the max total length
            mean_total_length = sum(example_lengths) // len(example_lengths)
            max_total_length = mean_total_length * batch_size
            log.info(
                "Since max_total_length is None, we will use the mean "
                f"length of examples ({mean_total_length}) * batch_size "
                f"({batch_size}) as the max total length ({max_total_length})."
            )

        num_batches = len(example_lengths) // batch_size
        if num_batches == 0:
            log.warning(
                "Batch size is too large for the dataset, setting num_batches "
                "to 1."
                f"({len(example_lengths)=}, {batch_size=})"
            )
            num_batches = 1

        self.shuffle_every_epoch = shuffle_every_epoch
        if shuffle_every_epoch:
            assert shuffle, (
                "shuffle_every_epoch requires shuffle=True, as it would be"
                " redundant otherwise."
            )
        log.info(
            f"Will attempt to make {num_batches} batches with a target "
            f"length of {max_total_length}."
        )

        # Group into fixed batches
        self.group_args = {
            "example_lengths": example_lengths,
            "num_batches": num_batches,
            "target_length": max_total_length,
            "drop_last": drop_last,
            "shuffle": shuffle,
        }
        self.batches = self._group_into_fixed_batches(**self.group_args)
        if sampler is not None:
            assert hasattr(sampler, "__len__")
            assert len(sampler) <= len(self.batches)  # type: ignore
            self.sampler = sampler

    def set_sampler(self, sampler: Sampler[int]) -> None:
        assert hasattr(sampler, "__len__")
        assert len(sampler) <= len(self.batches)  # type: ignore
        self.sampler = sampler

    def _group_into_fixed_batches(
        self,
        example_lengths: list[int],
        num_batches: int,
        target_length: int,
        drop_last: bool,
        shuffle: bool,
    ) -> List[List[int]]:
        """
        Groups indices into a fixed number of batches, trying to maximize length
        within each batch, with equal number of items per batch
        (if drop_last=True).

        Returns a list of lists, where each inner list is a batch of indices.
        """
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
            added_to_batch = False
            for i, batch_idx in enumerate(indices_to_check):
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

                best_batch_idx = -1
                best_i = -1
                min_batch_length = float("inf")
                for i, batch_idx in enumerate(indices_to_check):
                    if batch_lengths[batch_idx] < min_batch_length:
                        min_batch_length = batch_lengths[batch_idx]
                        best_batch_idx = batch_idx
                        best_i = i

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
                    for i, batch_idx in enumerate(indices_to_check):
                        batches[batch_idx].append(index)
                        batch_lengths[batch_idx] += example_length
                        item_count_in_batches[batch_idx] += 1
                        added_to_batch = True
                        log.warning(
                            f"Item {index} with length"
                            f" {example_length} added to batch"
                            f" {batch_idx} potentially exceeding"
                            " max_total_length, or due to fallback when no"
                            " suitable batch was found."
                        )
                        if item_count_in_batches[batch_idx] == items_per_batch:
                            indices_to_check.pop(i)
                        break
                    if not added_to_batch:
                        # This should really not happen if drop_last=True and
                        # logic is correct.
                        raise RuntimeError(
                            f"Failed to add item {index} to any batch. This "
                            "should not happen with drop_last=True and correct "
                            "logic."
                        )

        length_first = item_count_in_batches[0]
        assert all(
            item_count == length_first for item_count in item_count_in_batches
        ), (item_count_in_batches, length_first)

        average_batch_length = sum(batch_lengths) / len(batch_lengths)
        min_batch_length = min(batch_lengths)
        max_batch_length = max(batch_lengths)
        log.info(
            "Done grouping examples into fixed batches. "
            f"Created {len(batches)} batches with an average length of "
            f"{average_batch_length:.2f} (target: {target_length}, "
            f"min: {min_batch_length}, max: {max_batch_length})."
        )

        return batches

    def __iter__(self):
        assert self.sampler is not None
        # TODO(liamhebert): If we wanted to have shuffled epochs, we could shuffle
        # here, and hopefully have the same number of batches. If we dont, we
        # would need to recreate the sampler, which could be complicated due to
        # how lightning injects distributed samplers.
        for batch_idx in self.sampler:
            yield self.batches[batch_idx]
        if self.shuffle_every_epoch:
            self.batches = self._group_into_fixed_batches(**self.group_args)

    def __len__(self):
        return len(self.batches) if self.sampler is None else len(self.sampler)


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
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx: int | list[int]):
        if isinstance(idx, list):
            return [self.dataset[self.indices[i]] for i in idx]
        return self.dataset[self.indices[idx]]

    def __getitems__(self, indices: List[int]):
        real_indices = [self.indices[i] for i in indices]
        if callable(getattr(self.dataset, "__getitems__", None)):
            return self.dataset.__getitems__(  # type: ignore[attr-defined]
                real_indices
            )
        else:
            return [self.dataset[idx] for idx in real_indices]

    def __len__(self):
        return len(self.indices)
