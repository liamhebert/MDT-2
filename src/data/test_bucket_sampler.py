from data.bucket_sampler import LengthGroupedSampler


def test_length_grouped_sampler():
    """Test the length grouped sampler using mean."""
    example_lengths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    batch_size = 3
    sampler = LengthGroupedSampler(example_lengths, batch_size=batch_size)
    batches = sampler.batches

    assert len(batches) == 3, [[example_lengths[i] for i in x] for x in batches]
    assert sorted([len(x) for x in batches], reverse=True) == [3, 3, 3]

    example_lengths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    batch_size = 5
    sampler = LengthGroupedSampler(example_lengths, batch_size=batch_size)
    batches = sampler.batches

    assert len(batches) == 2, [[example_lengths[i] for i in x] for x in batches]
    assert [len(x) for x in batches] == [5, 5]


def test_length_grouped_sampler_set_max_length():
    """Test the length grouped sampler using set max."""

    example_lengths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    batch_size = 2
    sampler = LengthGroupedSampler(
        example_lengths, batch_size=batch_size, max_total_length=11
    )
    batches = sampler.batches

    assert len(batches) == 5, [[example_lengths[i] for i in x] for x in batches]
    assert sorted([len(x) for x in batches], reverse=True) == [2, 2, 2, 2, 2]

    other_sampler = LengthGroupedSampler(
        example_lengths, batch_size=batch_size, max_total_length=15
    )
    other_batches = other_sampler.batches
    assert other_batches != batches
