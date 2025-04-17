"""Tests for the generic dataset class."""

import pathlib
import pytest
import torch
import numpy as np

from data.collated_datasets import ContrastiveTaskDataset
from data.collated_datasets import NodeBatchedDataDataset
from data.types import ContrastiveLabels
from data.types import Labels
from datamodule import DataModule


class DummyNodeTaskDataset(NodeBatchedDataDataset):
    """Dummy task dataset to simulate a node dataset."""

    @property
    def has_node_labels(self) -> bool:
        """Indicating the dataset has node labels."""
        return True

    def retrieve_label(self, data: dict) -> dict[str, bool | int]:
        """Return node labels, which is a boolean label and mask."""
        assert "label" in data, data.keys()
        if data["label"] == "NA":
            return {Labels.Ys: -100}
        return {Labels.Ys: data["label"] != "NA"}


class DummyGraphTaskDataset(ContrastiveTaskDataset):
    """Dummy task dataset to simulate a contrastive dataset."""

    @property
    def has_graph_labels(self) -> bool:
        """Indicating the dataset has graph labels."""
        return True

    def retrieve_label(self, data: dict) -> dict[str, bool | int]:
        """Return constrastive graph labels, which is a boolean y, mask,
        hard y.
        """
        assert "label" in data, data.keys()
        return {
            ContrastiveLabels.Ys: data["label"] != "NA",
            ContrastiveLabels.HardYs: data["label"] != "NA",
        }


def recursive_update(data: dict) -> dict:
    """Utility function to fix the data structure, while not deleting the
    original formatted data.

    TODO(liamhebert): Remove this function when the data is fixed in the
    original dataset.
    """
    data.update(data["data"])
    del data["data"]
    for child in data["tree"]:
        recursive_update(child)
    return data


@pytest.fixture(scope="function")
def dataset(tmp_path: pathlib.Path):
    """Generate a dummy node dataset without graph splitting."""
    (tmp_path / "processed").mkdir(exist_ok=True)
    return DummyNodeTaskDataset(
        raw_graph_path="test10",
        root="tasks/sample_test_data",
        output_graph_path=tmp_path,
        max_distance_length=2,
        split_graphs=False,
    )


@pytest.fixture(scope="function")
def split_dataset(tmp_path: pathlib.Path):
    """Generate a dummy node dataset with graph splitting."""
    (tmp_path / "processed").mkdir(exist_ok=True)
    return DummyNodeTaskDataset(
        raw_graph_path="test10",
        root="tasks/sample_test_data",
        output_graph_path=tmp_path,
        split_graphs=True,
    )


@pytest.fixture(scope="function")
def graph_dataset(tmp_path: pathlib.Path):
    """Generate a dummy graph constrastive dataset without graph splitting."""
    (tmp_path / "processed").mkdir(exist_ok=True)
    return DummyGraphTaskDataset(
        raw_graph_path="test10",
        root="tasks/sample_test_data",
        output_graph_path=tmp_path,
        split_graphs=False,
    )


def test_process_split(split_dataset: DummyNodeTaskDataset):
    """Test the process method for a dataset with graph splitting."""
    split_dataset.process()
    data = []
    for x in split_dataset:
        data += [x]
        assert torch.sum(x[0]["y"][Labels.Ys] != -100) == 1

    assert len(split_dataset) == 16
    assert len(data) == 15
    split_dataset.collate_fn(data)


def test_node_process(dataset: DummyNodeTaskDataset):
    """Test the process method for a dataset without graph splitting."""
    dataset.process()
    data = []
    for x in dataset:
        data += [x]

    assert len(dataset) == 10
    assert len(data) == 11
    res = dataset.collate_fn(data)

    assert len(res["x"]["image_input"]["pixel_values"]) == 5, res["x"][
        "image_input"
    ]


def test_graph_dataset_process(graph_dataset: DummyGraphTaskDataset):
    """Test the process method for a contrastive dataset."""
    graph_dataset.process()
    data = []
    for x in graph_dataset:
        data += [x]

    assert len(graph_dataset) == 10
    assert len(data) == 11
    res = graph_dataset.collate_fn(data)

    assert res["y"][ContrastiveLabels.Ys].shape == (11,)
    assert res["y"][ContrastiveLabels.HardYs].shape == (11,)
    assert len(res["x"]["image_input"]["pixel_values"]) == 5, res["x"][
        "image_input"
    ]


def test_graph_process_debug(tmp_path: pathlib.Path):
    (tmp_path / "processed").mkdir(exist_ok=True)
    dataset = DummyGraphTaskDataset(
        raw_graph_path="test10",
        root="tasks/sample_test_data",
        output_graph_path=str(tmp_path),
        split_graphs=False,
        debug=3,
    )
    dataset.process()

    data = []
    for x in dataset:
        data += [x]

    assert len(data) == 4
    assert len(dataset) == 3

    res = dataset.collate_fn(data)
    assert res["y"][ContrastiveLabels.Ys].shape == (4,)
    assert res["y"][ContrastiveLabels.HardYs].shape == (4,)

    for indices in [dataset.train_idx, dataset.valid_idx, dataset.test_idx]:
        data_2 = []
        sizes = dataset.get_sizes(indices)
        assert len(sizes) == len(indices)
        for idx in indices:
            data_2 += [dataset[idx]]


@pytest.mark.parametrize("group_size", [2, 4])
def test_grouped_graphs(tmp_path: pathlib.Path, group_size: int):
    (tmp_path / "processed").mkdir(exist_ok=True)
    dataset = DummyGraphTaskDataset(
        raw_graph_path="test10",
        root="tasks/sample_test_data",
        output_graph_path=str(tmp_path),
        split_graphs=False,
        group_size=group_size,
        train_size=6,
        valid_size=2,
        test_size=2,
        debug=10,
    )
    dataset.process()

    num_groups = 6 // group_size + 2 // group_size + 2 // group_size

    data = []
    for x in dataset:
        data += [x]
        assert len(x) == group_size

    assert len(data) == num_groups
    for indices in [dataset.train_idx, dataset.valid_idx, dataset.test_idx]:
        data_2 = []
        sizes = dataset.get_sizes(indices)
        assert len(sizes) == len(indices)
        for idx in indices:
            data_2 += [dataset[idx]]


def test_grouped_graphs_read(tmp_path: pathlib.Path):
    (tmp_path / "processed").mkdir(exist_ok=True)
    dataset = DummyGraphTaskDataset(
        raw_graph_path="test10",
        root="tasks/sample_test_data",
        output_graph_path=str(tmp_path),
        split_graphs=False,
        group_size=2,
    )
    dataset.process()

    data = []
    for x in dataset:
        data += [x]
        assert len(x) == 2

    for indices in [dataset.train_idx, dataset.valid_idx, dataset.test_idx]:
        data_2 = []
        sizes = dataset.get_sizes(indices)
        assert len(sizes) == len(indices)
        for idx in indices:
            data_2 += [dataset[idx]]


def test_cached_dataset(dataset: DummyNodeTaskDataset):
    """Test the process method for a dataset without graph splitting."""
    dataset.process()
    data = []
    print(len(dataset))
    for x in dataset:
        data += [x]

    assert len(dataset) == 10
    assert len(data) == 11
    dataset.collate_fn(data)


def test_flatten_graph(dataset: DummyNodeTaskDataset):
    """
    Test the flatten graph method, which converts a json tree to a flat graph
    with distances.
    """
    tree = {
        "id": "root",
        "images": ["image1.png"],
        "distances": {"root": [0, 0], "child1": [0, 1], "child2": [0, 1]},
        "rotary_position": [0, 0],
        "data": {"title": "Root Title", "body": "Root Body", "label": "A"},
        "tree": [
            {
                "id": "child1",
                "images": [],
                "rotary_position": [0, 1],
                "distances": {
                    "root": [1, 0],
                    "child1": [0, 0],
                    "child2": [1, 1],
                },
                "data": {"body": "Child 1 Body", "label": "B"},
                "tree": [],
            },
            {
                "id": "child2",
                "images": ["image2.png"],
                "rotary_position": [1, 1],
                "distances": {
                    "root": [1, 0],
                    "child1": [1, 1],
                    "child2": [0, 0],
                },
                "data": {"body": "Child 2 Body", "label": "NA"},
                "tree": [],
            },
        ],
    }
    tree = recursive_update(tree)

    flattened_graph = dataset.flatten_graph(tree)

    assert flattened_graph["images"] == ["image1.png", None, "image2.png"]
    assert flattened_graph["id"] == ["root", "child1", "child2"]
    assert flattened_graph["parent_id"] == ["root", "root", "root"]
    assert flattened_graph["is_root"] == [True, False, False]
    assert [x[Labels.Ys] for x in flattened_graph["y"]] == [True, True, -100]
    assert flattened_graph["distances"] == [
        {"root": [0, 0], "child1": [0, 1], "child2": [0, 1]},
        {"root": [1, 0], "child1": [0, 0], "child2": [1, 1]},
        {"root": [1, 0], "child1": [1, 1], "child2": [0, 0]},
    ]
    assert flattened_graph["rotary_position"] == [[0, 0], [0, 1], [1, 1]]
    assert flattened_graph["text"] == [
        "Title: Root Title\nBody: Root Body",
        "Comment: Child 1 Body",
        "Comment: Child 2 Body",
    ]


def test_process_graph(dataset: DummyNodeTaskDataset):
    """Test the end-to-end process of an individual graph."""
    tree = {
        "id": "root",
        "images": [],
        "data": {"title": "Root Title", "body": "Root Body", "label": "A"},
        "tree": [
            {
                "id": "child1",
                "images": [],
                "data": {"body": "Child 1 Body", "label": "B"},
                "tree": [],
            },
            {
                "id": "child2",
                "images": [],
                "data": {"body": "Child 2 Body", "label": "NA"},
                "tree": [],
            },
        ],
    }
    tree = recursive_update(tree)

    data = dataset.process_graph(tree)

    assert data["text"] is not None
    assert data["y"][Labels.Ys].tolist() == [True, True, -100]
    assert data["image_mask"].tolist() == [False, False, False]
    assert data["distance"].tolist() == [
        [[0, 0], [0, 1], [0, 1]],
        [[1, 0], [0, 0], [1, 1]],
        [[1, 0], [1, 1], [0, 0]],
    ]

    assert data["out_degree"].tolist() == [2, 0, 0]

    assert data["text"]["input_ids"].shape == (3, 512)
    assert data["text"]["attention_mask"].shape == (3, 512)
    np.testing.assert_allclose(
        data["rotary_position"],
        np.array([[0, 0], [0, 1], [1, 1]], dtype=np.uint8),
    )


@pytest.mark.skip("skipping since distance index is removed.")
def test_truncated_distance(dataset: DummyNodeTaskDataset):
    """
    Testing whether we can correctly map distances to the correct indices.
    """
    tree = {
        "id": "root",
        "images": [],
        "data": {"title": "Root Title", "body": "Root Body", "label": "A"},
        "tree": [
            {
                "id": "child1",
                "images": [],
                "data": {"body": "Child 1 Body", "label": "B"},
                "tree": [
                    {
                        "id": "child2",
                        "images": [],
                        "data": {"body": "Child 2 Body", "label": "NA"},
                        "tree": [
                            {
                                "id": "child3",
                                "images": [],
                                "data": {"body": "Child 3 Body", "label": "NA"},
                                "tree": [],
                            }
                        ],
                    },
                ],
            },
        ],
    }
    tree = recursive_update(tree)

    data = dataset.process_graph(tree)

    distance_indices = data["distance_index"]
    # 0 -> 0 == 1 -> 1 == 2 -> 2
    assert (
        distance_indices[0][0]
        == distance_indices[1][1]
        == distance_indices[2][2]
    )
    # 0 -> 1 == 1 -> 2
    assert distance_indices[0][1] == distance_indices[1][2]
    # 1 -> 0 == 2 -> 1
    assert distance_indices[1][0] == distance_indices[2][1]
    # 0 -> 2 != 2 -> 0
    assert distance_indices[0][2] != distance_indices[2][0]

    # Since we clamp the distance to 2, the distance 0 -> 2 == 0 -> 3
    assert distance_indices[0][2] == distance_indices[0][3]


@pytest.mark.parametrize("case", ["node", "graph"])
def test_wrapped_datamodule(
    case: str,
    dataset: DummyNodeTaskDataset,
    graph_dataset: DummyGraphTaskDataset,
):
    """Tests whether a higher-level datamodule can correctly read from a
    task dataset.
    """
    if case == "node":
        data = dataset
    else:
        data = graph_dataset

    dm = DataModule(
        dataset=data,
        train_batch_size=2,
        test_batch_size=1,
        num_workers=1,
        pin_memory=False,
        cache_dataset=False,
        use_length_grouped_sampler=False,
    )

    dm.prepare_data()
    dm.setup("fit")
    train_dl = dm.train_dataloader()
    val_dl = dm.val_dataloader()
    test_dl = dm.test_dataloader()

    assert len(train_dl) == 4  # 7 / 2 = 3.5 -> 4
    assert len(val_dl) == 2
    assert len(test_dl) == 2

    for dl in [val_dl, test_dl]:
        for x in dl:
            assert x["x"]["num_total_graphs"] == 1

    sizes = []
    for x in train_dl:
        sizes += [x["x"]["num_total_graphs"]]

    assert all(x == 2 for x in sizes[:-1])
    assert sizes[-1] == 1


@pytest.mark.parametrize("case", ["set_length", "multiplier"])
def test_length_sampler_datamodule_smoketest(
    case,
    graph_dataset: DummyGraphTaskDataset,
):
    if case == "set_length":
        max_total_length = 5
        max_length_multipler = None
    else:
        max_total_length = None
        max_length_multipler = 3

    dm = DataModule(
        dataset=graph_dataset,
        train_batch_size=2,
        test_batch_size=1,
        num_workers=1,
        pin_memory=False,
        cache_dataset=True,
        use_length_grouped_sampler=True,
        max_total_length=max_total_length,
        max_length_multiplier=max_length_multipler,
    )

    dm.prepare_data()
    dm.setup("fit")
    train_dl = dm.train_dataloader()
    for x in train_dl:
        ...
