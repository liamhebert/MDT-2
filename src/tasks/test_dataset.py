"""Tests for the generic dataset class."""

import pathlib

import pytest
import torch

from data.collated_datasets import ContrastiveTaskDataset
from data.collated_datasets import NodeBatchedDataDataset
from data.types import ContrastiveLabels
from data.types import Labels
from datamodule import DataModule


class DummyNodeTaskDataset(NodeBatchedDataDataset):
    """Dummy task dataset to simulate a node dataset."""

    def has_node_labels(self) -> bool:
        """Indicating the dataset has node labels."""
        return True

    def retrieve_label(self, tree: dict) -> dict[str, bool | int]:
        """Return node labels, which is a boolean label and mask."""
        if tree["label"] == "NA":
            return {Labels.Ys: -100}
        return {Labels.Ys: tree["label"] != "NA"}


class DummyGraphTaskDataset(ContrastiveTaskDataset):
    """Dummy task dataset to simulate a contrastive dataset."""

    def has_graph_labels(self) -> bool:
        """Indicating the dataset has graph labels."""
        return True

    def retrieve_label(self, tree: dict) -> dict[str, bool | int]:
        """Return constrastive graph labels, which is a boolean y, mask,
        hard y.
        """
        return {
            ContrastiveLabels.Ys: tree["label"] != "NA",
            ContrastiveLabels.HardYs: tree["label"] != "NA",
        }


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
        assert torch.sum(x.y[Labels.Ys] != -100) == 1

    assert len(split_dataset) == 15
    assert len(data) == 15
    split_dataset.collate_fn(data)


def test_node_process(dataset: DummyNodeTaskDataset):
    """Test the process method for a dataset without graph splitting."""
    dataset.process()
    data = []
    for x in dataset:
        data += [x]

    assert len(dataset) == 10
    assert len(data) == 10
    dataset.collate_fn(data)


def test_graph_dataset_process(graph_dataset: DummyGraphTaskDataset):
    """Test the process method for a contrastive dataset."""
    graph_dataset.process()
    data = []
    for x in graph_dataset:
        data += [x]

    assert len(graph_dataset) == 10
    assert len(data) == 10
    res = graph_dataset.collate_fn(data)

    assert res["y"][ContrastiveLabels.Ys].shape == (10,)
    assert res["y"][ContrastiveLabels.HardYs].shape == (10,)


def test_cached_dataset(dataset: DummyNodeTaskDataset):
    """Test the process method for a dataset without graph splitting."""
    dataset.process()
    dataset.populate_cache(force_all=True)
    data = []
    print(len(dataset))
    for x in dataset:
        data += [x]

    assert len(dataset) == 10
    assert len(data) == 10
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
    json_data = {
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

    data = dataset.process_graph(json_data)

    assert data.text is not None
    assert data.edge_index.tolist() == [[0, 1, 2], [0, 0, 0]]
    assert data.y[Labels.Ys].tolist() == [True, True, -100]
    assert data.image_mask.tolist() == [False, False, False]
    assert data.distance.tolist() == [
        [[0, 0], [0, 1], [0, 1]],
        [[1, 0], [0, 0], [1, 1]],
        [[1, 0], [1, 1], [0, 0]],
    ]
    torch.testing.assert_close(data.attn_bias, torch.zeros([3, 3]))
    assert data.in_degree.tolist() == [2, 1, 1]
    assert data.out_degree.tolist() == [2, 1, 1]

    assert data.text["input_ids"].shape == (3, 512)
    assert data.text["attention_mask"].shape == (3, 512)


def test_truncated_distance(dataset: DummyNodeTaskDataset):
    """
    Testing whether we can correctly map distances to the correct indices.
    """
    json_data = {
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

    data = dataset.process_graph(json_data)

    distance_indices = data.distance_index
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
