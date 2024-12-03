import pathlib

import pytest
import torch

from tasks.dataset import NodeBatchedDataDataset


class DummyTaskDataset(NodeBatchedDataDataset):
    def retrieve_label(self, tree: dict) -> tuple[int, bool]:
        return tree["label"] != "NA", tree["label"] != "NA"


@pytest.fixture(scope="function")
def dataset(tmp_path: pathlib.Path):
    (tmp_path / "processed").mkdir()
    return DummyTaskDataset(
        raw_graph_path="test10.json",
        image_path="",
        root="tasks/sample_test_data",
        output_graph_path=tmp_path,
        max_distance_length=2,
    )


@pytest.fixture(scope="function")
def split_dataset(tmp_path: pathlib.Path):
    (tmp_path / "processed").mkdir()
    return DummyTaskDataset(
        raw_graph_path="test10.json",
        image_path="",
        root="tasks/sample_test_data",
        output_graph_path=tmp_path,
        split_graphs=True,
    )


def test_process(dataset: DummyTaskDataset):
    assert len(dataset) == 10

    k = 0
    num_images = 0
    for x in dataset:
        k += 1
        if hasattr(x, "images"):
            num_images += x["images"]["pixel_values"].shape[0]


def test_flatten_graph(dataset: DummyTaskDataset):
    tree = {
        "id": "root",
        "images": ["image1.png"],
        "distances": {"root": [0, 0], "child1": [0, 1], "child2": [0, 1]},
        "data": {"title": "Root Title", "body": "Root Body", "label": "A"},
        "tree": [
            {
                "id": "child1",
                "images": [],
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
    assert flattened_graph["y"] == [True, True, False]
    assert flattened_graph["y_mask"] == [True, True, False]
    assert flattened_graph["distances"] == [
        {"root": [0, 0], "child1": [0, 1], "child2": [0, 1]},
        {"root": [1, 0], "child1": [0, 0], "child2": [1, 1]},
        {"root": [1, 0], "child1": [1, 1], "child2": [0, 0]},
    ]
    assert flattened_graph["text"] == [
        "Title: Root Title\n Body: Root Body",
        "Comment: Child 1 Body",
        "Comment: Child 2 Body",
    ]


def test_process_graph(dataset: DummyTaskDataset):
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
    assert data.y.tolist() == [True, True, False]
    assert data.y_mask.tolist() == [True, True, False]
    assert data.image_mask.tolist() == [False, False, False]
    assert data.distance.tolist() == [
        [[0, 0], [0, 1], [0, 1]],
        [[1, 0], [0, 0], [1, 1]],
        [[1, 0], [1, 1], [0, 0]],
    ]
    torch.testing.assert_close(data.attn_bias, torch.zeros([3, 3]))
    assert data.in_degree.tolist() == [2, 1, 1]
    assert data.out_degree.tolist() == [2, 1, 1]

    assert data.text["input_ids"].shape == (3, 256)
    assert data.text["attention_mask"].shape == (3, 256)


def test_truncated_distance(dataset: DummyTaskDataset):
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
