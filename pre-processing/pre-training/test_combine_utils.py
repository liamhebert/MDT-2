"""Various tests to ensure our combine function works."""

from combine_utils import combine_nodes_to_tree


def test_empty_data_list():
    assert combine_nodes_to_tree([]) is None


def test_single_root_node():
    data = [{"id": "1", "parent_id": None}]
    expected = {"id": "1", "tree": [], "data": {"id": "1", "parent_id": None}}
    assert combine_nodes_to_tree(data) == expected


def test_simple_parent_child():
    data = [{"id": "1", "parent_id": None}, {"id": "2", "parent_id": "1"}]
    expected = {
        "id": "1",
        "tree": [
            {"id": "2", "tree": [], "data": {"id": "2", "parent_id": "1"}}
        ],
        "data": {"id": "1", "parent_id": None},
    }
    assert combine_nodes_to_tree(data) == expected


def test_multiple_children():
    data = [
        {"id": "1", "parent_id": None},
        {"id": "2", "parent_id": "1"},
        {"id": "3", "parent_id": "1"},
    ]
    expected = {
        "id": "1",
        "tree": [
            {"id": "2", "tree": [], "data": {"id": "2", "parent_id": "1"}},
            {"id": "3", "tree": [], "data": {"id": "3", "parent_id": "1"}},
        ],
        "data": {"id": "1", "parent_id": None},
    }
    assert combine_nodes_to_tree(data) == expected


def test_max_depth_limit():
    data = [
        {"id": "1", "parent_id": None},
        {"id": "2", "parent_id": "1"},
        {"id": "3", "parent_id": "2"},
    ]
    expected = {
        "id": "1",
        "tree": [
            {"id": "2", "tree": [], "data": {"id": "2", "parent_id": "1"}}
        ],
        "data": {"id": "1", "parent_id": None},
    }
    assert combine_nodes_to_tree(data, max_depth=2) == expected


def test_no_root_node():
    data = [{"id": "2", "parent_id": "1"}]  # No root node
    assert combine_nodes_to_tree(data) is None


def test_node_with_missing_id():
    data = [{"parent_id": None}, {"id": "1", "parent_id": None}]
    expected = {"id": "1", "tree": [], "data": {"id": "1", "parent_id": None}}
    assert combine_nodes_to_tree(data) == expected


def test_complex_tree():
    data = [
        {"id": "1", "parent_id": None},
        {"id": "2", "parent_id": "1"},
        {"id": "3", "parent_id": "1"},
        {"id": "4", "parent_id": "2"},
        {"id": "5", "parent_id": "3"},
        {"id": "6", "parent_id": "4"},
    ]
    expected = {
        "id": "1",
        "tree": [
            {
                "id": "2",
                "tree": [
                    {
                        "id": "4",
                        "tree": [
                            {
                                "id": "6",
                                "tree": [],
                                "data": {"id": "6", "parent_id": "4"},
                            }
                        ],
                        "data": {"id": "4", "parent_id": "2"},
                    }
                ],
                "data": {"id": "2", "parent_id": "1"},
            },
            {
                "id": "3",
                "tree": [
                    {
                        "id": "5",
                        "tree": [],
                        "data": {"id": "5", "parent_id": "3"},
                    }
                ],
                "data": {"id": "3", "parent_id": "1"},
            },
        ],
        "data": {"id": "1", "parent_id": None},
    }
    assert combine_nodes_to_tree(data) == expected


def test_data_field_preserved():
    data = [{"id": "1", "parent_id": None, "name": "Root"}]
    expected = {
        "id": "1",
        "tree": [],
        "data": {"id": "1", "parent_id": None, "name": "Root"},
    }
    assert combine_nodes_to_tree(data) == expected


def test_max_depth_zero():
    data = [{"id": "1", "parent_id": None}, {"id": "2", "parent_id": "1"}]
    assert combine_nodes_to_tree(data, max_depth=0) is None
