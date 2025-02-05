"""Utilities for 2-combine_and_compress_trees.py, so that we can test them."""

# TODO(liamhebert): The fact that we have to do this, because you can't import
# files that start with a number, is silly. We should fix our naming convention.


def combine_nodes_to_tree(data_list: list[dict], max_depth=None) -> dict | None:
    """
    Converts a list of dictionaries with 'id' and 'parent_id' fields into a
    nested tree structure, assuming there is only one root node in the list.
    Limits the depth of the tree.

    Args:
        data_list: A list of dictionaries, where each dictionary has 'id' and
        'parent_id' fields. Assumes there is exactly one item with
        parent_id=None (root node).
        max_depth: The maximum depth of the tree to build (optional). Nodes
        deeper than this are not included. If None, there is no depth limit.

    Returns:
        A dictionary representing the root node of the tree, up to the specified
        max_depth, or None if no root is found. The root node has 'id' and
        'tree' (a list of children nodes) fields. Returns None if no root node.
    """

    node_map: dict[str, dict] = {}
    children_map: dict[str, list[dict]] = {}
    root_node = None

    # Edge case if max_depth is 0, for some reason.
    if max_depth == 0:
        return None

    for item in data_list:
        node_id = item.get("id")
        parent_id = item.get("parent_id")

        if node_id is None:
            continue

        node = {"id": node_id, "tree": [], "data": item}
        node_map[node_id] = node

        if parent_id is not None:
            if parent_id not in children_map:
                children_map[parent_id] = []
            children_map[parent_id].append(node)
        else:
            root_node = node  # Found the root node

    if root_node is None:
        # Handle case where no root node is found (though we should always have
        # one)
        return None

    def build_tree_recursive(nodes: list[dict], current_depth: int):
        if max_depth is not None and current_depth >= max_depth:
            return []

        tree_nodes = []
        for node in nodes:
            children = children_map.get(node["id"], [])
            node["tree"] = build_tree_recursive(children, current_depth + 1)
            tree_nodes.append(node)
        return tree_nodes

    # create new root node
    children = children_map.get(root_node["id"], [])
    # Root nodes are at depth 1
    root_node["tree"] = build_tree_recursive(children, 1)

    return root_node
