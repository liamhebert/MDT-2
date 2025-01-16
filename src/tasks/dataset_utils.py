"""Various utilities to help with dataset processing that are reusable."""

import copy
import re
import unicodedata

import emoji


def compute_relative_distance(tree) -> dict:
    """Computes the relative distance of each node in the tree to all other
    nodes.

    This function computes distance by measuring the number of up hops and down
    hops between nodes in the tree. The distance is stored at the node level as a
    dictionary of node_id -> [up, down] where up is the number of hops to the
    common parent and down is the number of hops to the leaf node.

    For instance, given a tree:
    {
            "id": 1,
            "tree": [
                {"id": 2, "tree": []},
                {"id": 3, "tree": [{"id": 4, "tree": []}]},
            ],
    }

    The distance between 1 -> 4 is [0, 2] because 1 is the root and 4 is two
    levels down, whereas 4 -> 1 is [2, 0] because 1 is two levels up from 4.
    Similarly, 2 -> 4 is [1, 2] because 2 is one level up (1) and two down
    (1 -> 3, 3 -> 4).

    Args:
        tree (dict): A nested dictionary representing the conversation tree.
        Each node in the tree must contain a unique "id" field and a "tree" field
        containing the nested tree.

    Returns:
        The same tree with nodes updated to include the "distances" field. Note
        this is done in-place, so the return is not necessary.
    """
    UP = 0
    DOWN = 1

    # The algorithm is a two-step process:
    # First, we calculate the distance depth first from the root to all nodes.
    # This traverses each branch individually setting the UP distance of a node
    # to the number of depth recursive calls it took to get there, and copying
    # the down distance of all the children nodes, adding 1 to each value.

    # Second, because the first process is depth first, we need to spread the
    # information from adjacent branches. Because the root node is the only node
    # with complete information, each node is missing a distance contained in the
    # root copies that distance with an additional hop to the root. This is also
    # done depth-first, but starting with complete information.

    def depth_first(node, depths={}) -> dict:
        # To avoid in-place modification, we copy the running distance dictionary.
        distances = copy.deepcopy(depths)
        # All nodes that we've already seen in this recursion are one above us
        # from the previous iteration
        for key in distances.keys():
            distances[key][UP] += 1
        # The current node is 0, 0
        distances[node["id"]] = [0, 0]

        for x in node["tree"]:
            # Do this recursion for all children
            val = depth_first(x, distances)
            # All new nodes we've picked up are one further below us from the
            # children.
            for key, value in val.items():
                # Nodes we've already seen (ie: parents of this node) are not
                # updated
                if key not in distances:
                    distances[key] = [value[UP], value[DOWN] + 1]

        node["distances"] = distances
        return distances

    def spread_adj_branch_distance(node, depths={}) -> None:
        for key, value in depths.items():
            # For all nodes not directly in this branch, we are one more away from
            # the root.
            if key not in node["distances"]:
                node["distances"][key] = [value[UP] + 1, value[DOWN]]
        for x in node["tree"]:
            spread_adj_branch_distance(x, node["distances"])

    depth_first(tree)
    spread_adj_branch_distance(tree)

    return tree


def clean_text(string: str) -> str:
    """Cleans and normalizes text for processing.

    This function performs the following operations:
    - Replaces all markdown links (ex: "[ref](url)") with [LINK1] ref [LINK2]
    - Replaces all remaining urls with complicated paths with
        [LINK1] <url>.com [LINK2], keeping just the top-level domain name.
    - Converts emojis to text representations (ie: 😂 -> :face_with_tears_of_joy:
    - Removes accented characters
    - Removes all non-ascii characters
    - Removes trailing whitespace

    Args:
        string (str): The string to clean and normalize.

    Returns:
        str: The cleaned and normalized string.
    """
    # Replace all markdown links with [LINK1] reference_text [LINK2]
    # (ie: [link](url) -> [LINK1] link [LINK2])
    markdown_regex = re.compile(
        "\[([\w\s\d]+)\]\(((?:\/|https?:\/\/)[\w\d./?=#]+)\)"
    )
    x = markdown_regex.sub("[LINK1] \g<1> [LINK2]", string)

    # Replace all urls with [LINK1] <url>.com [LINK2]
    all_url_regex = re.compile(
        "https?:\\/\\/(?:www\\.)?([-a-zA-Z0-9@:%._\\+~#=]{1,256})\\."
        "[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&\\/=]*)"
    )
    x = all_url_regex.sub("[LINK1] \g<1>.com [LINK2]", x)

    # Convert emojis to text representations (ie: 😂 -> :face_with_tears_of_joy:)
    x = emoji.demojize(x)
    # Remove accented characters
    x = (
        unicodedata.normalize("NFKD", x)
        .encode("ASCII", "ignore")
        .decode("ASCII")
    )
    # Remove all non-ascii characters
    x = "".join(i for i in x if ord(i) < 128)

    # Remove trailing whitespace
    x = x.strip()

    return x
