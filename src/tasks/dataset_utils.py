import copy
import re
import unicodedata

import emoji
import torch


def compute_relative_distance(tree) -> None:
    UP = 0
    DOWN = 1

    def depth_first(node, depths={}) -> dict:
        # calculate the depth of the current node to all other nodes
        distances = copy.deepcopy(depths)
        # All nodes that we've already seen in this recursion are one above us from
        # the previous iteration
        for key in distances.keys():
            distances[key][UP] += 1
        # the current node is 0, 0
        distances[node["id"]] = [0, 0]

        for x in node["tree"]:
            # Do this recursion for all children
            val = depth_first(x, distances)
            # All new nodes we've picked up are one below us from the children
            for key, value in val.items():
                # Nodes we've already seen (ie: parents of this node) are not updated
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


def clean_text(string: str) -> str:
    # Replace all markdown links with [LINK1] reference_text [LINK2]
    # (ie: [link](url) -> [LINK1] link [LINK2])
    markdown_regex = re.compile(
        "\[([\w\s\d]+)\]\(((?:\/|https?:\/\/)[\w\d./?=#]+)\)"
    )
    x = markdown_regex.sub("[LINK1] \g<1> [LINK2]", string)

    # Replace all urls with [LINK1] <url>.com [LINK2]
    all_url_regex = re.compile(
        "https?:\\/\\/(?:www\\.)?([-a-zA-Z0-9@:%._\\+~#=]{1,256})\\.[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&\\/=]*)"
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
