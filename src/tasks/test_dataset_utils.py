import torch

import tasks.dataset_utils as dut


class TestCleanTest:
    def test_clean_text_removes_extra_whitespace(self):
        # Test that clean_text removes leading and trailing whitespace
        assert dut.clean_text("  Hello, World!  ") == "Hello, World!"

    def test_clean_text_replaces_links_with_placeholders(self):
        # Test that clean_text replaces links with placeholders
        assert (
            dut.clean_text("[link](https://example.com)")
            == "[LINK1] link [LINK2]"
        )
        assert (
            dut.clean_text("Visit https://example.com for more info.")
            == "Visit [LINK1] example.com [LINK2] for more info."
        )

    def test_clean_text_replaces_emojis(self):
        # Test that clean_text replaces emojis with text representations
        assert dut.clean_text("😂") == ":face_with_tears_of_joy:"

    def test_clean_text_removes_accents(self):
        # Test that clean_text removes accents from characters
        assert dut.clean_text("Café") == "Cafe"
        assert (
            dut.clean_text("Some text with non-ascii characters: ñ, ü, å.")
            == "Some text with non-ascii characters: n, u, a."
        )

    def test_clean_text_combined(self):
        # Test that clean_text handles a combination of all cases
        assert (
            dut.clean_text(
                "  Hello, World! Visit https://example.com for more info. 😂 Café [link](https://example.com) Some text with non-ascii characters: ñ, ü, å.  "
            )
            == "Hello, World! Visit [LINK1] example.com [LINK2] for more info. :face_with_tears_of_joy: Cafe [LINK1] link [LINK2] Some text with non-ascii characters: n, u, a."
        )


class TestComputeRelativeDistance:
    def test_single_node_tree(self):
        tree = {"id": 1, "tree": []}
        dut.compute_relative_distance(tree)
        assert tree["distances"] == {1: [0, 0]}

    def test_two_level_tree(self):
        tree = {"id": 1, "tree": [{"id": 2, "tree": []}]}
        dut.compute_relative_distance(tree)
        assert tree["distances"] == {1: [0, 0], 2: [0, 1]}
        assert tree["tree"][0]["distances"] == {1: [1, 0], 2: [0, 0]}

    def test_three_level_tree(self):
        tree = {"id": 1, "tree": [{"id": 2, "tree": [{"id": 3, "tree": []}]}]}
        dut.compute_relative_distance(tree)
        assert tree["distances"] == {1: [0, 0], 2: [0, 1], 3: [0, 2]}
        assert tree["tree"][0]["distances"] == {1: [1, 0], 2: [0, 0], 3: [0, 1]}
        assert tree["tree"][0]["tree"][0]["distances"] == {
            1: [2, 0],
            2: [1, 0],
            3: [0, 0],
        }

    def test_multiple_branches(self):
        tree = {
            "id": 1,
            "tree": [
                {"id": 2, "tree": []},
                {"id": 3, "tree": [{"id": 4, "tree": []}]},
            ],
        }
        dut.compute_relative_distance(tree)
        assert tree["distances"] == {1: [0, 0], 2: [0, 1], 3: [0, 1], 4: [0, 2]}
        assert tree["tree"][0]["distances"] == {
            1: [1, 0],
            2: [0, 0],
            3: [1, 1],
            4: [1, 2],
        }
        assert tree["tree"][1]["distances"] == {
            1: [1, 0],
            2: [1, 1],
            3: [0, 0],
            4: [0, 1],
        }
        assert tree["tree"][1]["tree"][0]["distances"] == {
            1: [2, 0],
            2: [2, 1],
            3: [1, 0],
            4: [0, 0],
        }
