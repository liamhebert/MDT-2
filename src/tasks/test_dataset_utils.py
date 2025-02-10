"""Tests for the dataset_utils module."""

import tasks.dataset_utils as dut


class TestCleanTest:
    """Tests for the clean_text function."""

    def test_clean_text_removes_extra_whitespace(self):
        """Test that clean_text removes leading and trailing whitespace."""
        assert dut.clean_text("  Hello, World!  ") == "Hello, World!"

    def test_clean_text_replaces_links_with_placeholders(self):
        """Test that clean_text replaces links with placeholders."""
        assert (
            dut.clean_text("[link](https://example.com)")
            == "[LINK1] link [LINK2]"
        )
        assert (
            dut.clean_text("Visit https://example.com/hello for more info.")
            == "Visit [LINK1] example.com [LINK2] for more info."
        )

    def test_clean_text_replaces_emojis(self):
        """Test that clean_text replaces emojis with text representations."""
        assert dut.clean_text("😂") == ":face_with_tears_of_joy:"

    def test_clean_text_removes_accents(self):
        """Test that clean_text removes accents from characters."""
        assert dut.clean_text("Café") == "Cafe"
        assert (
            dut.clean_text("Some text with non-ascii characters: ñ, ü, å.")
            == "Some text with non-ascii characters: n, u, a."
        )

    def test_clean_text_combined(self):
        """Test that clean_text handles a combination of all cases."""
        assert dut.clean_text(
            (
                "  Hello, World! Visit https://example.com/hello for more info. "
                "😂 Café [link](https://example.com) Some text with non-ascii "
                "characters: ñ, ü, å.  "
            )
        ) == (
            "Hello, World! Visit [LINK1] example.com [LINK2] for more info. "
            ":face_with_tears_of_joy: Cafe [LINK1] link [LINK2] Some text with "
            "non-ascii characters: n, u, a."
        )


class TestComputeRelativeDistance:
    """Tests for the compute_relative_distance function."""

    def test_single_node_tree(self):
        """Test that compute_relative_distance works for a single node tree.

        Note that under this use case, there would be no relative distances and
        just the single node in the tree.
        """
        tree = {"id": 1, "tree": []}
        dut.compute_relative_distance(tree)
        assert tree["distances"] == {1: [0, 0]}
        assert tree["rotary_position"] == [0, 0]

    def test_two_level_tree(self):
        """Test that compute_relative_distance works for a two-level tree."""
        tree = {"id": 1, "tree": [{"id": 2, "tree": []}]}
        dut.compute_relative_distance(tree)
        assert tree["distances"] == {1: [0, 0], 2: [0, 1]}
        assert tree["tree"][0]["distances"] == {1: [1, 0], 2: [0, 0]}
        assert tree["rotary_position"] == [0, 0]
        assert tree["tree"][0]["rotary_position"] == [0, 1]

    def test_three_level_tree(self):
        """Test that compute_relative_distance works for a three-level tree."""
        tree = {"id": 1, "tree": [{"id": 2, "tree": [{"id": 3, "tree": []}]}]}
        dut.compute_relative_distance(tree)
        assert tree["distances"] == {1: [0, 0], 2: [0, 1], 3: [0, 2]}
        assert tree["tree"][0]["distances"] == {1: [1, 0], 2: [0, 0], 3: [0, 1]}
        assert tree["tree"][0]["tree"][0]["distances"] == {
            1: [2, 0],
            2: [1, 0],
            3: [0, 0],
        }
        assert tree["rotary_position"] == [0, 0]
        assert tree["tree"][0]["rotary_position"] == [0, 1]
        assert tree["tree"][0]["tree"][0]["rotary_position"] == [0, 2]

    def test_multiple_branches(self):
        """Test that compute_relative_distance works for a branching tree."""
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
        assert tree["rotary_position"] == [0, 0]
        assert tree["tree"][0]["rotary_position"] == [0, 1]
        assert tree["tree"][1]["rotary_position"] == [1, 1]
        assert tree["tree"][1]["tree"][0]["rotary_position"] == [1, 2]

    def test_complex_rotary(self):
        tree = {
            "id": 1,
            "tree": [
                {
                    "id": 2,
                    "tree": [{"id": 6, "tree": []}, {"id": 7, "tree": []}],
                },
                {
                    "id": 3,
                    "tree": [{"id": 4, "tree": []}, {"id": 5, "tree": []}],
                },
            ],
        }
        dut.compute_relative_distance(tree)

        def get_distance(tree, distances={}):
            distances[tree["id"]] = tree["rotary_position"]
            for child in tree["tree"]:
                get_distance(child, distances)
            return distances

        distances = get_distance(tree)
        # Split, Depth
        assert distances == {
            1: [0, 0],
            2: [0, 1],
            6: [0, 2],
            7: [1, 2],
            3: [2, 1],
            4: [2, 2],
            5: [3, 2],
        }
