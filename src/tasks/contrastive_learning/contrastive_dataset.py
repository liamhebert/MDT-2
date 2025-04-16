"""Contrastive pre-training dataset for the contrastive learning task."""

from utils.pylogger import RankedLogger
from typing import Any

from data.types import ContrastiveLabels
from data.collated_datasets import ContrastiveTaskDataset

logging = RankedLogger(__name__)


class ContrastivePreTrainingDataset(ContrastiveTaskDataset):
    """
    Task dataset for Contrastive DisCo Pre-Training.
    """

    def __init__(self, politics_and_gender_only=False, **kwargs):
        groups = {
            "Partisan A": {
                "a": [
                    "democrats",
                    "OpenChristian",
                    "GamerGhazi",
                    "excatholic",
                    "EnoughLibertarianSpam",
                    "AskAnAmerican",
                    "lastweektonight",
                ],
                "b": [
                    "Conservative",
                    "progun",
                    "TrueChristian",
                    "Catholicism",
                    "askaconservative",
                    "AskTrumpSupporters",
                    "CGPGrey",
                ],
            },
            "Partisan B": {
                "a": [
                    "hillaryclinton",
                    "SandersForPresident",
                    "askhillarysupporters",
                    "BlueMidterm2018",
                    "badwomensanatomy",
                    "PoliticalVideo",
                    "liberalgunowners",
                    "GrassrootsSelect",
                    "GunsAreCool",
                ],
                "b": [
                    "The_Donald",
                    "KotakuInAction",
                    "HillaryForPrison",
                    "AskThe_Donald",
                    "PoliticalHumor",
                    "ChoosingBeggars",
                    "uncensorednews",
                    "Firearms",
                    "DNCleaks",
                    "dgu",
                ],
            },
            # "Affluence": {
            #     "a": [
            #         "vagabond",
            #         "hitchhiking",
            #         "DumpsterDiving",
            #         "almosthomeless",
            #         "AskACountry",
            #         "KitchenConfidential",
            #         "Nightshift",
            #         "alaska",
            #         "fuckolly",
            #         "FolkPunk",
            #     ],
            #     "b": [
            #         "backpacking",
            #         "hiking",
            #         "Frugal",
            #         "personalfinance",
            #         "travel",
            #         "Cooking",
            #         "fitbit",
            #         "CampingandHiking",
            #         "gameofthrones",
            #         "IndieFolk",
            #     ],
            # },
            "Gender": {
                "a": [
                    "AskMen",
                    "TrollYChromosome",
                    "AskMenOver30",
                    "OneY",
                    "TallMeetTall",
                    "daddit",
                    "ROTC",
                    "FierceFlow",
                    "malelivingspace",
                    "predaddit",
                ],
                "b": [
                    "AskWomen",
                    "CraftyTrolls",
                    "AskWomenOver30",
                    "women",
                    "bigboobproblems",
                    "Mommit",
                    "USMilitarySO",
                    "HaircareScience",
                    "InteriorDesign",
                    "BabyBumps",
                ],
            },
            "Age": {
                "a": [
                    "teenagers",
                    "youngatheists",
                    "teenrelationships",
                    "saplings",
                    "hsxc",
                    "trackandfield",
                    "bapccanada",
                    "RedHotChiliPeppers",
                ],
                "b": [
                    "RedditForGrownups",
                    "TrueAtheism",
                    "relationship_advice",
                    "eldertrees",
                    "running",
                    "trailrunning",
                    "canadacordcutters",
                    "pearljam",
                ],
            },
            # "Edgy": {
            #     "a": [
            #         "memes",
            #         "watchpeoplesurvive",
            #         "MissingPersons",
            #         "twinpeaks",
            #         "pickuplines",
            #         "texts",
            #         "startrekgifs",
            #         "subredditoftheday",
            #         "peeling",
            #         "rapbattles",
            #     ],
            #     "b": [
            #         "ImGoingToHellForThis",
            #         "watchpeopledie",
            #         "MorbidReality",
            #         "TrueDetective",
            #         "MeanJokes",
            #         "FiftyFifty",
            #         "DaystromInstitute",
            #         "SRSsucks",
            #         "bestofworldstar",
            #     ],
            # },
        }

        groups["Partisan A"]["a"].extend(groups["Partisan B"]["a"])
        groups["Partisan A"]["b"].extend(groups["Partisan B"]["b"])
        del groups["Partisan B"]

        if politics_and_gender_only:
            self.tag = "politics_and_gender_pretrain"
            groups = {
                "Partisan A": groups["Partisan A"],
                "Gender": groups["Gender"],
            }
            logging.info("Using only politics and gender subreddits.")
        else:
            self.tag = "all_pretrain"
            logging.info("Using all subreddits.")

        # Map to convert a subreddit name to a group
        self.idx_map = {}
        idx = 0
        duplicates_found = []
        for _, value in groups.items():
            a = idx
            b = idx + 1
            idx += 2
            for side, subreddits in value.items():
                if side == "a":
                    pos, neg = a, b
                else:
                    pos, neg = b, a
                for subreddit in subreddits:
                    if subreddit in self.idx_map:
                        logging.warning("Duplicate index found: %s", subreddit)
                        duplicates_found.append(subreddit)
                    else:
                        self.idx_map[subreddit] = (pos, neg)

        if duplicates_found:
            logging.warning(
                f"Total duplicates found: {len(duplicates_found)}. This may"
                " affect model results."
            )

        super().__init__(**kwargs)

    def retrieve_label(self, data: dict[str, Any]) -> dict[str, bool | int]:
        """Retrieves the graph label from the root node.

        Since has_graph_labels is True, this function is only called on the root
        node of the graph. This function is called with the "data" field of the
        node, which contains whatever auxiliary information is available for the
        node.

        Args:
            data (dict[str, Any]): The metadata of the comment.

        Returns:
            dict[str, bool | int]: The label information for the node, which must
                contain "Ys" and "YMask" as keys.
        """
        assert "subreddit" in data, (
            '"subreddit" key not found in data, please check that the data ',
            "format is compatible with ContrastiveLearning.",
            f"{data.keys()=}",
        )

        subreddit = data["subreddit"]
        if subreddit not in self.idx_map:
            return {
                ContrastiveLabels.Ys: -100,
                ContrastiveLabels.HardYs: -100,
            }

        pos, neg = self.idx_map[subreddit]

        return {
            ContrastiveLabels.Ys: pos,
            ContrastiveLabels.HardYs: neg,
        }
