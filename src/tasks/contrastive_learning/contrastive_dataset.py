import logging
from typing import Any

from data.types import ContrastiveLabels
from tasks.dataset import ContrastiveTaskDataset


class ContrastivePreTrainingDataset(ContrastiveTaskDataset):
    def __init__(self, **kwargs):
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
            "Affluence": {
                "a": [
                    "vagabond",
                    "hitchhiking",
                    "DumpsterDiving",
                    "almosthomeless",
                    "AskACountry",
                    "KitchenConfidential",
                    "Nightshift",
                    "alaska",
                    "fuckolly",
                    "FolkPunk",
                ],
                "b": [
                    "backpacking",
                    "hiking",
                    "Frugal",
                    "personalfinance",
                    "travel",
                    "Cooking",
                    "fitbit",
                    "CampingandHiking",
                    "gameofthrones",
                    "IndieFolk",
                ],
            },
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
                    "AskMen",
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
                    "AskMenOver30",
                    "eldertrees",
                    "running",
                    "trailrunning",
                    "MaleFashionMarket",
                    "canadacordcutters",
                    "pearljam",
                ],
            },
            "Edgy": {
                "a": [
                    "memes",
                    "watchpeoplesurvive",
                    "MissingPersons",
                    "twinpeaks",
                    "pickuplines",
                    "texts",
                    "startrekgifs",
                    "subredditoftheday",
                    "peeling",
                    "rapbattles",
                ],
                "b": [
                    "ImGoingToHellForThis",
                    "watchpeopledie",
                    "MorbidReality",
                    "TrueDetective",
                    "MeanJokes",
                    "FiftyFifty",
                    "DaystromInstitute",
                    "SRSsucks",
                    "bestofworldstar",
                ],
            },
        }

        groups["Partisan A"]["a"].extend(groups["Partisan B"]["a"])
        groups["Partisan A"]["b"].extend(groups["Partisan B"]["b"])
        del groups["Partisan B"]

        # Map to convert a subreddit name to a group
        self.idx_map = {}
        idx = 0
        for _, value in self.groups.items():
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
                        logging.error("Duplicate index found: %s", subreddit)
                    self.idx_map[subreddit] = (pos, neg)

        super().__init__(**kwargs)

    def has_graph_labels(self) -> bool:
        return True

    def retrieve_label(self, data: dict[str, Any]) -> dict[str, bool | int]:
        """Retrieves the label for the comment corresponding to
        HatefulDiscussions.

        Args:
            data (dict[str, Any]): The metadata of the comment.

        Returns:
            tuple[int, bool]: _description_
        """
        assert (
            "label" in data
        ), '"label" key not found in data, please check that the data format is compatible with HatefulDiscussions.'

        subreddit = data["subreddit"]
        if subreddit not in self.idx_map:
            logging.error("Subreddit not found: %s", subreddit)
            return {
                ContrastiveLabels.Ys: -1,
                ContrastiveLabels.YMask: False,
                ContrastiveLabels.HardYs: -1,
            }

        pos, neg = self.idx_map[subreddit]

        return {
            ContrastiveLabels.Ys: pos,
            ContrastiveLabels.YMask: True,
            ContrastiveLabels.HardYs: neg,
        }
