"""
Processes .zst archives of Reddit comments and sorts them into
topic-specific files.
"""

import json
from tqdm import tqdm
from glob import glob
import os
import re

# The pattern for the months of archives to look for.
ARCHIVE_MONTHS = "2017-1*"
DATA_PATH = "/mnt/DATA/reddit_share/data_test/raw"


def main():
    """Entry point to process the files."""
    # groups = {
    #     "Partisan A": {
    #         "a": [
    #             "democrats",
    #             "GunsAreCool",
    #             "OpenChristian",
    #             "GamerGhazi",
    #             "excatholic",
    #             "EnoughLibertarianSpam",
    #             "AskAnAmerican",
    #             "askhillarysupporters",
    #             "liberalgunowners",
    #             "lastweektonight",
    #         ],
    #         "b": [
    #             "Conservative",
    #             "progun",
    #             "TrueChristian",
    #             "KotakuInAction",
    #             "Catholicism",
    #             "ShitRConversativeSays",
    #             "askaconservative",
    #             "AskTrumpSupporters",
    #             "Firearms",
    #             "CGPGrey",
    #         ],
    #     },
    #     "Partisan B": {
    #         "a": [
    #             "hillaryclinton",
    #             "GamerGhazi",
    #             "SandersForPresident",
    #             "askhillarysupporters",
    #             "BlueMidterm2018",
    #             "badwomensanatomy",
    #             "PoliticalVideo",
    #             "liberalgunowners",
    #             "GrassrootsSelect",
    #             "GunsAreCool",
    #         ],
    #         "b": [
    #             "The_Donald",
    #             "KotakuInAction",
    #             "HillaryForPrison",
    #             "AskThe_Donald",
    #             "PoliticalHumor",
    #             "ChoosingBeggars",
    #             "uncensorednews",
    #             "Firearms",
    #             "DNCleaks",
    #             "dgu",
    #         ],
    #     },
    #     "Affluence": {
    #         "a": [
    #             "vagabond",
    #             "hitchhiking",
    #             "DumpsterDiving",
    #             "almosthomeless",
    #             "AskACountry",
    #             "KitchenConfidential",
    #             "Nightshift",
    #             "alaska",
    #             "fuckolly",
    #             "FolkPunk",
    #         ],
    #         "b": [
    #             "backpacking",
    #             "hiking",
    #             "Frugal",
    #             "personalfinance",
    #             "travel",
    #             "Cooking",
    #             "fitbit",
    #             "CampingandHiking",
    #             "gameofthrones",
    #             "IndieFolk",
    #         ],
    #     },
    #     "Gender": {
    #         "a": [
    #             "AskMen",
    #             "TrollYChromosome",
    #             "AskMenOver30",
    #             "OneY",
    #             "TallMeetTall",
    #             "daddit",
    #             "ROTC",
    #             "FierceFlow",
    #             "malelivingspace",
    #             "predaddit",
    #         ],
    #         "b": [
    #             "AskWomen",
    #             "CraftyTrolls",
    #             "AskWomenOver30",
    #             "women",
    #             "bigboobproblems",
    #             "Mommit",
    #             "USMilitarySO",
    #             "HaircareScience",
    #             "InteriorDesign",
    #             "BabyBumps",
    #         ],
    #     },
    #     "Age": {
    #         "a": [
    #             "teenagers",
    #             "youngatheists",
    #             "teenrelationships",
    #             "AskMen",
    #             "saplings",
    #             "hsxc",
    #             "trackandfield",
    #             "TeenMFA",
    #             "bapccanada",
    #             "RedHotChiliPeppers",
    #         ],
    #         "b": [
    #             "RedditForGrownups",
    #             "TrueAtheism",
    #             "relationship_advice",
    #             "AskMenOver30",
    #             "eldertrees",
    #             "running",
    #             "trailrunning",
    #             "MaleFashionMarket",
    #             "canadacordcutters",
    #             "pearljam",
    #         ],
    #     },
    #     "Edgy": {
    #         "a": [
    #             "memes",
    #             "watchpeoplesurvive",
    #             "MissingPersons",
    #             "twinpeaks",
    #             "pickuplines",
    #             "texts",
    #             "startrekgifs",
    #             "subredditoftheday",
    #             "peeling",
    #             "rapbattles",
    #         ],
    #         "b": [
    #             "ImGoingToHellForThis",
    #             "watchpeopledie",
    #             "MorbidReality",
    #             "TrueDetective",
    #             "MeanJokes",
    #             "FiftyFifty",
    #             "DaystromInstitute",
    #             "SRSsucks",
    #             "Gore",
    #             "bestofworldstar",
    #         ],
    #     },
    # }

    groups = {
        "Test-LocalCity": {
            "a": [
                "sanfrancisco",
                "bayarea",
                "toronto",
                "vancouver",
                "calgary",
                "waterloo",
                "texas",
                "ohio",
                "newyorkcity",
                "losangeles",
            ],
        }
    }

    # a dictionary of the name of each subreddit in groups and the topic it
    # belongs to. If there are multiple topics, note each one
    subreddit_to_topic = {}
    for topic, subreddits in groups.items():
        for group, subs in subreddits.items():
            for sub in subs:
                if sub not in subreddit_to_topic:
                    subreddit_to_topic[sub] = []
                subreddit_to_topic[sub] += [topic]

    path = "/mnt/DATA/reddit_share/"
    counts = {key: 0 for key in subreddit_to_topic.keys()}
    subreddit_regex = re.compile('"subreddit":"([^"]+)"')
    subreddit_null_regex = re.compile('"subreddit_id":null')
    postid_regex = re.compile('"id":"([^"]+)"')
    linkid_regex = re.compile('"link_id":"([^"]+)"')
    crosspost_regex = re.compile('"crosspost_parent":".._([^"]+)"')
    for file in tqdm(
        list(
            glob(path + f"RC_{ARCHIVE_MONTHS}.zst")
            + list(glob(path + f"RS_{ARCHIVE_MONTHS}.zst"))
        ),
        desc="Files",
        position=0,
    ):
        # print(file)
        os.system("zstd -d -T24 --memory=2048MiB " + file)
        with open(file[:-4], "r") as f:
            # find the total number of lines in the file
            num_lines = 0
            for line in tqdm(f, desc="Counting Lines", position=1, leave=True):
                num_lines += 1
            f.seek(0)
            for line in tqdm(
                f, total=num_lines, desc="Lines", position=2, leave=True
            ):
                # find the first subreddit mentioned
                subreddit = subreddit_regex.search(line)
                postids = postid_regex.findall(line)
                crosspost_ids = crosspost_regex.findall(line)
                # Filter out crosspost ids from a list of possible ids
                for crosspost_id in crosspost_ids:
                    if crosspost_id in postids:
                        postids.remove(crosspost_id)
                # Assume post IDs have length less than 8. If not, it's not
                # postid and we can remove it
                for i in reversed(range(len(postids))):
                    if len(postids[i]) > 8:
                        postids.pop(i)

                if subreddit is None:
                    # If subreddit_id is null, assume it's an ad
                    if subreddit_null_regex.search(line) is None:
                        print("Subreddit doesn't exist")
                        print(line)
                    continue
                subreddit = subreddit.group(1)
                if len(postids) != 1:
                    print(f"Incorrect IDs: {postids}")
                    print(line)
                    continue
                postid = postids[0]
                if "RC" in file:
                    # Comments have link IDs as a field. Use that.
                    linkid = linkid_regex.search(line)
                    linkid = linkid.group(1)[3:]
                else:
                    # Original posts don't have link IDs, so we just use
                    # the post ID as the link ID
                    linkid = postid
                if subreddit in subreddit_to_topic:
                    counts[subreddit] += 1
                    # write line to topic.txt file
                    for topic in subreddit_to_topic[subreddit]:
                        os.makedirs(
                            f"{DATA_PATH}/{topic}/{subreddit}/{linkid}",
                            exist_ok=True,
                        )
                        if "RC" in file:
                            with open(
                                f"{DATA_PATH}/{topic}/{subreddit}"
                                + f"/{linkid}/RC.txt",
                                "a+",
                            ) as topic_file:
                                topic_file.write(line)
                        else:
                            with open(
                                f"{DATA_PATH}/{topic}/{subreddit}"
                                + f"/{linkid}/POST.txt",
                                "a+",
                            ) as topic_file:
                                topic_file.write(line)
        os.remove(file[:-4])

    json.dump(counts, open("counts.json", "w"))


if __name__ == "__main__":
    main()
