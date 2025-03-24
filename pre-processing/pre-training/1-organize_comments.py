"""
Processes .zst archives of Reddit comments and sorts them into
topic-specific files.
"""

import orjson 
from tqdm import tqdm
from glob import glob
import os
import re
from heapq import heappush, heappop
import threading


# The pattern for the months of archives to look for.
ARCHIVE_MONTHS = "2018-*"
DATA_PATH = "./raw"
KEEP_COUNT = 20000

def extract_data(data: str, is_root: bool) -> dict:
    """Extracts the data from a line of the Reddit comment file."""
    data: dict = orjson.loads(data)
    # print(data)
    assert "data" not in data, data
    
    if not is_root:
        # print(data)
        # meta: dict = data["data"]
        return {
            "subreddit": data["subreddit"],
            "id": data["id"],
            "parent_id": data["parent_id"],
            "link_id": data["link_id"],
            "score": data["score"],
            "body": data["body"],
        }
    else:
        preview = data.get("preview", None)
        if preview is not None:
            preview = preview["images"][0]["source"]
        return {
            "subreddit": data["subreddit"],
            "id": data["id"],
            "score": data["score"],
            "body": data.get("selftext", "<empty body>"),
            "title": data["title"],
            "preview": preview,
        }

def thread_decompress(index: int, file: str):
    return os.system(f"zstd -d --long=31 -o ./tmp-{index} " + file)

def main():
    """Entry point to process the files."""
    groups = {
        "Partisan A": {
            "a": [
                "democrats",
                "GunsAreCool",
                "OpenChristian",
                "GamerGhazi",
                "excatholic",
                "EnoughLibertarianSpam",
                "AskAnAmerican",
                "askhillarysupporters",
                "liberalgunowners",
                "lastweektonight",
            ],
            "b": [
                "Conservative",
                "progun",
                "TrueChristian",
                "KotakuInAction",
                "Catholicism",
                "ShitRConversativeSays",
                "askaconservative",
                "AskTrumpSupporters",
                "Firearms",
                "CGPGrey",
            ],
        },
        "Partisan B": {
            "a": [
                "hillaryclinton",
                "GamerGhazi",
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
                "TeenMFA",
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
                "Gore",
                "bestofworldstar",
            ],
        },
    }

    # groups = {
    #     "Test-LocalCity": {
    #         "a": [
    #             # "sanfrancisco",
    #             # "bayarea",
    #             "toronto",
    #             # "vancouver",
    #             # "calgary",
    #             # "waterloo",
    #             # "texas",
    #             # "ohio",
    #             # "newyorkcity",
    #             # "losangeles",
    #         ],
    #     }
    # }

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
    # path = "./"
    linkid_regex = re.compile('"link_id":"([^"]+)"')
    subreddit_regex = re.compile('"subreddit":"([^"]+)"')
    
    data_heaps = {sub: [] for sub in subreddit_to_topic.keys()}
    current_thread = None
    
    files = list(enumerate(glob(path + f"RS_{ARCHIVE_MONTHS}.zst")))
    current_thread = threading.Thread(target=thread_decompress, args=files[0])
    current_thread.start()

    def process_submissions(file: str, index: int):
        date = os.path.basename(file).split("_")[-1].split(".")[0]
        year = int(date[:4])
        month = int(date[5:]) + (year * 12)

        with open(f"tmp-{index}", "r") as f:
            for i, line in tqdm(enumerate(f), position=1):
                subreddit = subreddit_regex.search(line).groups()
                is_valid = any(sub in subreddit_to_topic for sub in subreddit)
                if not is_valid:
                    continue

                data = extract_data(line, is_root=True)

                subreddit = data["subreddit"]
                if subreddit not in subreddit_to_topic:
                    continue

                score = data["score"]
                if score < 25:
                    continue

                heappush(data_heaps[subreddit], (score, month, i, data))
                if len(data_heaps[subreddit]) > KEEP_COUNT:
                    heappop(data_heaps[subreddit])
        os.remove(f"./tmp-{index}")

    for (i, file), (j, next_file) in tqdm(
        zip(files, files[1:]),
        desc="Reading submissions",
        position=0,
    ):
        current_thread.join()

        current_thread = threading.Thread(target=thread_decompress, args=(j, next_file))
        current_thread.start()
        
        process_submissions(file, i)
    
    current_thread.join()
    # process the last file
    process_submissions(files[-1][0], files[-1][1])
    

    # get link_ids from heap for each subreddit that survived
    ids_to_keep = {sub: set() for sub in subreddit_to_topic.keys()}
    for sub, heap in data_heaps.items():
        print("Subreddit: ", sub, " Size: ", len(heap))
        topic = subreddit_to_topic[sub][0]
        # Start from a clean slate each time
        if os.path.isdir(f"{DATA_PATH}/{topic}/{sub}"):
            os.system(f"rm -r {DATA_PATH}/{topic}/{sub}")
        
        os.makedirs(f"{DATA_PATH}/{topic}/{sub}", exist_ok=True)

        with open(f"{DATA_PATH}/{topic}/{sub}/POST.txt", "wb") as f:
            for _, _, _, data in heap:
                ids_to_keep[sub].add(data["id"])
                f.write(orjson.dumps(data, option=orjson.OPT_APPEND_NEWLINE))
    
    del data_heaps
    # now we can read the comments

    files = list(enumerate(glob(path + f"RC_{ARCHIVE_MONTHS}.zst")))
    current_thread = threading.Thread(target=thread_decompress, args=files[0])
    current_thread.start()

    def process_comments(index: int):
        with open(f"./tmp-{index}", "r") as f:
            for line in tqdm(f, position=1):
                subreddit = subreddit_regex.search(line).group(1)
                if not subreddit in subreddit_to_topic:
                    continue

                link_id = linkid_regex.search(line).group(1)[3:]
                
                if link_id in ids_to_keep[subreddit]:
                    topic = subreddit_to_topic[subreddit][0]
                    data = extract_data(line, is_root=False)
                    with open(f"{DATA_PATH}/{topic}/{subreddit}/RC.txt", "ab") as f:
                        f.write(orjson.dumps(data, option=orjson.OPT_APPEND_NEWLINE))
        os.remove(f"./tmp-{index}")
    

    for (i, file), (j, next_file) in tqdm(
        zip(files, files[1:]),
        desc="Reading comments",
        position=0,
    ):
        current_thread.join()

        current_thread = threading.Thread(target=thread_decompress, args=(j, next_file))
        current_thread.start()
        
        process_comments(i)
    
    current_thread.join()
    # process the last file
    process_comments(files[-1][0])



if __name__ == "__main__":
    main()
