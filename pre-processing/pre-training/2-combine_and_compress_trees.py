"""Script to combine lists of comments and posts into a tree structure."""

import json
from tqdm import tqdm
from joblib import Parallel, delayed
import os
import gc
import shutil
from combine_utils import combine_nodes_to_tree

ROOT_PATH = "/mnt/DATA/reddit_share"
DATA_PATH_RAW = ROOT_PATH + "/data_test/raw"
DATA_PATH_PROCESSED = ROOT_PATH + "/data_test/processed"


def main():
    CATEGORY = "Test-LocalCity"
    # output should be year-month-combined.json
    # json with
    """
    {
        data: {} (including "label": <>)
        id: ""
        tree: [<>]
    }
    """

    def process_file(subreddit):
        # this includes the topic prefix
        # if f'{DATA_PATH_PROCESSED}/{file[9:-7]}.json' in files_we_have:
        #     return 0
        # print(subreddit)
        path = f"{DATA_PATH_RAW}/{CATEGORY}/{subreddit}"
        if not os.path.isdir(path):
            return  # Not a directory. Abort
        os.makedirs(f"{DATA_PATH_PROCESSED}/{CATEGORY}", exist_ok=True)
        for postid in os.listdir(path):
            sub = f"{path}/{postid}/POST.txt"
            comment = f"{path}/{postid}/RC.txt"

            if not (os.path.exists(sub)):
                invalid_dir = f"{path}/{postid}"
                if os.path.isdir(invalid_dir):
                    # Remove the entire folder
                    shutil.rmtree(invalid_dir, ignore_errors=True)
                continue

            with open(sub, "r") as f:
                raw_file = f.read().strip()
                if len(raw_file.split("\n")) >= 2:
                    print(sub)
                data = json.loads(raw_file)
                # This filters out all discussions with a score less than 25
                if data["score"] < 25:
                    continue
                link_id = data["id"]

                graph = [data]

            if os.path.exists(comment):
                with open(comment, "r") as comment_f:
                    for line in comment_f:
                        if line == "\n":
                            continue
                        node = json.loads(line)
                        graph.append(node)

            processed_graph = combine_nodes_to_tree(graph)
            if not processed_graph:
                print("Invalid processed graph")
            gc.collect()
            # mem_count = 0
            # os.makedirs(
            #     os.path.dirname(f"{DATA_PATH_PROCESSED}/{file[9:-7]}.json"),
            #     exist_ok=True,
            # )
            with open(
                f"{DATA_PATH_PROCESSED}/{CATEGORY}/{subreddit}.json", "a+"
            ) as write:
                write.write(json.dumps(processed_graph) + "\n")

    # NOTE: Right now this is set to just read from the files in Test-LocalCity
    # but it should be all the files you want to read from. It can handle
    # multiple subreddits at once. (data/raw/*/*-RC.txt, for example)

    dirs_to_process = os.listdir(f"{DATA_PATH_RAW}/{CATEGORY}")
    Parallel(n_jobs=-1)(
        delayed(process_file)(file)
        for file in tqdm(dirs_to_process, total=len(dirs_to_process))
    )


def count_size_of_tree(x: dict) -> int:
    """Returns the size of the tree."""
    return sum([count_size_of_tree(y) for y in x["tree"]]) + 1


if __name__ == "__main__":
    main()
