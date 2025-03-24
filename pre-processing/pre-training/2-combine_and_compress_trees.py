"""Script to combine lists of comments and posts into a tree structure."""

import orjson
from glob import glob
from tqdm import tqdm
from joblib import Parallel, delayed
import os
from combine_utils import combine_nodes_to_tree
import math

ROOT_PATH = "./"
DATA_PATH_RAW = ROOT_PATH + "/raw"
DATA_PATH_PROCESSED = ROOT_PATH + "/processed"


def count_size_of_tree(x):
    """
    Recursively count the size of the tree x
    """
    return sum([count_size_of_tree(y) for y in x["tree"]]) + 1


def trim_and_get_size(comment: dict, depth=0):
    """
    Trim the tree so that branching factor is limited to 2.
    For each node, the "top" two child are selected (and others are ignored)
    We prefer children with greater score. If there's a tie, we prefer
    children with larger tree size.
    """
    scores = []  # (score, size, index)
    infs = 0
    for i, child in enumerate(comment["tree"]):
        if depth + 1 < 4:
            res = trim_and_get_size(child, depth + 1)
            scores += [(comment["score"], res, i)]
            if res == math.inf:
                infs += 1
        else:
            child["tree"] = []
            scores += [(comment["score"], 0, i)]

    # NOTE: This makes sure that the branching factor is 2. We keep only the
    # branches with the best score, then the most nodes. This is a heuristic to
    # only keep branches with active conversations, but it is not perfect.
    trimed_size = max(2, infs)
    scores = sorted(scores, key=lambda x: (x[0], x[1]), reverse=True)[
        :trimed_size
    ]
    new_size = sum([s[1] for s in scores])
    comment["tree"] = [comment["tree"][x[2]] for x in scores]
    return new_size


def main():
    # output should be year-month-combined.json
    # json with
    """
    {
        data: {}
        id: ""
        tree: [<>]
    }
    """
    CATEGORY = "Test-LocalCity"

    def process_file(path):
        print(f"Processing {path}")

        if not os.path.isdir(path):
            print(f"{path} is not a directory")
            return  # Not a directory. Abort
        subreddit = os.path.basename(path)
        category = os.path.basename(os.path.dirname(path))
        os.makedirs(f"{DATA_PATH_PROCESSED}/{category}", exist_ok=True)

        sub = f"{path}/POST.txt"
        comment = f"{path}/RC.txt"
        if not os.path.exists(sub) or not os.path.exists(comment):
            print(f"{sub} or {comment} does not exist, skipping")
            return

        with open(sub, "r") as f:
            graphs = {}
            for line in f:
                if line == "\n":
                    continue
                data = orjson.loads(line)
                graphs[data["id"]] = [data]

        with open(comment, "r") as comment_f:
            for line in comment_f:
                if line == "\n":
                    continue
                node = orjson.loads(line)
                graphs[node["link_id"]].append(node)

        with open(
            f"{DATA_PATH_PROCESSED}/{category}/{subreddit}.json", "wb"
        ) as write:
            for post_id, graph in graphs.items():
                if len(graph) < 10:
                    continue

                processed_graph = combine_nodes_to_tree(graph)
                trim_and_get_size(processed_graph)

                if not processed_graph:
                    print("Invalid processed graph")

                write.write(
                    orjson.dumps(
                        processed_graph, option=orjson.OPT_APPEND_NEWLINE
                    )
                )

    # NOTE: Right now this is set to just read from the files in Test-LocalCity
    # but it should be all the files you want to read from. It can handle
    # multiple subreddits at once. (data/raw/*/*-RC.txt, for example)

    dirs_to_process = glob(f"{DATA_PATH_RAW}/*/*")
    Parallel(n_jobs=1)(
        delayed(process_file)(file)
        for file in tqdm(dirs_to_process, total=len(dirs_to_process))
    )


if __name__ == "__main__":
    main()
