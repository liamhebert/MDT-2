import math
import json
from glob import glob
from tqdm import tqdm
import os
from joblib import Parallel, delayed
from heapq import heappush, heappop

ROOT_PATH = "/mnt/DATA/reddit_share"
DATA_PROCESSED = ROOT_PATH + "/data_test/processed"
DATA_PRUNED = ROOT_PATH + "/data_test/pruned"

KEEP_COUNT = 20000


def main():
    # NOTE: This is currently hard coded to only access files in Test-LocalCity
    # but it can be generalized to access all files in data/processed
    """
    Get all trees inside DATA_PROCESSED and prune them,
    output them in DATA_PRUNED
    """
    Parallel(n_jobs=-1)(
        delayed(process)(file)
        for file in tqdm(list(glob(DATA_PROCESSED + "/Test-LocalCity/*.json")))
    )


def process(file):
    """
    Read all trees from file and output the top KEEP_COUNT trees by score.
    All trees have to have a min size of 10 and min score of 25.
    Trees are pruned before added.
    """
    path = os.path.dirname(file)
    file_name = os.path.basename(file)
    os.makedirs(DATA_PRUNED + "/" + path.split("/")[2], exist_ok=True)
    size = 0
    count = 0
    total = 0
    with open(path + "/" + file_name, "r") as read, open(
        DATA_PRUNED + "/" + path.split("/")[2] + "/" + file_name, "w"
    ) as write:
        # A heap that stores the top KEEP_COUNT comments by score
        # This is a min heap, with the top element being the smallest score
        # When the size of the heap exceeds KEEP_COUNT, we pop the top element
        data_heap = []
        for line in read:
            data = json.loads(line)
            size = count_size_of_tree(data)
            # We only keep posts with a size
            # greater then 10 and a score greater than 25.
            # After that, the top KEEP_COUNT comments by score is selected
            if size > 10 and data["data"]["score"] > 25:
                count += 1
                trim_and_get_size(data)
                heappush(data_heap, (data["data"]["score"], json.dumps(data)))
                if len(data_heap) > KEEP_COUNT:
                    heappop(data_heap)
            total += 1
        for score, data_str in data_heap:
            write.write(data_str + "\n")

    print(f"{file_name} {count} {total}")


def count_size_of_tree(x):
    """
    Recursively count the size of the tree x
    """
    return sum([count_size_of_tree(y) for y in x["tree"]]) + 1


def trim_and_get_size(comment: dict, depth=0):
    """
    Trim the tree so that branching factor is limited to 2
    """
    scores = []  # (score, size, index)
    infs = 0
    for i, child in enumerate(comment["tree"]):
        if depth + 1 < 4:
            res = trim_and_get_size(child, depth + 1)
            scores += [(comment["data"]["score"], res, i)]
            if res == math.inf:
                infs += 1
        else:
            child["tree"] = []
            scores += [(comment["data"]["score"], 0, i)]

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


if __name__ == "__main__":
    main()
