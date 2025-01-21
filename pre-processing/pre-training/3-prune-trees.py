import math
import json
from glob import glob
from tqdm import tqdm
import os
from joblib import Parallel, delayed


def main():
    # NOTE: This is currently hard coded to only access files in Test-LocalCity
    # but it can be generalized to access all files in data/processed
    Parallel(n_jobs=-1)(
        delayed(process)(file)
        for file in tqdm(list(glob("data/processed/Test-LocalCity/*.json")))
    )


def process(file):
    path = os.path.dirname(file)
    file_name = os.path.basename(file)
    os.makedirs("data/pruned/" + path.split("/")[2], exist_ok=True)
    size = 0
    count = 0
    total = 0
    with open(path + "/" + file_name, "r") as read, open(
        "data/pruned/" + path.split("/")[2] + "/" + file_name, "w"
    ) as write:
        for line in read:
            data = json.loads(line)
            size = count_size_of_tree(data)
            # TODO: This condition currently says to keep all posts with a size
            # greater then 10 and a score greater than 25.
            # Instead, we should keep the top 20,000 submissions, which also
            # abide by these conditions.
            if size > 10 and data["data"]["score"] > 25:
                count += 1
                trim_and_get_size(data)
                write.write(json.dumps(data) + "\n")
            total += 1

    print(f"{file_name} {count} {total}")


def count_size_of_tree(x):
    return sum([count_size_of_tree(y) for y in x["tree"]]) + 1


def trim_and_get_size(comment: dict, depth=0):
    sizes = []  # (size, index)
    infs = 0
    for i, child in enumerate(comment["tree"]):
        if depth + 1 < 4:
            res = trim_and_get_size(child, depth + 1)
            sizes += [(res, i)]
            if res == math.inf:
                infs += 1
        else:
            child["tree"] = []
            sizes += [(0, i)]

    # NOTE: This makes sure that the branching factor is 2. We keep only the
    # branches with the most nodes. This is a heuristic to only keep branches
    # with active conversations, but it is not perfect.
    # TODO: We should factor in upvotes and downvotes to determine the most
    # promising branches. (comment['data']['score'])
    trimed_size = max(2, infs)
    sizes = sorted(sizes, key=lambda x: x[0], reverse=True)[:trimed_size]
    new_size = sum([s[0] for s in sizes])
    comment["tree"] = [comment["tree"][x[1]] for x in sizes]
    return new_size


if __name__ == "__main__":
    main()
