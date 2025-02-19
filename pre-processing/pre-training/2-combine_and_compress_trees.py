import json
from tqdm import tqdm
from glob import glob
from joblib import Parallel, delayed
import os
import gc
import shutil

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
    files_we_have = list(glob(f"{DATA_PATH_PROCESSED}/*/*.json"))
    files_we_have = []
    skip = {}

    def process_file(subreddit, num):
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

            counts = 0
            with open(sub, "r") as f:
                raw_file = f.read().strip()
                if len(raw_file.split("\n")) >= 2:
                    print(sub)
                data = json.loads(raw_file)
                # This filters out all discussions with a score less than 25
                if data["score"] < 25:
                    continue
                link_id = data["id"]

                graph = {
                    link_id: {
                        "data": data,
                        "tree": [],
                        "id": link_id,
                        "depth": 0,
                    }
                }
                counts = 1

            missing = []

            def add_to_graph(node, parent_id):
                nonlocal counts
                if parent_id in skip:
                    skip[node["id"]] = True
                    return True
                elif parent_id in graph:
                    parent_depth = graph[parent_id]["depth"]
                    if (
                        parent_depth + 1 > 4
                    ):  # NOTE: We also enforce a depth limit here
                        skip[node["id"]] = True
                    else:
                        graph[node["id"]] = {
                            "data": node,
                            "tree": [],
                            "id": node["id"],
                            "depth": parent_depth + 1,
                        }
                        graph[parent_id]["tree"] += [graph[node["id"]]]
                        counts += 1
                    return True
                return False

            if os.path.exists(comment):
                with open(comment, "r") as comment_f:
                    for line in comment_f:
                        if line == "\n":
                            continue
                        node = json.loads(line)
                        parent_id = node["parent_id"][3:]
                        node["parent_id"] = node["parent_id"][3:]

                        if not add_to_graph(node, parent_id):
                            missing += [(parent_id, node)]

            for parent_id, data in missing:
                add_to_graph(data, parent_id)

            graph = {link_id: graph[link_id]}

            gc.collect()
            # mem_count = 0
            # os.makedirs(
            #     os.path.dirname(f"{DATA_PATH_PROCESSED}/{file[9:-7]}.json"),
            #     exist_ok=True,
            # )
            with open(
                f"{DATA_PATH_PROCESSED}/{CATEGORY}/{subreddit}.json", "a+"
            ) as write:
                write.write(json.dumps(graph[link_id]) + "\n")
                # count_size = count_size_of_tree(data[key])
                # if counts != count_size:
                #     print(counts, count_size)
                # mem_count += 1
                # if mem_count > 10000:
                #     gc.collect()
                #     mem_count = 0

    # NOTE: Right now this is set to just read from the files in Test-LocalCity
    # but it should be all the files you want to read from. It can handle
    # multiple subreddits at once. (data/raw/*/*-RC.txt, for example)

    res = Parallel(n_jobs=-1)(
        delayed(process_file)(file, i)
        for i, file in tqdm(
            enumerate(os.listdir(f"{DATA_PATH_RAW}/{CATEGORY}"))
        )
    )
    # with open('complete-graphs.json', 'w') as file:
    #     for graph_file in glob('complete-graphs-*.json'):
    #         with open(graph_file, 'r') as read:
    #             for line in read:
    #                 file.write(line)
    # print('labels: ', sum(res))


def count_size_of_tree(x):
    return sum([count_size_of_tree(y) for y in x["tree"]]) + 1


if __name__ == "__main__":
    main()
