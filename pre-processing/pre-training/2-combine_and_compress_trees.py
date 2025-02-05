import json
from tqdm import tqdm
from glob import glob
from joblib import Parallel, delayed
import os
import gc


def main():
    # output should be year-month-combined.json
    # json with
    """
    {
        data: {} (including "label": <>)
        id: ""
        tree: [<>]
    }
    """
    files_we_have = list(glob("data/processed/*/*.json"))
    files_we_have = []
    skip = {}

    def process_file(file, num):
        # this includes the topic prefix
        # if f'data/processed/{file[9:-7]}.json' in files_we_have:
        #     return 0

        path = file[:-7]
        sub = path + ".txt"
        comment = path + "-RC.txt"

        if not (os.path.exists(sub) and os.path.exists(comment)):
            print("missing: ", sub, comment)
            return 0

        # TODO: Right now, we save all graphs in memory, and only at the end do
        # we write them to disk. The reason why we do that is that we, stupidly,
        # do not make separate files for each post but rather group them all into
        # one file. This is a problem because its hard to know when we have
        # exhausted all the files for a given post. We should fix that.
        graph = {}
        counts = {}
        link_ids = []
        for line in open(sub, "r"):
            if line == "\n":
                continue
            data = json.loads(line)
            # This filters out all discussions with a score less than 25
            if data["score"] < 25:
                continue
            link_id = data["id"]
            link_ids += [link_id]

            graph[link_id] = {
                link_id: {"data": data, "tree": [], "id": link_id, "depth": 0}
            }
            counts[link_id] = 1

        missing = []

        def add_to_graph(node, parent_id, link_id):
            if link_id not in graph:
                return True
            if parent_id in skip:
                skip[node["id"]] = True
                return True
            elif parent_id in graph[link_id]:
                parent_depth = graph[link_id][parent_id]["depth"]
                if (
                    parent_depth + 1 > 4
                ):  # NOTE: We also enforce a depth limit here
                    skip[node["id"]] = True
                else:
                    graph[link_id][node["id"]] = {
                        "data": node,
                        "tree": [],
                        "id": node["id"],
                        "depth": parent_depth + 1,
                    }
                    graph[link_id][parent_id]["tree"] += [
                        graph[link_id][node["id"]]
                    ]
                    counts[link_id] += 1
                return True
            return False

        for line in open(comment, "r"):
            if line == "\n":
                continue
            node = json.loads(line)
            parent_id = node["parent_id"][3:]
            node["parent_id"] = node["parent_id"][3:]
            link_id = node["link_id"][3:]
            if link_id not in graph:
                continue

            if not add_to_graph(node, parent_id, link_id):
                missing += [(link_id, parent_id, node)]

        for link_id, parent_id, data in missing:
            add_to_graph(data, parent_id, link_id)

        del missing
        for key in graph.keys():
            graph[key] = {key: graph[key][key]}

        gc.collect()
        mem_count = 0
        os.makedirs(
            os.path.dirname(f"data/processed/{file[9:-7]}.json"), exist_ok=True
        )
        with open(f"data/processed/{file[9:-7]}.json", "w") as write:
            for key, data in graph.items():
                write.write(json.dumps(data[key]) + "\n")
                # count_size = count_size_of_tree(data[key])
                # if counts[key] != count_size:
                #     print(counts[key], count_size)
                del data[key]
                mem_count += 1
                if mem_count > 10000:
                    gc.collect()
                    mem_count = 0

    # NOTE: Right now this is set to just read from the files in Test-LocalCity
    # but it should be all the files you want to read from. It can handle
    # multiple subreddits at once. (data/raw/*/*-RC.txt, for example)

    res = Parallel(n_jobs=-1)(
        delayed(process_file)(file, i)
        for i, file in tqdm(
            enumerate(list(glob("data/raw/Test-LocalCity/*-RC.txt")))
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
