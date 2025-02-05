"""Script to combine lists of comments and posts into a tree structure."""

import json
from tqdm import tqdm
from glob import glob
from joblib import Parallel, delayed
import os
import gc
from combine_utils import combine_nodes_to_tree


def main():
    """
    {
        data: {} (including "label": <>)
        id: ""
        tree: [<>]
    }
    """

    def process_file(file):

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
        graph: dict[str, list[dict]] = {}
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

            graph[link_id] = [
                {
                    "data": data,
                    "tree": [],
                    "id": link_id,
                    "parent_id": None,
                }
            ]
            counts[link_id] = 1

        # This is obviously inefficient as we have to read everything into memory,
        # but will be fixed when we have separate files for each post.
        for line in open(comment, "r"):
            if line == "\n":
                continue
            node = json.loads(line)
            link_id = node["link_id"][3:]
            if link_id not in graph:
                continue
            graph[link_id].append(node)

        processed_graphs: dict[str, dict] = {}
        for key, value in graph.items():
            res = combine_nodes_to_tree(value)
            if res:
                processed_graphs[key] = res
            else:
                print(f"No root node found for graph {key}")

        gc.collect()
        os.makedirs(
            os.path.dirname(f"data/processed/{file[9:-7]}.json"), exist_ok=True
        )
        with open(f"data/processed/{file[9:-7]}.json", "w") as write:
            for key, data in processed_graphs.items():
                write.write(json.dumps(data) + "\n")

    # NOTE: Right now this is set to just read from the files in Test-LocalCity
    # but it should be all the files you want to read from. It can handle
    # multiple subreddits at once. (data/raw/*/*-RC.txt, for example)

    files_to_process = list(glob("data/raw/Test-LocalCity/*-RC.txt"))
    Parallel(n_jobs=-1)(
        delayed(process_file)(file)
        for file in tqdm(files_to_process, total=len(files_to_process))
    )


def count_size_of_tree(x: dict) -> int:
    """Returns the size of the tree."""
    return sum([count_size_of_tree(y) for y in x["tree"]]) + 1


if __name__ == "__main__":
    main()
