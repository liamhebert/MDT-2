import orjson
from tqdm import tqdm
import re
from PIL import Image
from io import BytesIO
from pathlib import Path
import socket
import urllib3.util.connection as urllib_cn
from glob import glob
import os

from requests_futures.sessions import FuturesSession
from requests_ratelimiter import LimiterAdapter

from requests.adapters import HTTPAdapter, Retry
import requests

IMAGE_FILES_PATH = "/mnt/DATA/reddit_share/images_files"

ADDED_IMAGES_PATH = "./added_images"
PROCESSED_FILES_PATH = "./processed"


def allowed_gai_family():
    """Forcing requests to use ipv4, which caused issues on some systems."""
    return socket.AF_INET


urllib_cn.allowed_gai_family = allowed_gai_family


s = requests.Session()
s_imgur = requests.Session()
s_redditmedia = requests.Session()

s_imgur.headers.update(
    {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit"
        + "/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
    }
)

# retry once
retries = Retry(
    total=1, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504]
)

# the per_minute limiter ensures we dont get banned from imgur.
# Experiment with this at your own risk.
imgur_adaptor = LimiterAdapter(per_minute=100)
redditmedia_adaptor = LimiterAdapter(per_minute=100)


# s.mount('http://', HTTPAdapter(max_retries=retries))
# s.mount('https://', HTTPAdapter(max_retries=retries))
s.mount("http://", HTTPAdapter())
s.mount("https://", HTTPAdapter())
s_imgur.mount("https://i.imgur.com/", imgur_adaptor)
s_imgur.mount("http://i.imgur.com/", imgur_adaptor)
s_imgur.mount("http://imgur.com/", imgur_adaptor)
s_redditmedia.mount("https://i.redditmedia.com/", redditmedia_adaptor)

image_session = FuturesSession(
    max_workers=18, session=s
)  # one worker per cpu core
imgur_session = FuturesSession(
    max_workers=20, session=s_imgur
)  # one worker per cpu core
redditmedia_session = FuturesSession(max_workers=20, session=s_redditmedia)
deleted_img_url = (
    "https://i.redd.it/EwG5Emc9PelOE-9TdeB2JlFwOK47ilV_bv0OWJXbpeY.jpg?"
    "auto=webp&amp;s=87d44a717eba5831e219c9a88b6d91c9f74cc333"
)

deleted_img = Image.open(
    BytesIO(image_session.get(deleted_img_url, timeout=3).result().content)
).getdata()
other_deleted_img = Image.open(
    "/mnt/DATA/reddit_share/" + "/deleted_imgur.png"
).getdata()


def main(file):
    """
    Extract images from file
    """
    processed_data_records = []
    image_links = []
    # print('finding images')

    # for file in list(glob('data/pruned/*/*.json')):
    path = os.path.dirname(file)
    file_name = os.path.basename(file)
    subreddit = file_name.split(".")[0]
    topic = path.split("/")[-1]
    os.makedirs(ADDED_IMAGES_PATH + "/" + topic, exist_ok=True)
    os.makedirs(f"{IMAGE_FILES_PATH}/{topic}/{subreddit}", exist_ok=True)

    with open(path + "/" + file_name, "r") as read:
        for line in tqdm(read, position=1, desc="finding images"):
            data = orjson.loads(line)
            link_id = data["id"]
            image_links += get_images(subreddit, link_id, data, topic)
            processed_data_records += [data]

    futures = []
    prefix = f"{IMAGE_FILES_PATH}/{topic}/{subreddit}/"
    images_known = list(glob(f"{prefix}/*/*"))
    images_known = [x.replace(prefix, "") for x in images_known]
    valid_formats = [".jpg", ".jpeg", ".png", ".svg"]
    # print('queuing downloads')

    for parent_id, id, images in tqdm(
        image_links, position=1, desc="queuing downloads"
    ):
        images = [x for x in images if any([y in x for y in valid_formats])]
        for i, image in enumerate(images):
            path = f"{prefix}/{parent_id}"
            if f"{parent_id}/{id}-{i}.png" in images_known:
                continue
            # futures += [(id, path, i)]
            # file.write(','.join([image, id, path, str(i)]) + '\n')

            # if the link does not start with 'i.imgur.com' but has
            # 'i.imgur.com' in it, use imgur session find the i.imgur.com
            # link and use that
            if (
                not image.startswith("https://i.imgur.com/")
                and "i.imgur.com/" in image
            ):  # some weird websites have imgur links but not direct links
                image = "https://i.imgur.com/" + image.split("i.imgur.com/")[-1]

            if image.startswith("https://i.imgur.com/") or image.startswith(
                "http://i.imgur.com/"
            ):
                futures += [
                    imgur_session.get(
                        image,
                        hooks={"response": hook_factory(id, path, 0)},
                        timeout=3,
                    )
                ]
            elif image.startswith("https://i.redditmedia.com/"):
                futures += [
                    redditmedia_session.get(
                        image,
                        hooks={"response": hook_factory(id, path, 0)},
                        timeout=3,
                    )
                ]
            else:
                futures += [
                    image_session.get(
                        image,
                        hooks={"response": hook_factory(id, path, 0)},
                        timeout=3,
                    )
                ]

    # print('waiting for results...')
    progress = tqdm(futures, miniters=1, position=1, desc="waiting for images")
    stat = {"got": 0, "failed": 0}
    for future in progress:
        try:
            res = future.result()
            if res.success[0]:
                stat["got"] += 1
            else:
                if (
                    "imgur" in res.success[-1]
                    or "redditmedia" in res.success[-1]
                ):
                    print(res.success)
                if (
                    not res.success[3] is None
                ) and "Too many open files" in str(res.success[3]):
                    print(res.success)
                stat["failed"] += 1
        except Exception as e:
            # print(e)
            stat["failed"] += 1
            continue
        progress.set_postfix(stat)

    # Updated images known after processing
    images_known = list(glob(f"{IMAGE_FILES_PATH}/{topic}/{subreddit}/*/*"))
    images_known = [x.replace(prefix, "") for x in images_known]

    def check_images(comment):
        # Flag is used to filter out comments that do not have images

        # flag = False
        # if len(comment['images']) != 0:
        #     #print(comment['images'])
        #     flag = True
        corrected = []
        for image in comment["images"]:
            link_id = image.split("/")[-2]
            img_path = link_id + "/" + image.split("/")[-1]
            if img_path in images_known:
                corrected += [image]
        comment["images"] = corrected
        # if len(comment['images']) == 0 and flag:
        #     print('none there!')
        for x in comment["tree"]:
            check_images(x)

    # print('writing results')
    with open(ADDED_IMAGES_PATH + "/" + topic + "/" + file_name, "wb") as write:
        for x in processed_data_records:
            check_images(x)
            write.write(orjson.dumps(x, option=orjson.OPT_APPEND_NEWLINE))

    # print('done!')


def hook_factory(name, path, i):
    """
    A factory function that returns a function formatting the saved image
    """

    def format_save_image(response, *args, **args_kwargs):

        try:
            if not response.ok:
                response.success = (
                    False,
                    name,
                    path,
                    response.status_code,
                    response.url,
                )
                return response
            img = Image.open(BytesIO(response.content))

            # check for deleted image

            # if it is same as deleted image
            if list(img.getdata()) == list(deleted_img):
                response.success = (
                    False,
                    name,
                    path,
                    "[deleted]",
                    response.url,
                )
                return

            # resize image
            max_width, max_height = 256, 256
            height = int(img.height * max_width / img.width)

            # one dimension with max width/height of 256
            if height > max_width:
                width = int(max_height * img.width / img.height)

                img = img.resize((width, max_height), Image.Resampling.LANCZOS)
            else:
                img = img.resize((max_width, height), Image.Resampling.LANCZOS)

            if list(img.getdata()) == list(other_deleted_img):
                return
            # image name = name + iterator
            image_name = name + "-" + str(i) + ".png"
            Path(path).mkdir(parents=True, exist_ok=True)
            img.save(f"{path}/{image_name}")
            img.close()
            response.success = (True, name, path, None, response.url)
        except Exception as e:
            response.success = (False, name, path, e, response.url)

        return response

    return format_save_image


image_pattern = "https?:\/\/(\S+?(?:jpe?g|png|gif|svg))"
https = "https://"


# prepend https:// to all urls in match_urls
def parse_images(body):
    """
    Scrape for image links in the body text and return them
    """
    # TODO: This currently does not find images that are reddit media hosted.
    # For instance, it would not find
    # https://i.redditmedia.com/rvSomf8la2uI9H6Su6v1TNga5ZP37Lo32izk_iQ8Ykc.jpg?s=bac5c203a7aa004b504f2e05ceb121f7
    # Because it breaks the pattern (ending in jpg for example)
    # But so many images are hosted in reddit media, so we should grab them.
    # NOTE: the link above will go to a website, but wget will download the raw
    # image. We should fix the regex to grab these images, but only the original.
    image_urls = re.findall(image_pattern, body)
    image_urls = [https + url for url in image_urls]
    return image_urls


def get_images(
    subreddit: str, link_id: str, comment: str, topic: str, is_root=True
):
    """
    Get a list of all image links from a comment.
    """
    image_urls = parse_images(comment["body"])

    if "preview" in comment and comment["preview"]:
        image_data = comment["preview"]
        if not image_data["url"].startswith("https://i.redditmedia.com/"):
            print("Image URL not start with redditmedia")
            print(image_data["url"])
        if is_root:
            image_urls.append(image_data["url"])
        else:
            print("Found image in non-root:")
            print(image_data["url"])
    # image_urls = [x for x in image_urls if 'i.imgur.com' in x]
    if len(image_urls) != 0:
        res = [(link_id, comment["id"], image_urls)]
        id = comment["id"]
        comment["images"] = [
            f"images_files/{topic}/{subreddit}/{link_id}/{id}-{i}.png"
            for i, x in enumerate(res)
        ]
    else:
        res = []
        comment["images"] = []

    for child in comment["tree"]:
        res += get_images(subreddit, link_id, child, topic, is_root=False)
    # res = [x for x in res if 'i.imgur.com' not in x[-1]]
    # for x in tree['tree']['children']:
    #     res = res + get_images(parent_id, x)
    return res


if __name__ == "__main__":
    # NOTE: edit me to be all the files you want to process
    for file in tqdm(
        list(glob(PROCESSED_FILES_PATH + "/*/*.json")),
        position=0,
        desc="Files",
    ):
        main(file)
