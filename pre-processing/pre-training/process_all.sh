#!/bin/bash

echo "===STARTING 1===="
python3.10 1-organize_comments.py
echo "===STARTING 2===="
python3.10 2-combine_and_compress_trees.py
echo "===STARTING 3===="
python3.10 3-get_images.py
