#!/bin/bash

for file in giga_raw_files/*/*.json; do
    echo "$file"
    mv "$file" "${file%.json}-data.json"
done
