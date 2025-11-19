#!/bin/bash

# list directories in the given path $1 stored in an array
DIRS=()
while IFS= read -r -d $'\0' dir; do
    DIRS+=("$dir")
done < <(find "$1" -mindepth 1 -maxdepth 1 -type d -print0)

# iterate over the directories and call single.sh for each
for dir in "${DIRS[@]}"; do
    /Data_large/marine/PythonProjects/SAR/sarpyx/scripts/up/single.sh "$dir"
done