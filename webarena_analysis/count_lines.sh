#!/bin/bash

DIRECTORY="/Users/guozhitong/11711-webarena/data"

if [ -d "$DIRECTORY" ]; then
    find "$DIRECTORY" -name '*.txt' -exec wc -l {} \;
else
    echo "Directory does not exist."
fi