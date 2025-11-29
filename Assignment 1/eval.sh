#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <Provide file name>"
    exit 1
fi

# Get the input data file from the command line argument
file="$1"

# Check if the file exists
if [ -e "$file" ]; then
    python3 1.py $file
else
    echo "File '$file' not found."
fi