#!/bin/bash

#######################################

required_command="black"
code_directories=("pina" "tests")

#######################################

# Test for required program
if ! command -v $required_command >/dev/null 2>&1; then
    echo "I require $required_command but it's not installed. Install dev dependencies."
    echo "Aborting." >&2
    exit 1
fi

# Run black formatter
for dir in "${code_directories[@]}"; do
    python -m black --line-length 80 "$dir"
done