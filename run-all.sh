#!/bin/bash

# List of your specific folders
folders=(
    "single-label-flipping-attack"
    "multi-label-random-flipping-attack"
    "model-weight-poisoning-attack"
    "backdoor-pattern-attack"
    "freerider-attack"
    "scaling-attack"
)

for folder in "${folders[@]}"; do
    echo "Entering $folder"
    cd "$folder" || { echo "Failed to cd into $folder"; exit 1; }

    echo "Running app.py in $folder"
    python3 app.py > cout.log 2>&1

    echo "Returning to parent directory"
    cd ..

    echo "Sleeping for 3 minutes..."
    sleep 180
done

echo "✅ All folders processed."
