#!/usr/bin/env bash
set -euo pipefail
IMAGE_COUNT="${IMAGE_COUNT:-100}"; IMAGE_WIDTH="${IMAGE_WIDTH:-256}"; IMAGE_HEIGHT="${IMAGE_HEIGHT:-256}"; ITERATIONS="${ITERATIONS:-3}"
python3 generate_dataset.py --output-dir data/input --count "$IMAGE_COUNT" --width "$IMAGE_WIDTH" --height "$IMAGE_HEIGHT"
make clean
make
mkdir -p artifacts
./cuda_image_batch --input-dir data/input --output-dir data/output --iterations "$ITERATIONS" 2>&1 | tee artifacts/run.log
cp data/output/results.csv artifacts/results.csv
printf "Input images: %s\nDimensions: %sx%s\nIterations per image: %s\nKernel: RGB 3x3 box blur\n" "$IMAGE_COUNT" "$IMAGE_WIDTH" "$IMAGE_HEIGHT" "$ITERATIONS" > artifacts/experiment_summary.txt
