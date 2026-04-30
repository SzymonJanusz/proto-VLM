#!/usr/bin/env bash
# Build all .npz batch files for sag_refseg.
#
# Prerequisites (run from repo root):
#   1. MS COCO train2014 images at data/coco/train2014/
#   2. git clone https://github.com/lichengunc/refer external/refer
#   3. git clone https://github.com/cocodataset/cocoapi external/coco
#      cd external/coco/PythonAPI && make
#   4. pip install scikit-image pycocotools
#
# Usage (from repo root):
#   bash sag_refseg/scripts/prepare_data.sh

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHONPATH_EXT="${REPO_ROOT}/external/refer:${REPO_ROOT}/external/coco/PythonAPI"

run_build() {
    PYTHONPATH="${PYTHONPATH_EXT}" python sag_refseg/scripts/build_batches.py -d "$1" -t "$2"
}

echo "=== Building Gref ==="
run_build Gref train
run_build Gref val

echo "=== Building unc ==="
run_build unc train
run_build unc val
run_build unc testA
run_build unc testB

echo "=== Building unc+ ==="
run_build "unc+" train
run_build "unc+" val
run_build "unc+" testA
run_build "unc+" testB

echo "All batches built successfully."
