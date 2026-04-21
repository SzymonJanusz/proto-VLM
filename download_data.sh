#!/bin/bash
# Download RefCOCO/RefCOCO+ annotations and COCO train2014 images to scratch.
# Run from login node: bash download_data.sh
set -e

SCRATCH=/net/tscratch/people/plgabedychaj
DATA=$SCRATCH/ctrl-o/data

mkdir -p "$DATA/images/mscoco/images"

# -----------------------------------------------------------------------
# COCO train2014 images (~13 GB)
# RefCOCO and RefCOCO+ both use COCO train2014 images.
# -----------------------------------------------------------------------
if [ ! -d "$DATA/images/mscoco/images/train2014" ]; then
    echo "==> Downloading COCO train2014 images (~13 GB)..."
    wget -c http://images.cocodataset.org/zips/train2014.zip \
        -O "$DATA/train2014.zip"
    echo "==> Unzipping COCO train2014..."
    unzip -q "$DATA/train2014.zip" -d "$DATA/images/mscoco/images/"
    rm "$DATA/train2014.zip"
    echo "==> COCO train2014 done."
else
    echo "==> COCO train2014 already downloaded, skipping."
fi

# -----------------------------------------------------------------------
# RefCOCO annotations
# Primary server: bvisionweb1.cs.unc.edu (may be offline in 2025)
# Fallback: download the pickle files from HuggingFace (see note below)
# -----------------------------------------------------------------------
download_refer() {
    local name=$1       # e.g. "refcoco" or "refcoco+"
    local url="http://bvisionweb1.cs.unc.edu/licheng/referit/data/${name}.zip"

    if [ -d "$DATA/$name" ]; then
        echo "==> $name already downloaded, skipping."
        return
    fi

    echo "==> Downloading $name annotations..."
    wget -c "$url" -O "$DATA/${name}.zip" && \
        unzip -q "$DATA/${name}.zip" -d "$DATA/" && \
        rm "$DATA/${name}.zip" && \
        echo "==> $name done." || \
        echo "WARNING: $name download failed. Primary server may be down.
  Manual fallback: download from https://huggingface.co/datasets/unc-nlp/${name}
  or ask a colleague for the .zip and place the extracted folder at $DATA/$name/"
}

download_refer "refcoco"
download_refer "refcoco+"

# -----------------------------------------------------------------------
# Verify structure
# -----------------------------------------------------------------------
echo ""
echo "==> Data directory structure:"
ls -lh "$DATA/"
echo ""
echo "Expected layout:"
echo "  $DATA/images/mscoco/images/train2014/  (COCO images)"
echo "  $DATA/refcoco/   (instances.json + refs(unc).p)"
echo "  $DATA/refcoco+/  (instances.json + refs(unc).p)"
