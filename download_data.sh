#!/bin/bash
# Download RefCOCO/RefCOCO+ annotations and COCO train2014 images to scratch.
# Run from login node: bash download_data.sh
set -e

SCRATCH=/net/tscratch/people/plgabedychaj
DATA=$SCRATCH/ctrl-o/data

mkdir -p "$DATA/images/mscoco/images"

# -----------------------------------------------------------------------
# COCO train2014 images (~13 GB)
# -----------------------------------------------------------------------
if [ ! -d "$DATA/images/mscoco/images/train2014" ]; then
    if [ -f "$DATA/train2014.zip" ]; then
        echo "==> Found existing train2014.zip, unzipping..."
    else
        echo "==> Downloading COCO train2014 images (~13 GB)..."
        wget -c http://images.cocodataset.org/zips/train2014.zip \
            -O "$DATA/train2014.zip"
    fi
    echo "==> Unzipping COCO train2014..."
    unzip -q "$DATA/train2014.zip" -d "$DATA/images/mscoco/images/"
    rm "$DATA/train2014.zip"
    echo "==> COCO train2014 done."
else
    echo "==> COCO train2014 already downloaded, skipping."
    rm -f "$DATA/train2014.zip"
fi

# -----------------------------------------------------------------------
# RefCOCO / RefCOCO+ annotations
# Primary UNC server is down. Falls back to Wayback Machine archive.
# -----------------------------------------------------------------------
UNC_BASE="http://bvisionweb1.cs.unc.edu/licheng/referit/data"
WBM_BASE="https://web.archive.org/web/20220413"

download_refer() {
    local name=$1   # "refcoco" or "refcoco+"
    local wbm_ts=$2 # Wayback Machine timestamp for this file

    if [ -d "$DATA/$name" ] && [ "$(ls -A $DATA/$name 2>/dev/null)" ]; then
        echo "==> $name already downloaded, skipping."
        return 0
    fi

    mkdir -p "$DATA/$name"

    # Try primary server first
    echo "==> Trying UNC server for $name..."
    if wget -q --timeout=15 --tries=1 \
            -c "$UNC_BASE/${name}.zip" -O "$DATA/${name}.zip" 2>/dev/null; then
        echo "==> UNC server succeeded."
    else
        rm -f "$DATA/${name}.zip"
        echo "==> UNC server down. Trying Wayback Machine archive..."
        wget -c "${WBM_BASE}${wbm_ts}/${UNC_BASE}/${name}.zip" \
            -O "$DATA/${name}.zip"
    fi

    echo "==> Unzipping $name..."
    unzip -q "$DATA/${name}.zip" -d "$DATA/"
    rm "$DATA/${name}.zip"
    echo "==> $name done."
}

# Wayback Machine timestamps from the archived URLs in github.com/lichengunc/refer/issues/14
download_refer "refcoco"  "011718"
download_refer "refcoco+" "011656"

# -----------------------------------------------------------------------
# Verify structure
# -----------------------------------------------------------------------
echo ""
echo "==> Data directory structure:"
ls -lh "$DATA/"
for name in refcoco refcoco+; do
    echo ""
    echo "  $name/:"
    ls "$DATA/$name/" 2>/dev/null || echo "    MISSING"
done
echo ""
echo "Expected files per dataset: instances.json  refs(unc).p"
