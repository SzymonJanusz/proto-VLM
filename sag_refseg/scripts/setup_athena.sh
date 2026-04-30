#!/usr/bin/env bash
# One-time setup for sag_refseg on Athena HPC.
# Run from the login node (NOT inside sbatch):
#   bash sag_refseg/scripts/setup_athena.sh
#
# Assumes:
#   - ~/proto-VLM is already cloned / up to date
#   - COCO train2014 images exist at $SCRATCH/ctrl-o/data/images/mscoco/images/train2014
#     (shared with the ctrl-o experiment; no re-download needed)
#   - $SCRATCH/venv already contains PyTorch from prior experiments
set -euo pipefail

SCRATCH=/net/tscratch/people/plgabedychaj
REPO=~/proto-VLM

# --------------------------------------------------------------------------- #
# 0. Dirs
# --------------------------------------------------------------------------- #
mkdir -p "$SCRATCH/logs"
mkdir -p "$SCRATCH/data/refcoco"
mkdir -p "$SCRATCH/checkpoints"
mkdir -p "$SCRATCH/eval_results/sag_refseg"
mkdir -p "$SCRATCH/.cache/huggingface/hub"

echo "==> Directories created"

# --------------------------------------------------------------------------- #
# 1. Modules + venv
# --------------------------------------------------------------------------- #
module load Python/3.10.4
module load CUDA/12.4.0
module load cuDNN/9.2.1.18-CUDA-12.4.0

source "$SCRATCH/venv/bin/activate"
echo "==> Activated venv: $SCRATCH/venv"

# --------------------------------------------------------------------------- #
# 2. Python packages required by sag_refseg
# --------------------------------------------------------------------------- #
echo "==> Installing sag_refseg Python dependencies..."
pip install --quiet \
    "timm>=0.9" \
    "einops>=0.6" \
    "transformers>=4.30" \
    "scikit-image>=0.21" \
    "pycocotools" \
    "wandb"

echo "==> Python packages installed"

# --------------------------------------------------------------------------- #
# 3. refer library (lichengunc/refer)
# --------------------------------------------------------------------------- #
REFER_DIR="$SCRATCH/refer"
if [ ! -d "$REFER_DIR" ]; then
    echo "==> Cloning lichengunc/refer..."
    git clone https://github.com/lichengunc/refer.git "$REFER_DIR"
else
    echo "==> refer already at $REFER_DIR, skipping clone"
fi

# Patch refer.py from Python 2 → 3 (same patch as ctrl-o setup.sh)
echo "==> Patching refer/refer.py for Python 3..."
python - <<'PYEOF'
import re, os
path = os.path.expandvars('$SCRATCH/refer/refer.py').replace('$SCRATCH', '/net/tscratch/people/plgabedychaj')
with open(path) as f:
    src = f.read()
src = re.sub(r'\bprint (.*)', r'print(\1)', src)
src = src.replace('import cPickle as pickle', 'import pickle')
src = src.replace('from external import mask', 'from pycocotools import mask')
src = src.replace("pickle.load(open(ref_file, 'r'))", "pickle.load(open(ref_file, 'rb'))")
with open(path, 'w') as f:
    f.write(src)
print('refer.py patched.')
PYEOF

# --------------------------------------------------------------------------- #
# 4. Vocabulary file
# --------------------------------------------------------------------------- #
VOCAB_DIR="$REPO/data"
mkdir -p "$VOCAB_DIR"
VOCAB_FILE="$VOCAB_DIR/vocabulary_Gref.txt"
if [ ! -f "$VOCAB_FILE" ]; then
    echo "==> Downloading vocabulary_Gref.txt from kdwonn/SaG..."
    curl -fsSL \
        "https://raw.githubusercontent.com/kdwonn/SaG/master/data/vocabulary_Gref.txt" \
        -o "$VOCAB_FILE"
    echo "==> Saved to $VOCAB_FILE"
else
    echo "==> vocabulary_Gref.txt already present, skipping"
fi

# --------------------------------------------------------------------------- #
# 5. Pre-download bert-base-uncased weights to HF cache
# --------------------------------------------------------------------------- #
echo "==> Caching bert-base-uncased..."
export HF_HUB_CACHE="$SCRATCH/.cache/huggingface/hub"
python -c "
from transformers import BertTokenizer, BertModel
BertTokenizer.from_pretrained('bert-base-uncased')
BertModel.from_pretrained('bert-base-uncased')
print('bert-base-uncased cached.')
"

# --------------------------------------------------------------------------- #
# Done
# --------------------------------------------------------------------------- #
echo ""
echo "============================================="
echo "Setup complete!"
echo ""
echo "Paths:"
echo "  Repo:         $REPO"
echo "  Venv:         $SCRATCH/venv"
echo "  refer:        $REFER_DIR"
echo "  COCO images:  $SCRATCH/ctrl-o/data/images/mscoco/images"
echo "  Vocab:        $VOCAB_FILE"
echo "  Batches out:  $SCRATCH/data/refcoco/"
echo "  Checkpoints:  $SCRATCH/checkpoints/"
echo ""
echo "Next: sbatch sag_refseg/scripts/slurm_build_batches.sh"
echo "============================================="
