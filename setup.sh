#!/bin/bash
# Setup CTRL-O + REFER on Athena HPC
# Run from login node: bash setup.sh
set -e

SCRATCH=/net/tscratch/people/plgabedychaj
WORKDIR=$SCRATCH/ctrl-o

mkdir -p "$WORKDIR"
cd "$WORKDIR"

# Redirect caches to scratch to avoid home directory quota issues
# HF_HUB_CACHE: model/dataset cache (large) → scratch
# Token stays in default ~/.cache/huggingface/token
export PIP_CACHE_DIR=$SCRATCH/.cache/pip
export HF_HUB_CACHE=$SCRATCH/.cache/huggingface/hub
mkdir -p "$PIP_CACHE_DIR" "$HF_HUB_CACHE"

echo "==> Loading modules..."
module load Python/3.10.4
module load CUDA/12.4.0
module load cuDNN/9.2.1.18-CUDA-12.4.0

# --- Clone repos ---
if [ ! -d "CTRL-O" ]; then
    echo "==> Cloning CTRL-O..."
    git clone https://github.com/dido1998/CTRL-O.git
    cd CTRL-O
    git checkout ed21675899bb700bd1d57f0d1e7693376bffc821
    cd "$WORKDIR"
else
    echo "==> CTRL-O already cloned, skipping."
fi

if [ ! -d "refer" ]; then
    echo "==> Cloning REFER API..."
    git clone https://github.com/lichengunc/refer.git
else
    echo "==> refer already cloned, skipping."
fi

# --- Create virtual environment ---
if [ ! -d "venv" ]; then
    echo "==> Creating venv..."
    python -m venv "$WORKDIR/venv"
fi
source "$WORKDIR/venv/bin/activate"
pip install --upgrade pip --quiet

# --- PyTorch (use 2.2.0 + torchvision 0.17.0 to match ctrl-o's constraints) ---
echo "==> Installing PyTorch 2.2.0 + CUDA 12.1..."
pip install torch==2.2.0 torchvision==0.17.0 \
    --index-url https://download.pytorch.org/whl/cu121 \
    --quiet

# --- CTRL-O core dependencies (from pyproject.toml) ---
echo "==> Installing CTRL-O dependencies..."
pip install --quiet \
    "pytorch-lightning==2.1.4" \
    "torchmetrics==1.3.0" \
    "torchdata==0.7.1" \
    "hydra-core==1.3.2" \
    "hydra-zen==0.7.0" \
    "pyyaml==6.0.1" \
    "timm==1.0.7" \
    "llm2vec==0.2.3" \
    "einops==0.6.0" \
    "opencv-python==4.10.0.84" \
    "scikit-image==0.21" \
    "scipy<=1.10" \
    "scikit-learn" \
    "tqdm" \
    "wandb" \
    "huggingface-hub>=0.32.4" \
    "tensorboardx" \
    "braceexpand" \
    "pyamg" \
    "omegaconf" \
    "pycocotools" \
    "matplotlib" \
    "torchtyping" \
    "accelerate"

# --- Install CTRL-O as editable package (--no-deps: torch/torchvision already pinned above) ---
echo "==> Installing CTRL-O package..."
pip install -e "$WORKDIR/CTRL-O" --no-deps --quiet

# --- Build REFER C extensions ---
# Convert refer.py from Python 2 to Python 3
echo "==> Patching refer/refer.py for Python 3..."
python -c "
import re
path = '$WORKDIR/refer/refer.py'
with open(path) as f:
    src = f.read()
src = re.sub(r'\bprint (.*)', r'print(\1)', src)
src = src.replace('import cPickle as pickle', 'import pickle')
src = src.replace('from external import mask', 'from pycocotools import mask')
with open(path, 'w') as f:
    f.write(src)
print('refer.py patched.')
"

# --- Pre-download LLM2Vec model (large, ~16 GB) ---
# Requires:
#   1. Access granted to https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
#   2. HF token set: run `huggingface-cli login` once, or export HUGGING_FACE_HUB_TOKEN=<token>
# Download LLM2Vec model files to cache (no RAM loading — that happens on the compute node).
# Requires approved access to https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
echo "==> Downloading LLM2Vec model weights to cache (no loading)..."
if ! huggingface-cli whoami &>/dev/null; then
    echo "WARNING: Not logged into HuggingFace. Run: huggingface-cli login"
else
    huggingface-cli download McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp \
        --cache-dir "$HF_HUB_CACHE" \
        && echo "==> LLM2Vec base model cached." \
        || echo "WARNING: LLM2Vec download failed (access pending?). Re-run setup.sh once access is approved."
    huggingface-cli download McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-unsup-simcse \
        --cache-dir "$HF_HUB_CACHE" \
        && echo "==> LLM2Vec adapter cached." \
        || echo "WARNING: LLM2Vec adapter download failed."
fi

# --- Download pretrained CTRL-O checkpoint ---
echo "==> Downloading pretrained CTRL-O model from HuggingFace..."
huggingface-cli download adidolkar123/pretrained_coco_vgcoco \
    --local-dir "$WORKDIR/pretrained_models/ctrlo" \
    --local-dir-use-symlinks False

echo ""
echo "======================================="
echo "Setup complete!"
echo "Activate env: source $WORKDIR/venv/bin/activate"
echo "Workdir:      $WORKDIR"
echo "======================================="
