#!/bin/bash
# Setup CTRL-O + REFER on Athena HPC
# Run from login node: bash setup.sh
set -e

SCRATCH=/net/tscratch/people/plgabedychaj
WORKDIR=$SCRATCH/ctrl-o

mkdir -p "$WORKDIR"
cd "$WORKDIR"

# Redirect pip cache to scratch to avoid home directory quota issues
export PIP_CACHE_DIR=$SCRATCH/.cache/pip
export HF_HOME=$SCRATCH/.cache/huggingface
mkdir -p "$PIP_CACHE_DIR" "$HF_HOME"

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

# --- PyTorch (CUDA 12.4 build, satisfies ^2.2.0 constraint) ---
echo "==> Installing PyTorch 2.4.0 + CUDA 12.4..."
pip install torch==2.4.0 torchvision==0.19.0 \
    --index-url https://download.pytorch.org/whl/cu124 \
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
    "accelerate"

# --- Install CTRL-O as editable package ---
echo "==> Installing CTRL-O package..."
pip install -e "$WORKDIR/CTRL-O" --quiet

# --- Build REFER C extensions ---
echo "==> Building REFER C extensions..."
cd "$WORKDIR/refer"
pip install cython --quiet
python setup.py build_ext --inplace
cd "$WORKDIR"

# --- Pre-download LLM2Vec model (large, ~16 GB) ---
echo "==> Pre-downloading LLM2Vec model (this may take a while)..."
python -c "
from llm2vec import LLM2Vec
LLM2Vec.from_pretrained(
    'McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp',
    peft_model_name_or_path='McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-unsup-simcse',
    device_map='cpu',
    torch_dtype='auto',
)
print('LLM2Vec model cached successfully.')
"

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
