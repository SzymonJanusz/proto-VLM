# sag_refseg — Implementation Notes

## Upstream source

**kdwonn/SaG** (ICCV 2023 — "Shatter and Gather for Text-to-Image Referring Image Segmentation")
Branch: `master`  
URL: https://github.com/kdwonn/SaG

All source files were fetched from the upstream repository and adapted in-place.
No separate clone is required.

---

## Upstream → local file mapping

| Upstream file | Local destination | Adaptations |
|---|---|---|
| `model/cross_modal_attention.py` | `sag_refseg/model/sag_model.py` | renamed module; updated imports |
| `model/aggregator.py` | `sag_refseg/model/aggregator.py` | relative imports |
| `model/attention.py` | `sag_refseg/model/attention.py` | relative imports |
| `model/decoder.py` | `sag_refseg/model/decoder.py` | relative imports |
| `model/encoders.py` | `sag_refseg/model/encoders.py` | relative imports |
| `model/pos_encoding.py` | `sag_refseg/model/pos_encoding.py` | relative imports |
| `model/vit.py` | `sag_refseg/model/vit.py` | relative imports; sub-package `model/transformers/` added |
| `model/img_encoder_cfg.yaml` | `sag_refseg/model/img_encoder_cfg.yaml` | verbatim |
| `loss/cma_loss.py` | `sag_refseg/loss/cma_loss.py` | relative imports |
| `loss/size_p_loss.py` | `sag_refseg/loss/size_p_loss.py` | relative imports |
| `sync_batchnorm/` | `sag_refseg/sync_batchnorm/` | relative imports |
| `data.py` | `sag_refseg/data/refer_dataset.py` | moved to sub-package; factory functions added |
| `option.py` | `sag_refseg/option.py` | added `--config`, `--data_root`, `--dataset`, `--out_dir`; two-pass argparse |
| `utils.py` | `sag_refseg/utils.py` | added `save_checkpoint`, `load_checkpoint`, `setup_logger`, `update_training_history` |
| `train_cma_recon.py` | `sag_refseg/train.py` | local checkpoint format; `training_history.json`; W&B project name |
| `eval_cma_recon.py` | `sag_refseg/evaluate.py` | outputs JSON to `eval_results/sag_refseg/`; supports `--split` CLI arg |
| `build_batches.py` | `sag_refseg/scripts/build_batches.py` | paths resolved relative to repo root |
| `train_eval_gref.sh` etc. | `sag_refseg/scripts/train_eval_*.sh` | use local entrypoints; JSON configs |

---

## Checkpoint format

Local format (differs from upstream):

```python
# save
torch.save({'state_dict': model.state_dict(), 'hparams': vars(args), 'epoch': epoch}, path)

# load
ckpt = torch.load(path, map_location='cpu')
state_dict = ckpt.get('state_dict', ckpt.get('model'))   # also accepts upstream 'model' key
```

Upstream format used `{'model': …, 'args': …, 'iou': …}`.
`load_checkpoint()` in `sag_refseg/utils.py` accepts both.

---

## Eval output

Results are written to:
```
eval_results/sag_refseg/{dataset}_{split}.json
```

Schema:
```json
{
  "dataset": "Gref",
  "split": "val",
  "ckpt": "checkpoints/sag_refseg_Gref/Gref/best.pth",
  "threshold": 0.5,
  "summary": {
    "max_cIoU": 45.2, "max_mIoU": 42.1,
    "avg_cIoU": 47.8, "avg_mIoU": 44.3,
    "min_cIoU": 12.3, "min_mIoU": 11.7
  },
  "per_sample": [
    {"img_id": "...", "index": 0, "max_iou": 0.61, "avg_iou": 0.58, "min_iou": 0.04}
  ]
}
```

---

## Data preparation

### Prerequisites

1. **MS COCO train2014 images** at `data/coco/train2014/`
2. **refer library**:
   ```bash
   git clone https://github.com/lichengunc/refer external/refer
   ```
3. **COCO API** (for `pycocotools`):
   ```bash
   git clone https://github.com/cocodataset/cocoapi external/coco
   cd external/coco/PythonAPI && make
   ```
   Or: `pip install pycocotools`
4. `pip install scikit-image`
5. **Vocabulary file**: `data/vocabulary_Gref.txt` (from refer repo or upstream SaG)

### Build all batches

```bash
bash sag_refseg/scripts/prepare_data.sh
```

Or individually (from repo root):

```bash
PYTHONPATH=external/refer:external/coco/PythonAPI \
    python sag_refseg/scripts/build_batches.py -d Gref -t train
# ... repeat for each dataset/split combination
```

### Expected layout

```
data/refcoco/
  Gref/
    train_batch/*.npz
    val_batch/*.npz
  unc/
    train_batch/*.npz
    val_batch/*.npz
    testA_batch/*.npz
    testB_batch/*.npz
  unc+/
    train_batch/*.npz
    val_batch/*.npz
    testA_batch/*.npz
    testB_batch/*.npz
```

---

## Dependencies

Added (not previously in the repo):

| Package | Purpose |
|---|---|
| `einops` | tensor rearrangement in model and training |
| `timm` | ViT backbone (via `get_img_backbone`) |
| `transformers` | BERT text encoder (`bert-base-uncased`) |
| `opencv-python` | image visualisation in eval (optional) |

Already present in the repo: `wandb`, `tqdm`, `torch`, `torchvision`.

For data prep only (not needed at train/eval time):
- `pycocotools`
- `scikit-image`
- `refer` (from `external/refer/`)

---

## Training

```bash
# Gref
python sag_refseg/train.py \
    --config sag_refseg/configs/gref.json \
    --data_root data/refcoco \
    --out_dir checkpoints/sag_refseg_Gref

# unc
python sag_refseg/train.py \
    --config sag_refseg/configs/unc.json \
    --data_root data/refcoco \
    --out_dir checkpoints/sag_refseg_unc

# unc+
python sag_refseg/train.py \
    --config sag_refseg/configs/unc+.json \
    --data_root data/refcoco \
    --out_dir checkpoints/sag_refseg_unc+

# Resume from checkpoint
python sag_refseg/train.py \
    --config sag_refseg/configs/gref.json \
    --ckpt checkpoints/sag_refseg_Gref/Gref/latest.pth
```

## Evaluation

```bash
# Gref
python sag_refseg/evaluate.py \
    --config sag_refseg/configs/gref.json \
    --data_split val \
    --ckpt checkpoints/sag_refseg_Gref/Gref/best.pth

# unc
for split in val testA testB; do
    python sag_refseg/evaluate.py \
        --config sag_refseg/configs/unc.json \
        --data_split "$split" \
        --ckpt checkpoints/sag_refseg_unc/unc/best.pth
done
```

---

## Smoke tests (no data required)

```bash
# Import check
python -c "from sag_refseg.model.sag_model import CrossModalAttentionRecon; print('OK')"
python -c "from sag_refseg.data.refer_dataset import ReferDataset; print('OK')"
python -c "from sag_refseg.loss.cma_loss import CMA_Loss; print('OK')"

# Config parsing
python sag_refseg/option.py --config sag_refseg/configs/gref.json
```
