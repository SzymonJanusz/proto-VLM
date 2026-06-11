# RIS Evaluation Guide

Evaluation of three referring image segmentation (RIS) methods on RefCOCO / RefCOCO+ / RefCOCOg:

| Method         | Script                          | Benchmark naming              |
|----------------|---------------------------------|-------------------------------|
| **SaG**        | `sag_refseg/evaluate.py`        | Gref / unc / unc+             |
| **CTRL-O**     | `inference_refer.py`            | refcocog / refcoco / refcoco+ |
| **PNP** (ours) | `scripts/evaluate_pnp_refer.py` | Gref / unc / unc+             |

Results from all three are compared with `scripts/compare_ris_results.py`.

---

## Prerequisites (Athena)

```bash
cd ~/proto-VLM && git pull

# proto-non-param must exist as a subdirectory of proto-VLM (needed by PNP eval)
# If cloned separately, create a symlink:
ln -sfn ~/proto-non-param ~/proto-VLM/proto-non-param

source /net/tscratch/people/plgabedychaj/venv/bin/activate
```

---

## 1. SaG

### 1a. Retrain (required — previous runs used random backbone weights)

The `vit_small_patch16_384` backbone previously failed to load pretrained weights due to a
timm 1.0.x API change. This is fixed in `sag_refseg/model/encoders.py`. Retrain all three
datasets from scratch:

```bash
cd ~/proto-VLM
sbatch sag_refseg/scripts/slurm_train_gref.sh
sbatch sag_refseg/scripts/slurm_train_unc.sh
sbatch sag_refseg/scripts/slurm_train_unc+.sh
```

Each job runs up to 2 days (50 epochs). Verify the backbone fix is active in the first log lines:

```bash
# Replace <JOBID> with the actual job ID from sbatch output
tail -20 /net/tscratch/people/plgabedychaj/logs/sag_train_gref_<JOBID>.out
# Expected: [encoders] Pretrained vit_small_patch16_384: N missing, M unexpected keys
# Bad sign: "No pretrained weights exist for this model. Using random initialization."
```

### 1b. Evaluate

Run after training completes (`best.pth` exists in the checkpoint directory):

```bash
SCRATCH=/net/tscratch/people/plgabedychaj

# RefCOCOg (Gref) — val only
DATASET=Gref SPLIT=val \
    CKPT=$SCRATCH/checkpoints/sag_refseg_Gref/Gref/best.pth \
    sbatch sag_refseg/scripts/slurm_eval.sh

# RefCOCO (unc) — val + testA + testB
for split in val testA testB; do
    DATASET=unc SPLIT=$split \
        CKPT=$SCRATCH/checkpoints/sag_refseg_unc/unc/best.pth \
        sbatch sag_refseg/scripts/slurm_eval.sh
done

# RefCOCO+ (unc+) — val + testA + testB
for split in val testA testB; do
    DATASET=unc+ SPLIT=$split \
        CKPT=$SCRATCH/checkpoints/sag_refseg_unc+/unc+/best.pth \
        sbatch sag_refseg/scripts/slurm_eval.sh
done
```

Results → `eval_results/sag_refseg/{DATASET}_{SPLIT}.json`  
Expected performance (SaG paper): ~50–57% oIoU on RefCOCO.

---

## 2. CTRL-O

No training needed — uses the pretrained CTRL-O checkpoint.

```bash
SCRATCH=/net/tscratch/people/plgabedychaj
CKPT=$SCRATCH/ctrl-o/pretrained_models/ctrlo/pretrained_model.ckpt
CONFIG=$SCRATCH/ctrl-o/pretrained_models/ctrlo/config.yaml

# RefCOCOg — val only, google split
DATASET=refcocog SPLITS="val" SPLITBY=google \
    CKPT=$CKPT CONFIG=$CONFIG \
    sbatch scripts/slurm_eval_ctrlo.sh

# RefCOCO — val + testA + testB
DATASET=refcoco \
    CKPT=$CKPT CONFIG=$CONFIG \
    sbatch scripts/slurm_eval_ctrlo.sh

# RefCOCO+ — val + testA + testB
DATASET=refcoco+ \
    CKPT=$CKPT CONFIG=$CONFIG \
    sbatch scripts/slurm_eval_ctrlo.sh
```

Each job takes up to 8 h (LLM2Vec text encoder is large, 64 GB RAM).  
Results → `eval_results/ctrlo/{refcocog,refcoco,refcoco+}_metrics.json`

---

## 3. PNP (ours — zero-shot RIS)

No RIS fine-tuning — the trained PNP checkpoint is evaluated zero-shot. The model
CLIP-encodes the referring expression, projects it into visual space via the learned
`text_projection_head`, and uses cosine similarity with DINOv2 patch tokens as a
spatial activation map.

Set `PNP_CKPT` to your trained checkpoint. PNP saves only one file (`ckpt.pth`,
overwritten each epoch — no separate best.pth). For the vg+coco run:

```bash
SCRATCH=/net/tscratch/people/plgabedychaj
PNP_CKPT=$SCRATCH/train_logs/coco_vg_baseline/ckpt.pth

# RefCOCOg (Gref) — val only
DATASET=Gref SPLIT=val CKPT=$PNP_CKPT \
    sbatch scripts/slurm_eval_pnp_refer.sh

# RefCOCO (unc) — val + testA + testB
for split in val testA testB; do
    DATASET=unc SPLIT=$split CKPT=$PNP_CKPT \
        sbatch scripts/slurm_eval_pnp_refer.sh
done

# RefCOCO+ (unc+) — val + testA + testB
for split in val testA testB; do
    DATASET=unc+ SPLIT=$split CKPT=$PNP_CKPT \
        sbatch scripts/slurm_eval_pnp_refer.sh
done
```

Results → `eval_results/pnp_refer/{DATASET}_{SPLIT}.json`

> **Note:** SaG and PNP share the same pre-built `.npz` batch files
> (`$SCRATCH/data/refcoco/`) so their per-sample coverage is identical.
> CTRL-O uses the REFER API directly — slight coverage differences may exist.

---

## 4. Compare results

After all eval jobs finish, generate a unified Markdown table:

```bash
cd ~/proto-VLM
python scripts/compare_ris_results.py

# Save to file
python scripts/compare_ris_results.py --out eval_results/comparison.md
cat eval_results/comparison.md
```

The script normalises all metrics to the same scale (percentages):

| Metric   | Meaning                                  |
|----------|------------------------------------------|
| **oIoU** | Overall IoU = ΣI / ΣU across all samples |
| **mIoU** | Mean per-sample IoU                      |

---

## Checking job status

```bash
squeue -u $USER

# Logs
tail -f /net/tscratch/people/plgabedychaj/logs/sag_train_gref_<JOBID>.out
tail -f /net/tscratch/people/plgabedychaj/logs/sag_eval_<JOBID>.out
tail -f /net/tscratch/people/plgabedychaj/logs/ctrlo_eval_<JOBID>.out
tail -f /net/tscratch/people/plgabedychaj/logs/pnp_eval_<JOBID>.out

# Training progress (epochs completed, best metric so far)
python -c "
import json, sys
h = json.load(open(sys.argv[1]))
print(len(h), 'epochs')
print('best avg_cIoU:', max(e['avg_cIoU'] for e in h))
" /net/tscratch/people/plgabedychaj/checkpoints/sag_refseg_Gref/Gref/training_history.json
```

---

## Dataset name mapping

| SaG / PNP | CTRL-O     | Benchmark               |
|-----------|------------|-------------------------|
| `Gref`    | `refcocog` | RefCOCOg (Google split) |
| `unc`     | `refcoco`  | RefCOCO (unc split)     |
| `unc+`    | `refcoco+` | RefCOCO+ (unc+ split)   |
