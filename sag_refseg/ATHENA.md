# Running sag_refseg on Athena HPC

## Cluster facts

| Item | Value |
|---|---|
| Scheduler | SLURM |
| GPU partition | `plgrid-gpu-a100` |
| GPU account | `plgunhype-gpu-a100` |
| User scratch | `/net/tscratch/people/plgabedychaj` (`$SCRATCH`) |
| Repo location | `~/proto-VLM` |
| Shared venv | `$SCRATCH/venv` |
| COCO images | `$SCRATCH/ctrl-o/data/images/mscoco/images/train2014` |
| HF cache | `$SCRATCH/.cache/huggingface/hub` |
| Logs | `$SCRATCH/logs/` |

All scripts in `sag_refseg/scripts/slurm_*.sh` are ready to submit with `sbatch`.

---

## Step 0 — Push repo changes to Athena

On your local machine:

```bash
# Commit everything (sag_refseg/ is currently untracked)
git add sag_refseg/ IMPLEMENTATION_NOTES.md
git commit -m "Add sag_refseg module (SaG ICCV 2023 port)"
git push origin model/caltech101   # or whichever branch you're on
```

On Athena login node:

```bash
cd ~/proto-VLM
git pull
```

---

## Step 1 — One-time setup (login node, ~5 min)

```bash
cd ~/proto-VLM
bash sag_refseg/scripts/setup_athena.sh
```

What it does:
- Installs `timm`, `einops`, `transformers`, `scikit-image`, `pycocotools`, `wandb` into `$SCRATCH/venv`
- Clones `lichengunc/refer` to `$SCRATCH/refer/` and patches it for Python 3
- Downloads `data/vocabulary_Gref.txt` from kdwonn/SaG
- Pre-caches `bert-base-uncased` weights to `$SCRATCH/.cache/huggingface/hub`

> **Note**: COCO images are already at `$SCRATCH/ctrl-o/data/images/mscoco/images/` from the ctrl-o experiment — no re-download needed.

---

## Step 2 — Build .npz batch files (SLURM job, ~4–6 h)

```bash
cd ~/proto-VLM
sbatch sag_refseg/scripts/slurm_build_batches.sh
```

Watch progress:
```bash
tail -f $SCRATCH/logs/sag_build_batches_<JOBID>.out
```

Expected output after completion:
```
data/refcoco/
  Gref/  train_batch/ (54 818 files)   val_batch/ (5 000 files)
  unc/   train_batch/ (120 624 files)  val_batch/ testA_batch/ testB_batch/
  unc+/  train_batch/ (120 624 files)  val_batch/ testA_batch/ testB_batch/
```

Verify:
```bash
find $SCRATCH/data/refcoco -name '*.npz' | awk -F/ '{print $(NF-1)}' | sort | uniq -c
```

> This step only needs to run once. Batches are stored in `$SCRATCH/data/refcoco/`.

---

## Step 3 — Train

### Option A: Individual dataset jobs

```bash
cd ~/proto-VLM

# Gref (~1–2 days on A100)
sbatch sag_refseg/scripts/slurm_train_gref.sh

# RefCOCO (unc)
sbatch sag_refseg/scripts/slurm_train_unc.sh

# RefCOCO+ (unc+)
sbatch sag_refseg/scripts/slurm_train_unc+.sh
```

Jobs are independent — submit all three at once if you want parallel runs.

### Option B: Resume from checkpoint

```bash
sbatch sag_refseg/scripts/slurm_train_gref.sh \
    --ckpt $SCRATCH/checkpoints/sag_refseg_Gref/Gref/latest.pth
```

### Monitoring

```bash
# Live log
tail -f $SCRATCH/logs/sag_train_gref_<JOBID>.out

# W&B (if logged in): https://wandb.ai — project "sag_refseg"

# SLURM queue
squeue -u plgabedychaj

# Cancel a job
scancel <JOBID>
```

### Checkpoint layout

```
$SCRATCH/checkpoints/sag_refseg_Gref/
  Gref/
    latest.pth          ← overwritten each epoch
    best.pth            ← best avg_mIoU on val
    training_history.json
```

---

## Step 4 — Evaluate

### Single split

```bash
DATASET=Gref SPLIT=val \
    CKPT=$SCRATCH/checkpoints/sag_refseg_Gref/Gref/best.pth \
    sbatch sag_refseg/scripts/slurm_eval.sh
```

### All splits for unc / unc+

```bash
for SPLIT in val testA testB; do
    DATASET=unc SPLIT=$SPLIT \
        CKPT=$SCRATCH/checkpoints/sag_refseg_unc/unc/best.pth \
        sbatch sag_refseg/scripts/slurm_eval.sh
done

for SPLIT in val testA testB; do
    DATASET=unc+ SPLIT=$SPLIT \
        CKPT="$SCRATCH/checkpoints/sag_refseg_unc+/unc+/best.pth" \
        sbatch sag_refseg/scripts/slurm_eval.sh
done
```

### Retrieve results

```bash
# On Athena
cat $SCRATCH/eval_results/sag_refseg/Gref_val.json | python -m json.tool

# Copy back to local machine
scp plgabedychaj@athena.cyfronet.pl:/net/tscratch/people/plgabedychaj/eval_results/sag_refseg/*.json \
    eval_results/sag_refseg/
```

Result JSON schema:
```json
{
  "dataset": "Gref", "split": "val", "threshold": 0.5,
  "summary": {
    "max_cIoU": 45.2, "max_mIoU": 42.1,
    "avg_cIoU": 47.8, "avg_mIoU": 44.3,
    "min_cIoU": 12.3, "min_mIoU": 11.7
  },
  "per_sample": [...]
}
```

The primary metric to report is **`avg_cIoU`** (cumulative IoU with weighted-average slot pooling).

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'refer'`
The refer library is not on `PYTHONPATH`. The SLURM scripts set this automatically. If running interactively:
```bash
export PYTHONPATH="$SCRATCH/refer:$PYTHONPATH"
```

### `FileNotFoundError: refs(google).p` (refer annotation data missing)
The refer library's `data/` directory is empty after a plain `git clone` — the annotation zips must be downloaded separately. Fix:
```bash
cd $SCRATCH/refer/data
wget http://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip
wget http://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip
wget "http://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip"
unzip refcocog.zip  && rm refcocog.zip
unzip refcoco.zip   && rm refcoco.zip
unzip 'refcoco+.zip' && rm 'refcoco+.zip'
```
Or if the ctrl-o experiment already has them, use that path instead:
```bash
ls $SCRATCH/ctrl-o/refer/data/   # check
# then resubmit with REFER_DIR pointing there
```

### `FileNotFoundError: vocabulary_Gref.txt`
Re-run setup: `bash sag_refseg/scripts/setup_athena.sh`  
Or manually: `curl -fsSL https://raw.githubusercontent.com/kdwonn/SaG/master/data/vocabulary_Gref.txt -o ~/proto-VLM/data/vocabulary_Gref.txt`

### `CUDA out of memory`
Reduce batch size via CLI override:
```bash
sbatch sag_refseg/scripts/slurm_train_gref.sh --batch_size 16
```

### Job killed (time limit)
Jobs are set to 2 days. Resume with `--ckpt latest.pth`:
```bash
sbatch sag_refseg/scripts/slurm_train_gref.sh \
    --ckpt $SCRATCH/checkpoints/sag_refseg_Gref/Gref/latest.pth
```

### `NaN loss` during training
Reduce learning rate or gradient clip:
```bash
sbatch sag_refseg/scripts/slurm_train_gref.sh --lr 5e-6 --grad_clip 0.05
```

### Check disk quota
```bash
du -sh $SCRATCH/data/refcoco/       # batch files (large: ~50–100 GB)
du -sh $SCRATCH/checkpoints/        # checkpoints (~1 GB each)
lfs quota -u plgabedychaj /net/tscratch
```

---

## Reference: full CLI overrides

Any argument defined in `sag_refseg/option.py` can be passed after the config file.
CLI arguments always override JSON config values.

```bash
# Example: longer training, no W&B, specific seed
sbatch sag_refseg/scripts/slurm_train_gref.sh \
    --num_epochs 100 \
    --no_wandb \
    --seed 42
```

---

## File locations quick reference

| What | Where on Athena |
|---|---|
| Repo | `~/proto-VLM/` |
| Venv | `$SCRATCH/venv/` |
| COCO images | `$SCRATCH/ctrl-o/data/images/mscoco/images/train2014/` |
| Refer library | `$SCRATCH/refer/` |
| Vocab file | `~/proto-VLM/data/vocabulary_Gref.txt` |
| Batch files | `$SCRATCH/data/refcoco/{Gref,unc,unc+}/{split}_batch/` |
| Checkpoints | `$SCRATCH/checkpoints/sag_refseg_{dataset}/{dataset}/` |
| Job logs | `$SCRATCH/logs/sag_*.out` |
| Eval results | `$SCRATCH/eval_results/sag_refseg/` |
| W&B project | `sag_refseg` (group = dataset name) |
