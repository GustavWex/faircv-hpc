#!/bin/bash
# DTU HPC job script — FairCV gender audit, Gemma 4 27B MoE
#
# Submit:  bsub < run_faircv.sh
# Status:  bjobs
# Output:  bpeek <jobid>   or   tail -f logs/faircv_<jobid>.out
#
# ── LSF directives ────────────────────────────────────────────────────────────
#BSUB -J faircv-gemma4
#BSUB -o logs/faircv_%J.out
#BSUB -e logs/faircv_%J.err
#BSUB -W 08:00                        # wall-clock HH:MM — 8 h covers ~1000 resumes
#BSUB -n 8                            # CPU cores (for data loading / tokenisation)
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8000]"           # 8 GB per core → 64 GB RAM total
#BSUB -gpu "num=1:mode=exclusive_process:gmem=79G"
#BSUB -q gpua100

# ── Modules ───────────────────────────────────────────────────────────────────
module purge
module load python3/3.11.13
module load cuda/12.4                 # adjust: `module avail cuda`

# ── Python environment ────────────────────────────────────────────────────────
# Create once on a login node:
#   python -m venv $HOME/faircv_env
#   source $HOME/faircv_env/bin/activate
#   pip install -r requirements_hpc.txt
source $HOME/faircv_env/bin/activate

# ── HuggingFace auth ──────────────────────────────────────────────────────────
# Read the token saved by `huggingface-cli login` (works regardless of HF_HOME)
export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null || echo "")

# ── Cache model weights on the compute node's local scratch ──────────────────
# /scratch/$LSB_JOBID is local to this node, typically 100s of GB, wiped after job.
# The model (~16 GB) is downloaded fresh each run — adds ~2 min to startup.
# Once /work3/s255911 is allocated, replace with: export HF_HOME=/work3/s255911/hf_cache
LOCAL_SCRATCH=/scratch/$LSB_JOBID
if [ ! -d "$LOCAL_SCRATCH" ]; then
    LOCAL_SCRATCH=/tmp/$LSB_JOBID   # fallback if /scratch not available
fi
export HF_HOME=$LOCAL_SCRATCH/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
mkdir -p "$HF_HOME"
echo "Using HF_HOME=$HF_HOME  ($(df -h $LOCAL_SCRATCH | tail -1 | awk '{print $4}') free)"

# ── Job parameters — edit these ───────────────────────────────────────────────
SAMPLE_SIZE=4800
BATCH_SIZE=8
# EXTRA_FLAGS="--no-4bit"   # uncomment for full bfloat16 (needs A100 80 GB)
# EXTRA_FLAGS="--mock"      # dry-run with fake scorer, no GPU required

# ── Run ───────────────────────────────────────────────────────────────────────
mkdir -p logs

echo "Job $LSB_JOBID started on $(hostname) at $(date)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

python FairCV_gender_audit.py \
    --sample-size "$SAMPLE_SIZE" \
    --batch-size  "$BATCH_SIZE"  \
    ${EXTRA_FLAGS:-}

echo "Done at $(date)"
