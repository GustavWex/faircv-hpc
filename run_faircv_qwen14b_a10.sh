#!/bin/bash
# DTU HPC job script — FairCV gender audit, Qwen2.5 14B Instruct, A10 queue
#
# Submit:  bsub < run_faircv_qwen14b_a10.sh
#
# ── LSF directives ────────────────────────────────────────────────────────────
#BSUB -J faircv-qwen14b-a10
#BSUB -o logs/faircv_%J.out
#BSUB -e logs/faircv_%J.err
#BSUB -W 08:00
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8000]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -q gpua10

# ── Modules ───────────────────────────────────────────────────────────────────
module purge
module load python3/3.11.13
module load cuda/12.4

# ── Python environment ────────────────────────────────────────────────────────
source $HOME/faircv_env/bin/activate

# ── HuggingFace auth ──────────────────────────────────────────────────────────
export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null || echo "")

# ── Scratch cache ─────────────────────────────────────────────────────────────
LOCAL_SCRATCH=/scratch/$LSB_JOBID
if [ ! -d "$LOCAL_SCRATCH" ]; then
    LOCAL_SCRATCH=/tmp/$LSB_JOBID
fi
export HF_HOME=$LOCAL_SCRATCH/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
mkdir -p "$HF_HOME"
echo "Using HF_HOME=$HF_HOME  ($(df -h $LOCAL_SCRATCH | tail -1 | awk '{print $4}') free)"

# ── Run ───────────────────────────────────────────────────────────────────────
mkdir -p logs

echo "Job $LSB_JOBID started on $(hostname) at $(date)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

python FairCV_gender_audit.py \
    --model  "Qwen/Qwen2.5-14B-Instruct" \
    --sample-size 4800 \
    --batch-size  8

echo "Done at $(date)"
