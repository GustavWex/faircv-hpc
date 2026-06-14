# DTU HPC — FairCV Job Submission Assistant

You are helping the user submit, monitor, and debug LLM inference jobs on the DTU HPC cluster for the FairCV gender bias audit project. Load everything below as active context before responding.

---

## Project layout

- Main script: `FairCV_gender_audit.py`
- Run scripts: `run_faircv_<model>.sh` (one per model)
- Results: `results/<model>_n<rows>_<date>.csv`
- Logs: `logs/faircv_<JOBID>.out` and `.err`
- Python env: `~/faircv_env` (activate with `source ~/faircv_env/bin/activate`)
- HF model cache: `/scratch/$LSB_JOBID/hf_cache` (falls back to `/tmp/$LSB_JOBID/hf_cache`)

---

## GPU queues

| Queue | GPU | VRAM | When to use |
|---|---|---|---|
| `gpua100` | A100 | 40 GB or 80 GB | Default — faster queue, fits all 4-bit models |
| `gpua10` | A10 | 24 GB | Backup only for models ≤12 GB in 4-bit |

**Avoid** `gmem=79G` unless strictly needed — very few 80 GB nodes, causes long queue times.
Omit `gmem` entirely unless the model is confirmed to not fit on 40 GB.

---

## Run script template

```bash
#!/bin/bash
#BSUB -J faircv-<model-shortname>
#BSUB -o logs/faircv_%J.out
#BSUB -e logs/faircv_%J.err
#BSUB -W 08:00                        # increase to 24:00 for thinking-mode models
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8000]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -q gpua100

module purge
module load python3/3.11.13
module load cuda/12.4

source $HOME/faircv_env/bin/activate

export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null || echo "")

LOCAL_SCRATCH=/scratch/$LSB_JOBID
if [ ! -d "$LOCAL_SCRATCH" ]; then LOCAL_SCRATCH=/tmp/$LSB_JOBID; fi
export HF_HOME=$LOCAL_SCRATCH/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
mkdir -p "$HF_HOME"
echo "Using HF_HOME=$HF_HOME  ($(df -h $LOCAL_SCRATCH | tail -1 | awk '{print $4}') free)"

mkdir -p logs
echo "Job $LSB_JOBID started on $(hostname) at $(date)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

python FairCV_gender_audit.py \
    --model       "org/model-name" \
    --sample-size 4800 \
    --batch-size  8

echo "Done at $(date)"
```

**Flags for `FairCV_gender_audit.py`:**
- `--model` — HuggingFace model ID (default: `google/gemma-4-26B-A4B-it`)
- `--sample-size` — number of résumés (max 4800)
- `--batch-size` — inference batch size (8 for most; 4 for slow 27B thinking models)
- `--no-thinking` — suppresses thinking mode via `/no_think` system message (Qwen3, Fanar-2)
- `--no-4bit` — full bfloat16 (~54 GB VRAM, requires 80 GB node)

---

## Key LSF commands

```bash
bsub < run_faircv_model.sh    # submit a job
bjobs                          # list your jobs
bjobs -l <JOBID>              # detailed status + queue position
bpeek <JOBID>                 # tail stdout of a running job
bkill <JOBID>                 # cancel a job
```

---

## Known model issues and fixes (already applied in the script)

### pad_token missing (Mistral-7B and others)
**Symptom:** `ValueError: Pipeline with tokenizer without pad_token cannot do batching`
**Fix:** After pipeline creation: `if _pipe.tokenizer.pad_token_id is None: _pipe.tokenizer.pad_token_id = _pipe.model.config.eos_token_id`

### No chat_template (Mistral Small 3.1 Tekken tokenizer)
**Symptom:** `ValueError: Cannot use chat template functions because tokenizer.chat_template is not set`
**Fix:** Inject standard Mistral instruct Jinja template manually after pipeline load.

### Thinking mode causes 100% parse failures (Fanar-2, Qwen3.6)
**Symptom:** Model generates long `<think>...</think>` blocks; `max_new_tokens=16` cuts them mid-block; parser gets no number.
**Fix (already in script):** `max_new_tokens=512`, `parse_score()` strips `<think>` blocks and takes the last number match.

### Thinking mode makes runs too slow to finish in 8h
**Symptom:** ~825/4800 résumés scored in 8h (Fanar-2 27B with thinking).
**Fix:** Use `--batch-size 4` and `-W 24:00` for thinking-mode 27B models.

### Suppressing thinking (Qwen3.6)
**Do not** use `tokenizer_kwargs={"enable_thinking": False}` in pipeline calls — transformers 5.x forwards it to `model.generate()` where it fails.
**Fix:** Use `--no-thinking` flag, which prepends `{"role": "system", "content": "/no_think"}` to each prompt.

### ALLaM tokenizer (permanently dropped)
**Symptom:** `ValueError: Error parsing line b'\x0e' in tokenizer.model` — both fast and slow tokenizer fail.
**Cause:** SentencePiece binary misidentified as tiktoken BPE format. Not fixable without rewriting tokenizer loading.
**Decision:** Do not attempt `humain-ai/ALLaM-7B-Instruct-preview` again.

### tokenizer_kwargs bug
**Symptom:** `ValueError: The following model_kwargs are not used by the model: ['tokenizer_kwargs']`
**Cause:** Passing `tokenizer_kwargs={}` (even empty) in pipeline `__call__` falls through to `generate()` in transformers 5.x.
**Fix:** Already fixed — use `/no_think` system message instead.

---

## VRAM estimates (4-bit quantization)

| Model size | Approx VRAM | Fits on |
|---|---|---|
| 7–12B dense | 4–7 GB | A10 or A100 |
| 24–27B dense | 13–16 GB | A100 40 GB |
| 26B MoE (4B active) | ~10 GB | A10 or A100 |
| 31B dense | ~17 GB | A100 40 GB |
| 27B thinking mode | 13–16 GB + long outputs | A100 40 GB, use 24h |

---

## Completed models (as of 2026-06-14)

| Model | HF ID | Rows | M-F gap |
|---|---|---|---|
| Gemma-4-E4B | google/gemma-4-E4B-it | 462 | -0.49 |
| Gemma-4-12B | google/gemma-4-12B-it | 4612 | -0.50 |
| Qwen2.5-14B | Qwen/Qwen2.5-14B-Instruct | 4750 | -0.22 |
| Mistral-7B | mistralai/Mistral-7B-Instruct-v0.3 | 4609 | -0.65 |
| Gemma-4-26B-MoE | google/gemma-4-26B-A4B-it | 4743 | -2.03 |

Pending: Mistral Small 3.1 24B, Gemma 4-31B, Qwen3.6-27B (no-think), Fanar-2-27B (24h).

---

## When the user invokes this skill

1. Check `bjobs` to see the current queue state.
2. Check `ls -lt results/` for any new completed CSVs.
3. If jobs are pending/running, offer to `bpeek` the running ones.
4. If new CSVs exist, analyse gender bias results across all models.
5. If submitting a new model, use the run script template above and apply the known fixes proactively.
