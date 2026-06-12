# FairCV HPC — Project Context

Workflow notes for anyone continuing this experiment on DTU HPC.

## Models being evaluated

| Model | Size | Rationale |
|---|---|---|
| `google/gemma-4-26B-A4B-it` | 26B MoE (4B active) | Heavy bias mitigation (Google) — key comparison point |
| `google/gemma-4-12B-it` | 12B | Intermediate Gemma checkpoint |
| `google/gemma-4-E4B-it` | E4B | Too small — run at 500 samples was non-conclusive (see below) |
| `mistralai/Mistral-7B-Instruct-v0.3` | 7B | Minimal safety tuning — expected to show more raw bias |
| `Qwen/Qwen2.5-14B-Instruct` | 14B | Different cultural training distribution |

All models use 4-bit quantization (`bitsandbytes`), 4800 samples, `batch_size=8`.

## Dataset

`FairCVdb.npy` — pre-processed FairCV résumé dataset, 4800 samples covering male/female/neutral gender labels.

## Key findings so far

**E4B run (500 samples) — non-conclusive:**
- Bimodal score distribution: ~48% scored 85, ~22% scored 1
- Gender means nearly identical (male=64.82, female=65.31, neutral=63.73)
- Likely insufficient sample size and/or model too small to detect subtle bias
- Full 4800-sample runs needed for statistical power

## GPU queue guidance (DTU HPC)

| Queue | GPU | Notes |
|---|---|---|
| `gpua100` | A100 (CC 8.0) | **Use this** — fastest turnover in practice |
| `gpuv100` | V100 (CC 7.0) | **Incompatible** — PyTorch build requires CC ≥ 7.5, causes float32 fallback + OOM |
| `gpuh100` | H100 | Restricted access |
| `gpul40s` | L40S (CC 8.9) | Compatible but slower turnover than gpua100 |

**Important:** The 26B-A4B model requires `gmem=79G` in the LSF script to land on an 80 GB A100.
Models ≤14B fit on 40 GB A100s — omit the `gmem` constraint for faster scheduling.

## Bugs fixed

- `quantization_config` leaked into `generate()` call → fixed by using `model_kwargs=_load_kw` in `pipeline()`
- Removed `temperature=None, top_p=None` from pipeline calls (deprecation warning with `do_sample=False`)
- `device_map="auto"` estimates memory from bfloat16 size (~52 GB) and dispatches layers to CPU, causing bitsandbytes 4-bit to refuse → fixed with `device_map={"": 0}` (bypasses estimation, puts all layers on GPU 0)

## Storage notes

- Home dir quota is ~25 GB — not enough for large model weights
- Compute node `/tmp` is ~700 GB — model downloads fit fine there (handled automatically by `run_faircv.sh`)
- If `/work3/$USER` is allocated, set `HF_HOME=/work3/$USER/hf_cache` in the run script to avoid re-downloading across jobs

## Python environment

```bash
module load python3/3.11.13
python -m venv $HOME/faircv_env
source $HOME/faircv_env/bin/activate
pip install -r requirements_hpc.txt
```

## Submitting and monitoring

```bash
bsub < run_faircv.sh          # submit
bjobs                          # check status
tail -f logs/faircv_<JOBID>.out   # stream output
```

Results saved as `faircv_results_<timestamp>.csv` in the project directory.
