# FairCV HPC — Gender Audit

Runs a gender-bias audit of LLMs on résumés from the FairCV dataset using the DTU HPC cluster (LSF / A100 GPUs).

## Files

| File | Description |
|---|---|
| `FairCV_gender_audit.py` | Main audit script |
| `FairCVdb.npy` | FairCV résumé dataset (pre-processed) |
| `requirements_hpc.txt` | Python dependencies |
| `run_faircv.sh` | LSF job script (Gemma 4 27B, A100) |

## Setup (once, on a login node)

```bash
module load python3/3.11.13
python -m venv $HOME/faircv_env
source $HOME/faircv_env/bin/activate
pip install -r requirements_hpc.txt
```

Log in to HuggingFace (required for gated models):

```bash
huggingface-cli login
```

## Running on HPC

```bash
bsub < run_faircv.sh
bjobs                         # check status
tail -f logs/faircv_<JOBID>.out
```

Results are saved as `faircv_results_<timestamp>.csv`.
