# Spectral Unlearning

Requires Linux, Python 3.11 and CUDA 12. Set `CUDA_HOME` to your toolkit location.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install packaging ninja wheel
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS=4 \
  python -m pip install flash-attn==2.8.3 --no-build-isolation --no-cache-dir
python setup_data.py --eval_logs
```

Run:

```bash
CUDA_VISIBLE_DEVICES=0 SEED=0 bash scripts/run_experiments.sh \
  Llama-3.2-1B-Instruct forget05 NPO
```

Use `bash scripts/run_experiments.sh --list` to see available combinations.

Results: `saves/experiments/<run>/results.tsv` and `results.tex`.
