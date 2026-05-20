# WorldChanger (VLA Foundry)

## Datasets

All datasets are stored on scratch at `/e/scratch/scifi/nadimpalli2/datasets/`.

### LLM Training Data

- **DCLM-baseline-1.0** (from `mlfoundations/dclm-baseline-1.0-parquet`): `/e/scratch/profound/thoeppe/data/WorldChanger/dclm/`
  - Lives under `profound/thoeppe`'s scratch (not the path noted above). Downloaded for the from-scratch LLM stage; matches the paper's Foundry-LLM-1.2B recipe (§4.1).
  - WebDataset shards: `shards/shard_*.tar` — 50,000 shards, 3.25 TB total, uniform ~65 MB each
  - Manifest: `shards/manifest.jsonl` — 50,000 entries × 10,000 samples/shard = 500M samples (= paper's full LLM budget)
  - **Format inside each tar: `{uuid}.json` containing `{"text": "<raw document>"}`. Untokenized, not packed.** Tokenization happens on the fly in `vla_foundry/data/pipelines/text_untokenized.py` with `padding="max_length", truncation=True, max_length=seq_len+1`.
  - Intended tokenizer: `HuggingFaceTB/SmolVLM2-256M-Video-Instruct` (vocab 49,280) — pass as `--data.tokenizer`. Set `--data.type text_untokenized` (not `text`).
  - Token budget at seq_len=2048: ~0.62 T tokens with SmolVLM2 (avg ~1,175 tok/doc; paper's "1T tokens" assumes a 2048 tok/doc ceiling).
  - Raw HF parquets (source, kept for re-conversion): `raw/` — 27,937 parquet files, 6.8 TB. Safe to delete once shards are trusted.
  - Build script: `/e/project1/profound/thoeppe/scripts/data/01_dclm.sh` (`download` phase on login node, `convert` phase on compute).
  - Verifier: `/e/project1/profound/thoeppe/scripts/data/check_dclm_shards.py` — all 50K shards pass deep check (members=10000, no JSON errors, no empty docs).

### VLM Training Data

- **PixMo-Cap**: `/e/scratch/scifi/nadimpalli2/datasets/pixmo-cap-vla-foundry/`
  - Format: WebDataset shards (`shards/`) with a `manifest.jsonl` index
- **DataCompDR-200M**: `/e/data1/metagenome/nadimpalli2/datacompdr-200m/`
  - Current layout: WebDataset tar shards under `images_wds/task_*/data_*.tar`
  - Full manifest: `/e/data1/metagenome/nadimpalli2/datacompdr-200m/manifest.jsonl`
  - Training manifest with validation holdout removed: `/e/data1/metagenome/nadimpalli2/datacompdr-200m/manifest_train_minus_val128shards.jsonl`
  - Validation manifest (128 held-out shards, 1,263,154 manifest samples; intended for ~1M actual validation samples on 8 nodes with 4 workers/GPU): `/e/data1/metagenome/nadimpalli2/datacompdr-200m/manifest_val128shards.jsonl`
  - Train/validation split summary: `/e/data1/metagenome/nadimpalli2/datacompdr-200m/manifest_train_val128shards_summary.txt`
  - Manifest summary: `/e/data1/metagenome/nadimpalli2/datacompdr-200m/manifest_summary.json`
  - Manifest contents: 20,979 readable shards, 206,716,142 supported image/text pairs
  - Excluded corrupt shards: 21 entries listed in `/e/data1/metagenome/nadimpalli2/datacompdr-200m/manifest_failed_shards.txt`
  - Sample format inside shards: `<sample_id>.<image_ext>`, `<sample_id>.txt`, `<sample_id>.json`; supported training image extensions are `jpg`, `jpeg`, `png`, and `webp`

### VLM Evaluation Data

- **COCO Captions (val2017)**: `/e/scratch/scifi/nadimpalli2/datasets/coco/`
  - Images (5,000 JPEGs): `val2017/`
  - Captions annotations: `annotations/captions_val2017.json`
  - Metrics: BLEU-1/2/3/4, METEOR, ROUGE-L, CIDEr, SPICE

## Checkpoints

Downloaded Hugging Face checkpoints are cached under `/e/scratch/scifi/nadimpalli2/hf_cache/`.

- **Foundry-LLM-1.2B-1T**: `/e/scratch/scifi/nadimpalli2/hf_cache/hub/models--TRI-ML--Foundry-LLM-1.2B-1T/snapshots/1e71bc75d3a37791568f13d35af3ba578e04a24f/`
  - VLM language-backbone checkpoint: `/e/scratch/scifi/nadimpalli2/hf_cache/hub/models--TRI-ML--Foundry-LLM-1.2B-1T/snapshots/1e71bc75d3a37791568f13d35af3ba578e04a24f/checkpoints/checkpoint_56.pt`

## VLM Training

### Cluster Environment

Training uses the 2026 PyTorch stack and an existing virtual environment:

```bash
module load Stages/2026 GCCcore/14.3.0 PyTorch/2.9.1 torchvision/0.24.1
source /e/scratch/scifi/nadimpalli2/venvs/worldchanger_vlm_py313_torch291/bin/activate
```

The Slurm script exports:

- `HF_HOME=/e/scratch/scifi/nadimpalli2/hf_cache`
- `HF_HUB_OFFLINE=1`
- `TRANSFORMERS_OFFLINE=1`

### 16-Node Smoke Test

Fastest measured smoke configuration is 16 nodes, 64 GPUs total, global batch 512, per-GPU batch 8, no gradient accumulation, and `torchcompile=True`.

```bash
sbatch scripts/slurm/vlm_llm1t_smoke.sbatch
```

The script defaults to:

- `--nodes=16`
- `PER_GPU_BATCH_SIZE=8`
- `GLOBAL_BATCH_SIZE=512`
- `TORCHCOMPILE=True`
- `DATASET_MANIFEST=/e/data1/metagenome/nadimpalli2/datacompdr-200m/manifest_train_minus_val128shards.jsonl`
- `VAL_DATASET_MANIFEST=/e/data1/metagenome/nadimpalli2/datacompdr-200m/manifest_val128shards.jsonl`

### Full 200M-Sample Run

Use the same script with overrides:

```bash
sbatch \
  --time=2-00:00:00 \
  --export=ALL,TOTAL_TRAIN_SAMPLES=200000000,NUM_CHECKPOINTS=40,TOTAL_VAL_SAMPLES=1000000,VAL_EVERY_N_CHECKPOINTS=1 \
  scripts/slurm/vlm_llm1t_smoke.sbatch
```

This trains 390,625 optimizer steps at global batch 512 and saves approximately every 5M training samples.

## VLM Evaluation

### Virtual Environment

Location: `/e/scratch/scifi/nadimpalli2/venvs/vlm_eval`

Activate with:
```bash
module load Stages/2025 GCCcore/.13.3.0 Python/3.12.3
source /e/scratch/scifi/nadimpalli2/venvs/vlm_eval/bin/activate
```

### Running COCO Captions Eval

```bash
python -m vla_foundry.eval.eval_coco_captions <experiment_dir> \
    --coco-dir /e/scratch/scifi/nadimpalli2/datasets/coco \
    --device cuda:0
```

Options:
- `--checkpoint <filename>` — specific checkpoint (default: latest)
- `--max-images N` — limit images for quick sanity checks
- `--output-dir <path>` — where to save results (default: `<experiment_dir>/eval_coco/`)

Script: `vla_foundry/eval/eval_coco_captions.py`

### LM-Eval Harness

The repo has an `lm-eval` dependency group for EleutherAI's lm-evaluation-harness.
Use the dedicated venv for any lm-eval work:

```bash
source /e/scratch/scifi/nadimpalli2/venvs/lm_eval/bin/activate
```

Path: `/e/scratch/scifi/nadimpalli2/venvs/lm_eval`

Quick check:
```bash
lm_eval --help
```
