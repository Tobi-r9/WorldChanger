# WorldChanger (VLA Foundry)

## Datasets

All datasets are stored on scratch at `/e/scratch/scifi/nadimpalli2/datasets/`.

### VLM Training Data

- **PixMo-Cap**: `/e/scratch/scifi/nadimpalli2/datasets/pixmo-cap-vla-foundry/`
  - Format: WebDataset shards (`shards/`) with a `manifest.jsonl` index
- **DataCompDR-200M**: `/e/data1/metagenome/nadimpalli2/datacompdr-200m/`
  - Current layout: WebDataset tar shards under `images_wds/task_*/data_*.tar`
  - Training manifest: `/e/data1/metagenome/nadimpalli2/datacompdr-200m/manifest.jsonl`
  - Manifest summary: `/e/data1/metagenome/nadimpalli2/datacompdr-200m/manifest_summary.json`
  - Manifest contents: 20,979 readable shards, 206,716,142 supported image/text pairs
  - Excluded corrupt shards: 21 entries listed in `/e/data1/metagenome/nadimpalli2/datacompdr-200m/manifest_failed_shards.txt`
  - Sample format inside shards: `<sample_id>.<image_ext>`, `<sample_id>.txt`, `<sample_id>.json`; supported training image extensions are `jpg`, `jpeg`, `png`, and `webp`

### VLM Evaluation Data

- **COCO Captions (val2017)**: `/e/scratch/scifi/nadimpalli2/datasets/coco/`
  - Images (5,000 JPEGs): `val2017/`
  - Captions annotations: `annotations/captions_val2017.json`
  - Metrics: BLEU-1/2/3/4, METEOR, ROUGE-L, CIDEr, SPICE

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
