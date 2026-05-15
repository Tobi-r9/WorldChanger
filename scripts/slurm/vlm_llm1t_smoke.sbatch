#!/bin/bash
#SBATCH --job-name=vlm-llm1t-smoke
#SBATCH --partition=booster
#SBATCH --nodes=16
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd /e/project1/scifi/Kalyan/WorldChanger

module purge
module load Stages/2026 GCCcore/14.3.0 PyTorch/2.9.1 torchvision/0.24.1

source /e/scratch/scifi/nadimpalli2/venvs/worldchanger_vlm_py313_torch291/bin/activate

export HF_HOME=/e/scratch/scifi/nadimpalli2/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=/e/project1/scifi/Kalyan/WorldChanger:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1
export WANDB_MODE=disabled

GPUS_PER_NODE=4
DATASET_MANIFEST=${DATASET_MANIFEST:-/e/data1/metagenome/nadimpalli2/datacompdr-200m/manifest_train_minus_val128shards.jsonl}
VAL_DATASET_MANIFEST=${VAL_DATASET_MANIFEST:-/e/data1/metagenome/nadimpalli2/datacompdr-200m/manifest_val128shards.jsonl}
DATA_NUM_WORKERS=${DATA_NUM_WORKERS:-4}
DATA_PREFETCH_FACTOR=${DATA_PREFETCH_FACTOR:-2}
PER_GPU_BATCH_SIZE=${PER_GPU_BATCH_SIZE:-8}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-512}
TOTAL_TRAIN_SAMPLES=${TOTAL_TRAIN_SAMPLES:-65536}
NUM_CHECKPOINTS=${NUM_CHECKPOINTS:-1}
TOTAL_VAL_SAMPLES=${TOTAL_VAL_SAMPLES:-0}
VAL_EVERY_N_CHECKPOINTS=${VAL_EVERY_N_CHECKPOINTS:-1}
PRECISION=${PRECISION:-amp_bfloat16}
TORCHCOMPILE=${TORCHCOMPILE:-True}
EXTRA_ARGS=${EXTRA_ARGS:-}
export DATASET_MANIFEST VAL_DATASET_MANIFEST DATA_NUM_WORKERS DATA_PREFETCH_FACTOR PER_GPU_BATCH_SIZE GLOBAL_BATCH_SIZE TOTAL_TRAIN_SAMPLES NUM_CHECKPOINTS TOTAL_VAL_SAMPLES VAL_EVERY_N_CHECKPOINTS PRECISION TORCHCOMPILE EXTRA_ARGS
MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

echo "job_id=${SLURM_JOB_ID}"
echo "nodes=${SLURM_NNODES}"
echo "node_list=${SLURM_JOB_NODELIST}"
echo "master=${MASTER_ADDR}:${MASTER_PORT}"
echo "venv=/e/scratch/scifi/nadimpalli2/venvs/worldchanger_vlm_py313_torch291"
echo "dataset_manifest=${DATASET_MANIFEST}"
echo "val_dataset_manifest=${VAL_DATASET_MANIFEST}"
echo "data_num_workers=${DATA_NUM_WORKERS}"
echo "data_prefetch_factor=${DATA_PREFETCH_FACTOR}"
echo "per_gpu_batch_size=${PER_GPU_BATCH_SIZE}"
echo "global_batch_size=${GLOBAL_BATCH_SIZE}"
echo "total_train_samples=${TOTAL_TRAIN_SAMPLES}"
echo "num_checkpoints=${NUM_CHECKPOINTS}"
echo "total_val_samples=${TOTAL_VAL_SAMPLES}"
echo "val_every_n_checkpoints=${VAL_EVERY_N_CHECKPOINTS}"
echo "precision=${PRECISION}"
echo "torchcompile=${TORCHCOMPILE}"
echo "extra_args=${EXTRA_ARGS}"
echo "started_at=$(date --iso-8601=seconds)"

srun --nodes="${SLURM_NNODES}" --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 \
  bash -c '
    VAL_ARGS=()
    if [[ "${TOTAL_VAL_SAMPLES}" -gt 0 ]]; then
      VAL_ARGS=(
        --total_val_samples "${TOTAL_VAL_SAMPLES}"
        --val_every_n_checkpoints "${VAL_EVERY_N_CHECKPOINTS}"
        --data.val_dataset_manifest "[\"${VAL_DATASET_MANIFEST}\"]"
        --data.val_dataset_weighting "[1.0]"
      )
    fi

    python -m torch.distributed.run \
      --nnodes="${SLURM_NNODES}" \
      --nproc_per_node=4 \
      --node_rank="${SLURM_NODEID}" \
      --rdzv_backend=c10d \
      --rdzv_endpoint="'"${MASTER_ADDR}:${MASTER_PORT}"'" \
      vla_foundry/main.py \
        --name "vlm_llm1t_smoke_${SLURM_NNODES}node_gbs${GLOBAL_BATCH_SIZE}_${SLURM_JOB_ID}" \
        --save_path /e/scratch/scifi/nadimpalli2/runs/vlm_smoke \
        --wandb False \
        --db_logging False \
        --log_every_n_steps 1 \
        --model "include vla_foundry/config_presets/models/smolvlm_load_llm.yaml" \
        --model.transformer.resume_from_checkpoint /e/scratch/scifi/nadimpalli2/hf_cache/hub/models--TRI-ML--Foundry-LLM-1.2B-1T/snapshots/1e71bc75d3a37791568f13d35af3ba578e04a24f/checkpoints/checkpoint_56.pt \
        --model.transformer.resume_weights_only True \
        --model.transformer.norm_type lp_layer_norm \
        --model.transformer.qk_norm True \
        --distributed.fsdp True \
        --data.type image_caption \
        --data.processor HuggingFaceTB/SmolVLM2-256M-Video-Instruct \
        --data.processor_kwargs "{\"size\":{\"longest_edge\":224},\"max_image_size\":{\"longest_edge\":224}}" \
        --data.dataset_manifest "[\"${DATASET_MANIFEST}\"]" \
        --data.dataset_modality "[\"image_caption\"]" \
        --data.dataset_weighting "[1.0]" \
        --data.num_workers "${DATA_NUM_WORKERS}" \
        --data.prefetch_factor "${DATA_PREFETCH_FACTOR}" \
        --data.seq_len 2048 \
        --data.img_num_tokens 64 \
        --data.image_size 224 \
        --model.vit.img_size 224 \
        --model.vit.patch_size 14 \
        --model.vit.projector_pixel_shuffle_factor 2 \
        --total_train_samples "${TOTAL_TRAIN_SAMPLES}" \
        --num_checkpoints "${NUM_CHECKPOINTS}" \
        --hparams.per_gpu_batch_size "${PER_GPU_BATCH_SIZE}" \
        --hparams.global_batch_size "${GLOBAL_BATCH_SIZE}" \
        --hparams.precision "${PRECISION}" \
        --hparams.torchcompile "${TORCHCOMPILE}" \
        "${VAL_ARGS[@]}" \
        ${EXTRA_ARGS}
  '

echo "finished_at=$(date --iso-8601=seconds)"
