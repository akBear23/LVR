#!/bin/bash
# Stage-2 GRPO+Retrieval training on M3-VQA.
# Generates data first (if not done), then launches training.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LVR_ROOT="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="$LVR_ROOT:$LVR_ROOT/src:$PYTHONPATH"

CONDA_ENV="lvr"
PYTHON="/home/lannth/miniconda3/envs/${CONDA_ENV}/bin/python3.11"
DEEPSPEED="/home/lannth/miniconda3/envs/${CONDA_ENV}/bin/deepspeed"

# ── Config ────────────────────────────────────────────────────────────────────
STAGE1_CHKPT="${1:-/mnt/data/lannth/mLAnR/checkpoints/LVR-7B}"
MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"

KB_PATH="/mnt/data/lannth/mLAnR/M3-VQA/encyclopedic_kb_wiki.json"
QUESTIONS="/mnt/data/lannth/mLAnR/M3-VQA/quesions.jsonl"
IMAGE_DIR="/mnt/data/lannth/mLAnR/M3-VQA/images"
TRAIN_JSON="$SCRIPT_DIR/m3vqa_train.json"
EVAL_JSON="$SCRIPT_DIR/m3vqa_eval.json"
CLIP_CACHE="$SCRIPT_DIR/.cache"
OUTPUT_DIR="$SCRIPT_DIR/checkpoints"

LVR_STEPS=8
TOP_K=3
NUM_GENERATIONS=4
LR=1e-6
TEMP=0.9

# ── Step 1: Prepare data (idempotent) ─────────────────────────────────────────
if [ ! -f "$TRAIN_JSON" ]; then
    echo "=== Preparing M3-VQA GRPO data ==="
    "$PYTHON" "$SCRIPT_DIR/prepare_data.py" \
        --questions   "$QUESTIONS" \
        --image_dir   "$IMAGE_DIR" \
        --output_train "$TRAIN_JSON" \
        --output_eval  "$EVAL_JSON"
fi

# ── Step 2: Train ─────────────────────────────────────────────────────────────
echo "=== Starting GRPO+Retrieval training ==="
"$DEEPSPEED" "$SCRIPT_DIR/launch.py" \
    --deepspeed "$LVR_ROOT/scripts/zero2.json" \
    --checkpoint_name "$STAGE1_CHKPT" \
    --model_id "$MODEL_NAME" \
    --data_path "$TRAIN_JSON" \
    --image_folder "$IMAGE_DIR" \
    --kb_path "$KB_PATH" \
    --clip_cache_dir "$CLIP_CACHE" \
    --retrieval_top_k $TOP_K \
    --retrieval_reward_weight 0.2 \
    --freeze_vision_tower True \
    --freeze_merger True \
    --freeze_llm False \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir "$OUTPUT_DIR" \
    --temperature $TEMP \
    --num_train_epochs 2 \
    --num_generations $NUM_GENERATIONS \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --max_completion_length 512 \
    --max_prompt_length 4096 \
    --image_min_pixels $((128 * 28 * 28)) \
    --image_max_pixels $((1280 * 28 * 28)) \
    --learning_rate $LR \
    --remove_unused_columns False \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --report_to wandb \
    --save_strategy steps \
    --save_steps 100 \
    --save_total_limit 10 \
    --dataloader_num_workers 4 \
    --decoding_strategy steps \
    --lvr_steps $LVR_STEPS \
    --run_name "grpo_retrieval_lvr${LVR_STEPS}_k${TOP_K}_g${NUM_GENERATIONS}"
