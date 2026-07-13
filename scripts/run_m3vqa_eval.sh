#!/bin/bash
#SBATCH --job-name=lvr_m3vqa
#SBATCH --output=/mnt/data/lannth/mLAnR/logs/m3vqa_%j.log
#SBATCH --error=/mnt/data/lannth/mLAnR/logs/m3vqa_%j.err
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

set -e

CONDA_ENV=lvr
MODEL_DIR=/mnt/data/lannth/mLAnR/checkpoints/LVR-7B
QUESTIONS=/mnt/data/lannth/mLAnR/M3-VQA/quesions.jsonl
IMAGE_DIR=/mnt/data/lannth/mLAnR/M3-VQA/images
OUTPUT_DIR=/mnt/data/lannth/mLAnR/results/m3vqa
LVR_CODE=/mnt/data/lannth/mLAnR/lvr

# ------------------------------------------------------------------
# Step 1: Install missing pip packages (idempotent)
# ------------------------------------------------------------------
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $CONDA_ENV

echo "=== Installing dependencies ==="
pip install transformers==4.51.3 huggingface-hub==0.30.2 accelerate==1.6.0 \
    safetensors==0.5.3 tokenizers==0.21.1 datasets tqdm pillow \
    qwen-vl-utils -q

# flash-attn requires a matching CUDA/torch build; skip if it fails
pip install flash-attn --no-build-isolation -q 2>/dev/null || \
    echo "WARNING: flash-attn install failed; falling back to eager attention"

# ------------------------------------------------------------------
# Step 2: Download model (idempotent — skips if already present)
# ------------------------------------------------------------------
echo "=== Downloading LVR-7B from HuggingFace ==="
mkdir -p $MODEL_DIR
huggingface-cli download vincentleebang/LVR-7B \
    --local-dir $MODEL_DIR \
    --local-dir-use-symlinks False

# ------------------------------------------------------------------
# Step 3: Run M3-VQA evaluation
# ------------------------------------------------------------------
echo "=== Running M3-VQA evaluation ==="
mkdir -p /mnt/data/lannth/mLAnR/logs

cd $LVR_CODE
python evaluation/evaluate_m3vqa.py \
    --model_path $MODEL_DIR \
    --questions $QUESTIONS \
    --image_dir $IMAGE_DIR \
    --output_dir $OUTPUT_DIR \
    --steps 4 8 16 \
    --decoding_strategy steps

echo "=== Done ==="
