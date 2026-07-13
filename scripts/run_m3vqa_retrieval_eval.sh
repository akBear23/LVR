#!/bin/bash
#SBATCH --job-name=lvr_m3vqa_retrieval
#SBATCH --output=/mnt/data/lannth/mLAnR/logs/m3vqa_retrieval_%j.log
#SBATCH --error=/mnt/data/lannth/mLAnR/logs/m3vqa_retrieval_%j.err
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00

set -e

CONDA_ENV=lvr
MODEL_DIR=/mnt/data/lannth/mLAnR/checkpoints/LVR-7B
KB_PATH=/mnt/data/lannth/mLAnR/M3-VQA/encyclopedic_kb_wiki.json
QUESTIONS=/mnt/data/lannth/mLAnR/M3-VQA/quesions.jsonl
IMAGE_DIR=/mnt/data/lannth/mLAnR/M3-VQA/images
OUTPUT_DIR=/mnt/data/lannth/mLAnR/results/m3vqa_retrieval
LVR_CODE=/mnt/data/lannth/mLAnR/lvr

# Setup
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $CONDA_ENV
mkdir -p /mnt/data/lannth/mLAnR/logs

cd $LVR_CODE

# Run evaluation with different retrieval modes
echo "=== Evaluation 1: Oracle Retrieval (ground-truth upper bound) ==="
python evaluation/evaluate_m3vqa_retrieval.py \
    --model_path $MODEL_DIR \
    --kb_path $KB_PATH \
    --questions $QUESTIONS \
    --image_dir $IMAGE_DIR \
    --output_dir $OUTPUT_DIR \
    --retrieval_mode oracle \
    --steps 4 8 16

echo "=== Evaluation 2: Entity-based Retrieval (entity name lookup) ==="
python evaluation/evaluate_m3vqa_retrieval.py \
    --model_path $MODEL_DIR \
    --kb_path $KB_PATH \
    --questions $QUESTIONS \
    --image_dir $IMAGE_DIR \
    --output_dir $OUTPUT_DIR \
    --retrieval_mode entity \
    --steps 4 8 16

echo "=== Evaluation 3: Hybrid Retrieval (oracle → fallback to entity) ==="
python evaluation/evaluate_m3vqa_retrieval.py \
    --model_path $MODEL_DIR \
    --kb_path $KB_PATH \
    --questions $QUESTIONS \
    --image_dir $IMAGE_DIR \
    --output_dir $OUTPUT_DIR \
    --retrieval_mode hybrid \
    --steps 4 8 16

echo "=== Done ==="
