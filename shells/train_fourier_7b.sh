#!/bin/bash

# LLaVA-FA Training Script for 7B Model
# Example training configuration with Fourier approximation

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Model and data paths (modify these according to your setup)
MODEL_PATH="liuhaotian/llava-v1.5-7b"
TEACHER_PATH=""  # Optional: path to teacher model for distillation
DATA_PATH="path/to/training/data.json"  # Your training data
IMAGE_FOLDER="path/to/images"           # Your image folder
OUTPUT_DIR="./output/llava-fa-7b-$(date +%Y%m%d_%H%M%S)"

# Training configuration
NUM_EPOCHS=3
BATCH_SIZE=4
LEARNING_RATE=2e-5
FOURIER_COMPRESSION=0.05

# Create output directory
mkdir -p $OUTPUT_DIR

echo "Starting LLaVA-FA training..."
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"
echo "Fourier compression ratio: $FOURIER_COMPRESSION"

# Launch training
python scripts/train_fourier.py \
    --model_name_or_path $MODEL_PATH \
    --teacher_model_path $TEACHER_PATH \
    --version v1 \
    --data_path $DATA_PATH \
    --image_folder $IMAGE_FOLDER \
    --vision_tower openai/clip-vit-large-patch14 \
    --mm_projector_type linear \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2400 \
    --save_total_limit 1 \
    --learning_rate $LEARNING_RATE \
    --weight_decay 0. \
    --warmup_steps 500 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name "llava-fa-7b-fourier-${FOURIER_COMPRESSION}" \
    --enable_fourier \
    --fourier_basis_type dct \
    --fourier_compression_ratio $FOURIER_COMPRESSION \
    --fourier_target_modules q_proj k_proj v_proj o_proj up_proj down_proj gate_proj \
    --use_frequency_scheduling \
    --frequency_scheduler_type linear \
    --scheduler_min_components 0.1 \
    --scheduler_max_components 0.8 \
    --distillation_alpha 0.5 \
    --distillation_temperature 4.0 \
    --hidden_distillation_weight 0.5

echo "Training completed! Model saved to: $OUTPUT_DIR"
