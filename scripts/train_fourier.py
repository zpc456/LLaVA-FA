#!/usr/bin/env python3
"""
Training script for LLaVA-FA (Fourier Approximation)
Supports knowledge distillation from teacher models like LLaVA-MoD
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import wandb

# Add LLaVA-FA to path
sys.path.append(str(Path(__file__).parent.parent))

from llavafd.model import LlavaLlamaForCausalLM
from llavafd.train.fourier_trainer import (
    FourierTrainer, FourierTrainingArguments, create_fourier_trainer,
    DEFAULT_FOURIER_TRAINING_CONFIG
)
from llavafd.fourier import (
    inject_fourier_adapters, FourierAdapterConfig, BasisType
)
from llavafd.data import make_supervised_data_module
from llavafd.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llavafd.conversation import conv_templates, SeparatorStyle
from llavafd.mm_utils import tokenizer_image_token
from llavafd.utils import smart_tokenizer_and_embedding_resize

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="LLaVA-FA Training")
    
    # Model arguments
    parser.add_argument("--model_name_or_path", type=str, required=True,
                       help="Path to pretrained student model")
    parser.add_argument("--teacher_model_path", type=str, default=None,
                       help="Path to teacher model for distillation")
    parser.add_argument("--version", type=str, default="v1",
                       help="Model version")
    parser.add_argument("--vision_tower", type=str, default=None,
                       help="Vision tower model")
    parser.add_argument("--mm_projector_type", type=str, default="linear",
                       help="Type of multimodal projector")
    parser.add_argument("--pretrain_mm_mlp_adapter", type=str, default=None,
                       help="Path to pretrained multimodal MLP adapter")
    
    # Data arguments  
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to training data")
    parser.add_argument("--eval_data_path", type=str, default=None,
                       help="Path to evaluation data")
    parser.add_argument("--image_folder", type=str, default="",
                       help="Folder containing images")
    parser.add_argument("--image_aspect_ratio", type=str, default="square",
                       help="Image aspect ratio")
    parser.add_argument("--lazy_preprocess", action="store_true",
                       help="Use lazy preprocessing")
    
    # Fourier approximation arguments
    parser.add_argument("--enable_fourier", action="store_true",
                       help="Enable Fourier approximation")
    parser.add_argument("--fourier_basis_type", type=str, default="dct",
                       choices=["dct", "dst", "fourier", "mixed"],
                       help="Type of frequency basis")
    parser.add_argument("--fourier_compression_ratio", type=float, default=0.05,
                       help="Overall compression ratio for Fourier approximation")
    parser.add_argument("--fourier_target_modules", type=str, nargs="+",
                       default=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
                       help="Target modules for Fourier injection")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4,
                       help="Training batch size per device")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4,
                       help="Evaluation batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                       help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=500,
                       help="Warmup steps")
    parser.add_argument("--save_steps", type=int, default=500,
                       help="Save checkpoint every N steps")
    parser.add_argument("--eval_steps", type=int, default=500,
                       help="Evaluation steps")
    parser.add_argument("--logging_steps", type=int, default=50,
                       help="Logging steps")
    parser.add_argument("--max_seq_length", type=int, default=2048,
                       help="Maximum sequence length")
    
    # Knowledge distillation arguments
    parser.add_argument("--distillation_alpha", type=float, default=0.5,
                       help="Distillation loss weight")
    parser.add_argument("--distillation_temperature", type=float, default=4.0,
                       help="Distillation temperature")
    parser.add_argument("--hidden_distillation_weight", type=float, default=0.5,
                       help="Hidden state distillation weight")
    
    # Frequency scheduling arguments
    parser.add_argument("--use_frequency_scheduling", action="store_true",
                       help="Use progressive frequency scheduling")
    parser.add_argument("--frequency_scheduler_type", type=str, default="linear",
                       choices=["linear", "exponential", "cosine", "step", "adaptive"],
                       help="Frequency scheduler type")
    parser.add_argument("--scheduler_min_components", type=float, default=0.1,
                       help="Minimum frequency components ratio")
    parser.add_argument("--scheduler_max_components", type=float, default=0.8,
                       help="Maximum frequency components ratio")
    
    # Other arguments
    parser.add_argument("--freeze_mm_mlp_adapter", action="store_true",
                       help="Freeze multimodal MLP adapter")
    parser.add_argument("--group_by_modality_length", action="store_true",
                       help="Group samples by modality length")
    parser.add_argument("--bits", type=int, default=16,
                       help="Quantization bits")
    parser.add_argument("--mm_use_im_start_end", action="store_true",
                       help="Use image start/end tokens")
    parser.add_argument("--mm_use_im_patch_token", action="store_true",
                       help="Use image patch tokens")
    parser.add_argument("--image_grid_pinpoints", type=str, default=None,
                       help="Image grid pinpoints")
    parser.add_argument("--report_to", type=str, default="wandb",
                       help="Experiment tracking platform")
    parser.add_argument("--run_name", type=str, default=None,
                       help="Run name for experiment tracking")
    
    return parser.parse_args()


def load_teacher_model(teacher_path: str, device_map="auto"):
    """Load teacher model for knowledge distillation"""
    logger.info(f"Loading teacher model from {teacher_path}")
    
    # Load teacher model (could be LLaVA-MoD or other LLaVA variant)
    teacher_model = LlavaLlamaForCausalLM.from_pretrained(
        teacher_path,
        torch_dtype=torch.float16,
        device_map=device_map,
        low_cpu_mem_usage=True
    )
    
    teacher_model.eval()
    logger.info("Teacher model loaded successfully")
    
    return teacher_model


def setup_fourier_approximation(model, args):
    """Setup Fourier approximation for the model"""
    logger.info("Setting up Fourier approximation...")
    
    # Create Fourier adapter configuration
    fourier_config = FourierAdapterConfig(
        target_modules=[f".*\\.{module}$" for module in args.fourier_target_modules],
        basis_type=BasisType(args.fourier_basis_type),
        compression_ratios={
            module: args.fourier_compression_ratio 
            for module in args.fourier_target_modules
        },
        scaling=1.0,
        dropout=0.1
    )
    
    # Inject Fourier adapters
    model, injector = inject_fourier_adapters(model, fourier_config, verbose=True)
    
    # Freeze original model parameters (only train Fourier adapters)
    injector.freeze_original_parameters(model)
    
    logger.info("Fourier approximation setup complete")
    return model, injector


def main():
    args = parse_args()
    
    # Set up logging
    transformers.utils.logging.set_verbosity_info()
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    
    # Initialize wandb if specified
    if args.report_to == "wandb":
        wandb.init(
            project="llava-fa",
            name=args.run_name or f"fourier-{args.fourier_basis_type}-{args.fourier_compression_ratio}",
            config=vars(args)
        )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=args.max_seq_length,
        padding_side="right",
        use_fast=False,
    )
    
    # Add special tokens
    tokenizer.pad_token = tokenizer.unk_token
    
    # Load student model
    logger.info(f"Loading student model from {args.model_name_or_path}")
    model = LlavaLlamaForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float16 if args.bits == 16 else torch.float32,
        low_cpu_mem_usage=True,
    )
    
    # Setup vision components
    if args.vision_tower is not None:
        model.get_vision_tower().load_model()
    
    # Load multimodal projector if specified
    if args.pretrain_mm_mlp_adapter is not None:
        mm_projector_weights = torch.load(args.pretrain_mm_mlp_adapter, map_location='cpu')
        mm_projector_weights = {k.split('.')[-1]: v for k, v in mm_projector_weights.items()}
        model.load_state_dict(mm_projector_weights, strict=False)
    
    # Setup Fourier approximation if enabled
    fourier_injector = None
    if args.enable_fourier:
        model, fourier_injector = setup_fourier_approximation(model, args)
    
    # Load teacher model for distillation
    teacher_model = None
    if args.teacher_model_path:
        teacher_model = load_teacher_model(args.teacher_model_path)
    
    # Update model config
    model.config.use_mm_proj = True
    model.config.mm_projector_type = args.mm_projector_type
    model.config.mm_use_im_start_end = args.mm_use_im_start_end
    model.config.mm_use_im_patch_token = args.mm_use_im_patch_token
    model.config.image_aspect_ratio = args.image_aspect_ratio
    
    if args.image_grid_pinpoints:
        model.config.image_grid_pinpoints = eval(args.image_grid_pinpoints)
    
    # Resize embeddings if needed
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict={"pad_token": "[PAD]"},
        tokenizer=tokenizer,
        model=model,
    )
    
    # Prepare data
    data_module = make_supervised_data_module(
        tokenizer=tokenizer,
        data_args={
            'data_path': args.data_path,
            'image_folder': args.image_folder,
            'lazy_preprocess': args.lazy_preprocess,
            'is_multimodal': True,
            'image_aspect_ratio': args.image_aspect_ratio,
            'image_grid_pinpoints': args.image_grid_pinpoints,
        }
    )
    
    # Setup training arguments
    training_args = FourierTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        evaluation_strategy="steps" if args.eval_data_path else "no",
        save_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss" if args.eval_data_path else None,
        greater_is_better=False,
        report_to=args.report_to,
        run_name=args.run_name,
        remove_unused_columns=False,
        dataloader_drop_last=True,
        group_by_length=args.group_by_modality_length,
        bf16=False,
        fp16=args.bits == 16,
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
        
        # Fourier-specific arguments
        distillation_alpha=args.distillation_alpha,
        distillation_temperature=args.distillation_temperature,
        hidden_distillation_weight=args.hidden_distillation_weight,
        use_frequency_scheduling=args.use_frequency_scheduling,
        frequency_scheduler_type=args.frequency_scheduler_type,
        scheduler_min_components=args.scheduler_min_components,
        scheduler_max_components=args.scheduler_max_components,
        log_frequency_stats=True,
    )
    
    # Create trainer
    trainer = create_fourier_trainer(
        model=model,
        teacher_model=teacher_model,
        train_dataset=data_module['train_dataset'],
        eval_dataset=data_module.get('eval_dataset'),
        training_args=training_args,
        data_collator=data_module['data_collator'],
        tokenizer=tokenizer,
    )
    
    # Start training
    logger.info("Starting training...")
    trainer.train()
    
    # Save the final model
    logger.info("Saving final model...")
    trainer.save_model()
    
    # Save Fourier adapters separately if used
    if fourier_injector:
        fourier_save_path = os.path.join(args.output_dir, "fourier_adapters.pt")
        fourier_injector.save_adapters(fourier_save_path)
        logger.info(f"Fourier adapters saved to {fourier_save_path}")
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
