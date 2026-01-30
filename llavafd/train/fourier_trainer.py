"""
Fourier Knowledge Distillation Trainer
Specialized trainer for LLaVA-FA with Fourier approximation and distillation
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer, TrainerCallback, TrainingArguments
from transformers.trainer_utils import has_length
from typing import Dict, List, Optional, Union, Any
import logging
import numpy as np
import wandb
from dataclasses import dataclass
from ..fourier import (
    FourierAdapter, FrequencyScheduler, FrequencySchedulerManager, 
    FourierOptimizer, create_frequency_scheduler, SCHEDULER_PRESETS
)

logger = logging.getLogger(__name__)


@dataclass
class FourierTrainingArguments(TrainingArguments):
    """
    Extended training arguments for Fourier approximation
    """
    # Knowledge distillation parameters
    distillation_alpha: float = 0.5
    distillation_temperature: float = 4.0
    hidden_distillation_weight: float = 0.5
    attention_distillation_weight: float = 0.0
    
    # Fourier-specific parameters
    frequency_regularization: float = 1e-4
    low_freq_weight: float = 1.0
    high_freq_weight: float = 2.0
    frequency_dropout_rate: float = 0.1
    
    # Frequency scheduling
    use_frequency_scheduling: bool = True
    frequency_scheduler_type: str = "linear"
    scheduler_warmup_steps: Optional[int] = None
    scheduler_min_components: float = 0.1
    scheduler_max_components: float = 1.0
    
    # Compression parameters
    enable_sparsification: bool = False
    target_sparsity: float = 0.5
    sparsification_schedule: str = "linear"
    
    # Logging and visualization
    log_frequency_stats: bool = True
    frequency_stats_steps: int = 100


class KnowledgeDistillationLoss(nn.Module):
    """
    Multi-component knowledge distillation loss
    Combines logit distillation, hidden state alignment, and attention transfer
    """
    
    def __init__(
        self,
        alpha: float = 0.5,
        temperature: float = 4.0,
        hidden_weight: float = 0.5,
        attention_weight: float = 0.0
    ):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.hidden_weight = hidden_weight
        self.attention_weight = attention_weight
    
    def forward(
        self,
        student_outputs,
        teacher_outputs,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute knowledge distillation loss
        
        Args:
            student_outputs: Student model outputs (logits and hidden states)
            teacher_outputs: Teacher model outputs (logits and hidden states) 
            labels: Ground truth labels for task loss
            
        Returns:
            Dictionary of loss components
        """
        losses = {}
        
        # Logit distillation loss
        if hasattr(student_outputs, 'logits') and hasattr(teacher_outputs, 'logits'):
            student_logits = student_outputs.logits
            teacher_logits = teacher_outputs.logits
            
            # Ensure same shape
            if student_logits.shape != teacher_logits.shape:
                # Handle shape mismatch (e.g., different sequence lengths)
                min_len = min(student_logits.shape[1], teacher_logits.shape[1])
                student_logits = student_logits[:, :min_len]
                teacher_logits = teacher_logits[:, :min_len]
            
            kl_loss = F.kl_div(
                F.log_softmax(student_logits / self.temperature, dim=-1),
                F.softmax(teacher_logits / self.temperature, dim=-1),
                reduction='batchmean'
            ) * (self.temperature ** 2)
            
            losses['kl_loss'] = kl_loss
        
        # Hidden state distillation
        if (self.hidden_weight > 0 and 
            hasattr(student_outputs, 'hidden_states') and 
            hasattr(teacher_outputs, 'hidden_states')):
            
            student_hiddens = student_outputs.hidden_states[-1]  # Last layer
            teacher_hiddens = teacher_outputs.hidden_states[-1]
            
            # Handle dimension mismatch
            if student_hiddens.shape != teacher_hiddens.shape:
                # Project to same dimension if needed
                if student_hiddens.shape[-1] != teacher_hiddens.shape[-1]:
                    projection = nn.Linear(
                        teacher_hiddens.shape[-1], 
                        student_hiddens.shape[-1]
                    ).to(teacher_hiddens.device)
                    teacher_hiddens = projection(teacher_hiddens)
                
                # Handle sequence length mismatch
                min_len = min(student_hiddens.shape[1], teacher_hiddens.shape[1])
                student_hiddens = student_hiddens[:, :min_len]
                teacher_hiddens = teacher_hiddens[:, :min_len]
            
            hidden_loss = F.mse_loss(student_hiddens, teacher_hiddens)
            losses['hidden_loss'] = hidden_loss
        
        # Attention distillation (optional)
        if (self.attention_weight > 0 and 
            hasattr(student_outputs, 'attentions') and 
            hasattr(teacher_outputs, 'attentions')):
            
            # Average attention across layers and heads
            student_attn = torch.stack(student_outputs.attentions).mean(0).mean(1)
            teacher_attn = torch.stack(teacher_outputs.attentions).mean(0).mean(1)
            
            if student_attn.shape != teacher_attn.shape:
                min_len = min(student_attn.shape[1], teacher_attn.shape[1])
                student_attn = student_attn[:, :min_len, :min_len]
                teacher_attn = teacher_attn[:, :min_len, :min_len]
            
            attention_loss = F.mse_loss(student_attn, teacher_attn)
            losses['attention_loss'] = attention_loss
        
        # Task loss (if labels provided)
        if labels is not None and hasattr(student_outputs, 'logits'):
            task_loss = F.cross_entropy(
                student_outputs.logits.view(-1, student_outputs.logits.shape[-1]),
                labels.view(-1),
                ignore_index=-100
            )
            losses['task_loss'] = task_loss
        
        return losses
    
    def compute_total_loss(self, loss_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Combine all loss components"""
        total_loss = torch.tensor(0.0, device=next(iter(loss_dict.values())).device)
        
        # Logit distillation
        if 'kl_loss' in loss_dict:
            total_loss += self.alpha * loss_dict['kl_loss']
        
        # Hidden state alignment  
        if 'hidden_loss' in loss_dict:
            total_loss += self.hidden_weight * loss_dict['hidden_loss']
        
        # Attention transfer
        if 'attention_loss' in loss_dict:
            total_loss += self.attention_weight * loss_dict['attention_loss']
        
        # Task loss
        if 'task_loss' in loss_dict:
            total_loss += (1 - self.alpha) * loss_dict['task_loss']
        
        return total_loss


class FourierTrainer(Trainer):
    """
    Specialized trainer for Fourier approximation with knowledge distillation
    """
    
    def __init__(
        self,
        model=None,
        teacher_model=None,
        args: FourierTrainingArguments = None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=None,
        model_init=None,
        compute_metrics=None,
        callbacks=None,
        optimizers=(None, None),
        preprocess_logits_for_metrics=None,
        **kwargs
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        
        self.teacher_model = teacher_model
        if self.teacher_model:
            self.teacher_model.eval()
            # Freeze teacher model
            for param in self.teacher_model.parameters():
                param.requires_grad = False
        
        # Initialize knowledge distillation loss
        self.kd_loss = KnowledgeDistillationLoss(
            alpha=args.distillation_alpha,
            temperature=args.distillation_temperature,
            hidden_weight=args.hidden_distillation_weight,
            attention_weight=args.attention_distillation_weight
        )
        
        # Initialize Fourier optimizer for regularization
        self.fourier_optimizer = FourierOptimizer(
            model=self.model,
            frequency_penalty=args.frequency_regularization,
            low_freq_weight=args.low_freq_weight,
            high_freq_weight=args.high_freq_weight
        )
        
        # Initialize frequency scheduler
        self.frequency_scheduler = None
        if args.use_frequency_scheduling:
            total_steps = args.max_steps or (
                len(train_dataset) * args.num_train_epochs // args.per_device_train_batch_size
            )
            
            scheduler_config = {
                'type': args.frequency_scheduler_type,
                'warmup_steps': args.scheduler_warmup_steps or total_steps // 10,
                'min_components': args.scheduler_min_components,
                'max_components': args.scheduler_max_components
            }
            
            self.frequency_scheduler = create_frequency_scheduler(
                scheduler_config, total_steps, self.model
            )
        
        # Add frequency scheduling callback
        if self.frequency_scheduler:
            self.add_callback(FrequencySchedulingCallback(self.frequency_scheduler))
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute loss with knowledge distillation and Fourier regularization
        """
        # Get student outputs
        outputs = model(**inputs)
        
        # Get teacher outputs if available
        teacher_outputs = None
        if self.teacher_model is not None:
            with torch.no_grad():
                teacher_outputs = self.teacher_model(**inputs)
        
        # Compute knowledge distillation loss
        loss_dict = self.kd_loss(
            student_outputs=outputs,
            teacher_outputs=teacher_outputs,
            labels=inputs.get("labels")
        )
        
        # Add frequency regularization
        freq_reg_loss = self.fourier_optimizer.compute_frequency_regularization()
        loss_dict['frequency_reg'] = freq_reg_loss
        
        # Combine losses
        total_loss = self.kd_loss.compute_total_loss(loss_dict)
        total_loss += freq_reg_loss
        
        # Log individual loss components
        if self.state.global_step % 10 == 0:
            for loss_name, loss_value in loss_dict.items():
                self.log({f"train/{loss_name}": loss_value.item()})
        
        return (total_loss, outputs) if return_outputs else total_loss
    
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Override training step to add Fourier-specific operations
        """
        model.train()
        inputs = self._prepare_inputs(inputs)
        
        # Apply frequency dropout
        if self.args.frequency_dropout_rate > 0:
            self.fourier_optimizer.apply_frequency_dropout(self.args.frequency_dropout_rate)
        
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        
        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        
        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)
        
        return loss.detach() / self.args.gradient_accumulation_steps
    
    def log(self, logs: Dict[str, float]) -> None:
        """
        Override log to add Fourier-specific metrics
        """
        # Add frequency statistics
        if (self.args.log_frequency_stats and 
            self.state.global_step % self.args.frequency_stats_steps == 0):
            
            freq_stats = self._compute_frequency_statistics()
            logs.update(freq_stats)
        
        super().log(logs)
    
    def _compute_frequency_statistics(self) -> Dict[str, float]:
        """Compute frequency-related statistics for logging"""
        stats = {}
        
        total_active_components = 0
        total_components = 0
        compression_ratios = []
        
        for name, module in self.model.named_modules():
            if hasattr(module, 'fourier_layer'):
                fourier_module = module.fourier_layer
            elif hasattr(module, 'get_compression_ratio'):
                fourier_module = module
            else:
                continue
            
            # Active components
            if hasattr(fourier_module, 'gate_in') and hasattr(fourier_module, 'gate_out'):
                active_in = (fourier_module.gate_in > 0.5).sum().item()
                active_out = (fourier_module.gate_out > 0.5).sum().item()
                total_in = fourier_module.gate_in.shape[0]
                total_out = fourier_module.gate_out.shape[0]
                
                total_active_components += active_in + active_out
                total_components += total_in + total_out
            
            # Compression ratio
            if hasattr(fourier_module, 'get_compression_ratio'):
                ratio, _, _ = fourier_module.get_compression_ratio()
                compression_ratios.append(ratio)
        
        if total_components > 0:
            stats['frequency/active_ratio'] = total_active_components / total_components
        
        if compression_ratios:
            stats['frequency/avg_compression_ratio'] = np.mean(compression_ratios)
            stats['frequency/min_compression_ratio'] = np.min(compression_ratios)
            stats['frequency/max_compression_ratio'] = np.max(compression_ratios)
        
        return stats


class FrequencySchedulingCallback(TrainerCallback):
    """
    Callback for frequency scheduling during training
    """
    
    def __init__(self, frequency_scheduler):
        self.frequency_scheduler = frequency_scheduler
    
    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Update frequency masks at each step"""
        if self.frequency_scheduler and hasattr(self.frequency_scheduler, 'step'):
            # Get current loss for adaptive scheduling
            current_loss = None
            if hasattr(state, 'log_history') and state.log_history:
                current_loss = state.log_history[-1].get('train_loss')
            
            self.frequency_scheduler.step(state.global_step, current_loss)


def create_fourier_trainer(
    model,
    teacher_model,
    train_dataset,
    eval_dataset,
    training_args: FourierTrainingArguments,
    data_collator=None,
    tokenizer=None,
    **kwargs
) -> FourierTrainer:
    """
    Factory function to create FourierTrainer with proper setup
    """
    trainer = FourierTrainer(
        model=model,
        teacher_model=teacher_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        **kwargs
    )
    
    return trainer


# Example usage configuration
DEFAULT_FOURIER_TRAINING_CONFIG = {
    "distillation_alpha": 0.5,
    "distillation_temperature": 4.0,
    "hidden_distillation_weight": 0.5,
    "frequency_regularization": 1e-4,
    "use_frequency_scheduling": True,
    "frequency_scheduler_type": "linear",
    "scheduler_min_components": 0.1,
    "scheduler_max_components": 0.8,
    "log_frequency_stats": True,
    "frequency_stats_steps": 100
}
