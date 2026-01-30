"""
FrequencyScheduler: Progressive frequency component activation
Gradually enables higher frequency components during training for stable convergence
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Callable, Union
import math
from enum import Enum
import numpy as np


class SchedulerType(Enum):
    """Types of frequency scheduling strategies"""
    LINEAR = "linear"          # Linear increase in frequency components
    EXPONENTIAL = "exponential"  # Exponential increase 
    COSINE = "cosine"         # Cosine annealing schedule
    STEP = "step"             # Step-wise increases
    ADAPTIVE = "adaptive"     # Adaptive based on loss/metrics


class FrequencyScheduler:
    """
    Scheduler for progressive frequency component activation
    
    Manages the gradual enabling of higher frequency components during training
    to improve stability and convergence of Fourier approximation.
    """
    
    def __init__(
        self,
        scheduler_type: SchedulerType = SchedulerType.LINEAR,
        total_steps: int = 1000,
        warmup_steps: int = 100,
        min_components: Union[int, float] = 0.1,
        max_components: Union[int, float] = 1.0,
        temperature: float = 1.0,
        adaptive_threshold: float = 0.1
    ):
        """
        Initialize frequency scheduler
        
        Args:
            scheduler_type: Type of scheduling strategy
            total_steps: Total training steps
            warmup_steps: Steps for initial warmup
            min_components: Minimum fraction/number of components to start with
            max_components: Maximum fraction/number of components
            temperature: Temperature for soft gating (higher = softer transitions)
            adaptive_threshold: Threshold for adaptive scheduling
        """
        self.scheduler_type = scheduler_type
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_components = min_components
        self.max_components = max_components
        self.temperature = temperature
        self.adaptive_threshold = adaptive_threshold
        
        self.current_step = 0
        self.loss_history: List[float] = []
        
    def step(self, step: Optional[int] = None, loss: Optional[float] = None) -> float:
        """
        Update scheduler state and return current frequency ratio
        
        Args:
            step: Current training step (if None, uses internal counter)
            loss: Current loss value for adaptive scheduling
            
        Returns:
            Current frequency component ratio [0, 1]
        """
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1
            
        if loss is not None:
            self.loss_history.append(loss)
        
        return self._compute_frequency_ratio()
    
    def _compute_frequency_ratio(self) -> float:
        """Compute current frequency component ratio"""
        if self.scheduler_type == SchedulerType.LINEAR:
            return self._linear_schedule()
        elif self.scheduler_type == SchedulerType.EXPONENTIAL:
            return self._exponential_schedule()
        elif self.scheduler_type == SchedulerType.COSINE:
            return self._cosine_schedule()
        elif self.scheduler_type == SchedulerType.STEP:
            return self._step_schedule()
        elif self.scheduler_type == SchedulerType.ADAPTIVE:
            return self._adaptive_schedule()
        else:
            raise ValueError(f"Unknown scheduler type: {self.scheduler_type}")
    
    def _linear_schedule(self) -> float:
        """Linear frequency component scheduling"""
        if self.current_step < self.warmup_steps:
            return self.min_components
        
        progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        progress = min(1.0, progress)
        
        return self.min_components + (self.max_components - self.min_components) * progress
    
    def _exponential_schedule(self) -> float:
        """Exponential frequency component scheduling"""
        if self.current_step < self.warmup_steps:
            return self.min_components
            
        progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        progress = min(1.0, progress)
        
        # Exponential growth
        exp_progress = (math.exp(progress * 3) - 1) / (math.exp(3) - 1)  # Scale to [0,1]
        
        return self.min_components + (self.max_components - self.min_components) * exp_progress
    
    def _cosine_schedule(self) -> float:
        """Cosine annealing frequency component scheduling"""
        if self.current_step < self.warmup_steps:
            # Linear warmup
            warmup_progress = self.current_step / self.warmup_steps
            return self.min_components * warmup_progress
        
        progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        progress = min(1.0, progress)
        
        # Cosine annealing (starts fast, slows down)
        cosine_progress = (1 + math.cos(math.pi * (1 - progress))) / 2
        
        return self.min_components + (self.max_components - self.min_components) * cosine_progress
    
    def _step_schedule(self) -> float:
        """Step-wise frequency component scheduling"""
        if self.current_step < self.warmup_steps:
            return self.min_components
            
        # Define step boundaries (can be customized)
        step_boundaries = [0.25, 0.5, 0.75, 1.0]
        step_values = np.linspace(self.min_components, self.max_components, len(step_boundaries))
        
        progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        progress = min(1.0, progress)
        
        for i, boundary in enumerate(step_boundaries):
            if progress <= boundary:
                return step_values[i]
                
        return self.max_components
    
    def _adaptive_schedule(self) -> float:
        """Adaptive frequency component scheduling based on loss"""
        if len(self.loss_history) < 10:  # Need some history
            return self._linear_schedule()  # Fall back to linear
        
        # Check if loss is decreasing (moving average)
        recent_losses = self.loss_history[-10:]
        early_losses = self.loss_history[-20:-10] if len(self.loss_history) >= 20 else recent_losses
        
        recent_avg = sum(recent_losses) / len(recent_losses)
        early_avg = sum(early_losses) / len(early_losses)
        
        # If loss is decreasing well, increase frequencies faster
        if recent_avg < early_avg - self.adaptive_threshold:
            # Fast increase
            base_ratio = self._exponential_schedule()
        elif recent_avg > early_avg + self.adaptive_threshold:
            # Slow increase (loss not decreasing well)  
            base_ratio = self._linear_schedule() * 0.8
        else:
            # Normal increase
            base_ratio = self._linear_schedule()
            
        return min(self.max_components, base_ratio)
    
    def get_frequency_mask(
        self, 
        total_components: int,
        soft: bool = True
    ) -> torch.Tensor:
        """
        Generate frequency mask for current step
        
        Args:
            total_components: Total number of frequency components
            soft: Whether to use soft (continuous) or hard (binary) mask
            
        Returns:
            Frequency mask tensor
        """
        ratio = self._compute_frequency_ratio()
        
        if isinstance(self.max_components, float):
            active_components = int(total_components * ratio)
        else:
            active_components = int(ratio * self.max_components)
        
        active_components = min(active_components, total_components)
        
        if soft:
            # Soft masking with sigmoid transition
            indices = torch.arange(total_components, dtype=torch.float32)
            # Create smooth transition around the cutoff
            center = float(active_components)
            mask = torch.sigmoid(self.temperature * (center - indices))
            return mask
        else:
            # Hard binary mask
            mask = torch.zeros(total_components)
            mask[:active_components] = 1.0
            return mask
    
    def reset(self):
        """Reset scheduler state"""
        self.current_step = 0
        self.loss_history.clear()


class FrequencySchedulerManager:
    """
    Manager for applying frequency scheduling to multiple Fourier modules
    """
    
    def __init__(
        self,
        model: nn.Module,
        scheduler: FrequencyScheduler,
        module_types: Optional[List[str]] = None
    ):
        """
        Initialize frequency scheduler manager
        
        Args:
            model: Model containing Fourier modules
            scheduler: Frequency scheduler instance
            module_types: Types of modules to schedule (None = all Fourier modules)
        """
        self.model = model
        self.scheduler = scheduler
        self.module_types = module_types or ["FourierLinear", "FourierAdapter"]
        
        # Find all schedulable modules
        self.schedulable_modules = self._find_schedulable_modules()
        
    def _find_schedulable_modules(self) -> Dict[str, nn.Module]:
        """Find all modules that can be scheduled"""
        modules = {}
        
        for name, module in self.model.named_modules():
            module_type = type(module).__name__
            if module_type in self.module_types:
                modules[name] = module
            elif hasattr(module, 'fourier_layer'):  # FourierAdapter case
                modules[name] = module.fourier_layer
                
        return modules
    
    def step(self, step: Optional[int] = None, loss: Optional[float] = None):
        """
        Update frequency masks for all schedulable modules
        
        Args:
            step: Current training step
            loss: Current loss value
        """
        # Update scheduler
        current_ratio = self.scheduler.step(step, loss)
        
        # Apply to all modules
        for name, module in self.schedulable_modules.items():
            if hasattr(module, 'gate_in') and hasattr(module, 'gate_out'):
                # FourierLinear case
                k_in = module.gate_in.shape[0]
                k_out = module.gate_out.shape[0]
                
                mask_in = self.scheduler.get_frequency_mask(k_in, soft=True)
                mask_out = self.scheduler.get_frequency_mask(k_out, soft=True)
                
                module.apply_frequency_mask(mask_in.to(module.gate_in.device), 
                                          mask_out.to(module.gate_out.device))
    
    def get_active_components_info(self) -> Dict[str, Dict[str, int]]:
        """Get information about active frequency components per module"""
        info = {}
        
        for name, module in self.schedulable_modules.items():
            if hasattr(module, 'gate_in') and hasattr(module, 'gate_out'):
                active_in = (module.gate_in > 0.5).sum().item()
                active_out = (module.gate_out > 0.5).sum().item()
                total_in = module.gate_in.shape[0]
                total_out = module.gate_out.shape[0]
                
                info[name] = {
                    'active_in': active_in,
                    'total_in': total_in,
                    'active_out': active_out,
                    'total_out': total_out,
                    'ratio_in': active_in / total_in,
                    'ratio_out': active_out / total_out
                }
        
        return info


class WarmupFrequencyScheduler(FrequencyScheduler):
    """
    Extended scheduler with customizable warmup strategies
    """
    
    def __init__(
        self,
        scheduler_type: SchedulerType = SchedulerType.LINEAR,
        total_steps: int = 1000,
        warmup_steps: int = 100,
        warmup_type: str = "linear",  # "linear", "constant", "exponential"
        **kwargs
    ):
        super().__init__(scheduler_type, total_steps, warmup_steps, **kwargs)
        self.warmup_type = warmup_type
    
    def _apply_warmup(self, base_ratio: float) -> float:
        """Apply warmup strategy to base ratio"""
        if self.current_step >= self.warmup_steps:
            return base_ratio
        
        warmup_progress = self.current_step / self.warmup_steps
        
        if self.warmup_type == "linear":
            warmup_factor = warmup_progress
        elif self.warmup_type == "constant":
            warmup_factor = 1.0 if self.current_step > 0 else 0.0
        elif self.warmup_type == "exponential":
            warmup_factor = (math.exp(warmup_progress) - 1) / (math.exp(1) - 1)
        else:
            warmup_factor = warmup_progress
            
        return self.min_components * warmup_factor
    
    def _compute_frequency_ratio(self) -> float:
        """Override to apply warmup"""
        if self.current_step < self.warmup_steps:
            return self._apply_warmup(0.0)
        else:
            return super()._compute_frequency_ratio()


def create_frequency_scheduler(
    scheduler_config: Dict,
    total_steps: int,
    model: Optional[nn.Module] = None
) -> Union[FrequencyScheduler, FrequencySchedulerManager]:
    """
    Factory function to create frequency scheduler
    
    Args:
        scheduler_config: Configuration dictionary
        total_steps: Total training steps
        model: Model for manager creation (optional)
        
    Returns:
        FrequencyScheduler or FrequencySchedulerManager
    """
    scheduler_type = SchedulerType(scheduler_config.get('type', 'linear'))
    
    scheduler = FrequencyScheduler(
        scheduler_type=scheduler_type,
        total_steps=total_steps,
        warmup_steps=scheduler_config.get('warmup_steps', total_steps // 10),
        min_components=scheduler_config.get('min_components', 0.1),
        max_components=scheduler_config.get('max_components', 1.0),
        temperature=scheduler_config.get('temperature', 1.0),
        adaptive_threshold=scheduler_config.get('adaptive_threshold', 0.1)
    )
    
    if model is not None:
        return FrequencySchedulerManager(model, scheduler)
    
    return scheduler


# Example usage and configuration presets
SCHEDULER_PRESETS = {
    "conservative": {
        "type": "linear",
        "min_components": 0.05,
        "max_components": 0.8,
        "warmup_steps": 500,
        "temperature": 2.0
    },
    "aggressive": {
        "type": "exponential", 
        "min_components": 0.1,
        "max_components": 1.0,
        "warmup_steps": 100,
        "temperature": 1.0
    },
    "adaptive": {
        "type": "adaptive",
        "min_components": 0.08,
        "max_components": 0.9,
        "warmup_steps": 200,
        "temperature": 1.5,
        "adaptive_threshold": 0.05
    }
}
