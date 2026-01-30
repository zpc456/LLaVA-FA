"""
FourierAdapter: LoRA-style Fourier approximation injection
Injects Fourier approximation as residual connections to existing linear layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Set, Pattern
import re
from .fourier_linear import FourierLinear
from .basis_factory import BasisType


class FourierAdapter(nn.Module):
    """
    Fourier Adapter for injecting Fourier approximation into existing linear layers
    
    Similar to LoRA, adds residual Fourier approximation:
    y = W @ x + α * Fourier(x)
    
    Args:
        in_features: Input dimension
        out_features: Output dimension  
        k_in: Number of input frequency components
        k_out: Number of output frequency components
        basis_type: Type of frequency basis
        scaling: Scaling factor for Fourier residual
        dropout: Dropout rate for regularization
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        k_in: Optional[int] = None,
        k_out: Optional[int] = None,
        basis_type: BasisType = BasisType.DCT,
        scaling: float = 1.0,
        dropout: float = 0.0,
        compression_ratio: float = 0.05,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.scaling = scaling
        
        # Create Fourier approximation layer
        self.fourier_layer = FourierLinear(
            in_features=in_features,
            out_features=out_features,
            k_in=k_in,
            k_out=k_out,
            basis_type=basis_type,
            bias=False,  # No bias for residual connection
            device=device,
            dtype=dtype,
            compression_ratio=compression_ratio
        )
        
        # Optional dropout for regularization
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Fourier adapter
        
        Args:
            x: Input tensor
            
        Returns:
            Fourier residual: α * Fourier(x)
        """
        fourier_output = self.fourier_layer(x)
        fourier_output = self.dropout(fourier_output)
        return self.scaling * fourier_output
    
    def reset_parameters(self):
        """Reset parameters with proper initialization"""
        # Reset Fourier layer parameters
        with torch.no_grad():
            # Initialize coefficients with small random values
            nn.init.normal_(self.fourier_layer.coefficients, std=0.02)
            # Initialize gates to ones (all frequencies enabled)
            nn.init.ones_(self.fourier_layer.gate_in)
            nn.init.ones_(self.fourier_layer.gate_out)


class FourierAdapterConfig:
    """Configuration for Fourier adapter injection"""
    
    def __init__(
        self,
        target_modules: Optional[List[str]] = None,
        exclude_modules: Optional[List[str]] = None,
        compression_ratios: Optional[Dict[str, float]] = None,
        basis_type: BasisType = BasisType.DCT,
        scaling: float = 1.0,
        dropout: float = 0.0,
        enable_per_layer_scaling: bool = False
    ):
        # Default target modules (similar to LoRA targeting)
        if target_modules is None:
            target_modules = [
                r".*\.q_proj$",
                r".*\.k_proj$", 
                r".*\.v_proj$",
                r".*\.o_proj$",
                r".*\.up_proj$",
                r".*\.down_proj$",
                r".*\.gate_proj$",
                r".*\.mm_projector.*\.linear.*$"
            ]
        
        self.target_modules = [re.compile(pattern) for pattern in target_modules]
        self.exclude_modules = [re.compile(pattern) for pattern in (exclude_modules or [])]
        
        # Compression ratios per module type
        default_ratios = {
            "q_proj": 0.04,
            "k_proj": 0.04, 
            "v_proj": 0.04,
            "o_proj": 0.05,
            "up_proj": 0.03,
            "down_proj": 0.03,
            "gate_proj": 0.03,
            "mm_projector": 0.05
        }
        
        self.compression_ratios = compression_ratios or default_ratios
        self.basis_type = basis_type
        self.scaling = scaling
        self.dropout = dropout
        self.enable_per_layer_scaling = enable_per_layer_scaling


class FourierAdapterInjector:
    """
    Injector for adding Fourier adapters to existing models
    
    Similar to PEFT's LoRA injection but for Fourier approximation
    """
    
    def __init__(self, config: FourierAdapterConfig):
        self.config = config
        self.injected_modules: Dict[str, FourierAdapter] = {}
        
    def inject_model(self, model: nn.Module, verbose: bool = True) -> nn.Module:
        """
        Inject Fourier adapters into model
        
        Args:
            model: Target model to inject adapters
            verbose: Whether to print injection info
            
        Returns:
            Modified model with Fourier adapters
        """
        injection_count = 0
        
        for name, module in model.named_modules():
            if self._should_inject(name, module):
                adapter = self._create_adapter(name, module)
                self._inject_adapter(model, name, module, adapter)
                self.injected_modules[name] = adapter
                injection_count += 1
                
                if verbose:
                    print(f"Injected FourierAdapter at {name}: "
                          f"{module.in_features}→{module.out_features} "
                          f"(compression: {adapter.fourier_layer.get_compression_ratio()[0]:.4f})")
        
        if verbose:
            total_original, total_compressed = self._compute_total_compression()
            print(f"\nTotal injection: {injection_count} modules")
            print(f"Total compression ratio: {total_compressed / total_original:.4f}")
            print(f"Parameter reduction: {total_original - total_compressed:,} → {total_compressed:,}")
        
        return model
    
    def _should_inject(self, name: str, module: nn.Module) -> bool:
        """Check if module should be injected with adapter"""
        if not isinstance(module, nn.Linear):
            return False
            
        # Check exclusions first
        for pattern in self.config.exclude_modules:
            if pattern.search(name):
                return False
        
        # Check inclusions
        for pattern in self.config.target_modules:
            if pattern.search(name):
                return True
                
        return False
    
    def _create_adapter(self, name: str, module: nn.Linear) -> FourierAdapter:
        """Create Fourier adapter for specific module"""
        # Determine compression ratio
        compression_ratio = self._get_compression_ratio(name)
        
        # Determine scaling factor
        scaling = self.config.scaling
        if self.config.enable_per_layer_scaling:
            # Adaptive scaling based on module type
            if any(proj in name for proj in ["q_proj", "k_proj", "v_proj"]):
                scaling *= 0.5  # Lower scaling for attention projections
            elif "mm_projector" in name:
                scaling *= 2.0  # Higher scaling for multimodal projectors
        
        adapter = FourierAdapter(
            in_features=module.in_features,
            out_features=module.out_features,
            basis_type=self.config.basis_type,
            scaling=scaling,
            dropout=self.config.dropout,
            compression_ratio=compression_ratio,
            device=module.weight.device,
            dtype=module.weight.dtype
        )
        
        return adapter
    
    def _get_compression_ratio(self, name: str) -> float:
        """Get compression ratio for specific module"""
        # Check for specific module type patterns
        for module_type, ratio in self.config.compression_ratios.items():
            if module_type in name:
                return ratio
        
        # Default compression ratio
        return 0.05
    
    def _inject_adapter(
        self, 
        model: nn.Module,
        name: str, 
        original_module: nn.Linear,
        adapter: FourierAdapter
    ):
        """Inject adapter into model by replacing module"""
        # Create wrapper that combines original + adapter
        wrapper = FourierLinearWrapper(original_module, adapter)
        
        # Replace module in model
        parent_name, module_name = name.rsplit('.', 1) if '.' in name else ('', name)
        
        if parent_name:
            parent_module = model.get_submodule(parent_name)
            setattr(parent_module, module_name, wrapper)
        else:
            setattr(model, module_name, wrapper)
    
    def _compute_total_compression(self) -> tuple[int, int]:
        """Compute total compression statistics"""
        total_original = 0
        total_compressed = 0
        
        for adapter in self.injected_modules.values():
            ratio, orig, comp = adapter.fourier_layer.get_compression_ratio()
            total_original += orig
            total_compressed += comp
            
        return total_original, total_compressed
    
    def get_trainable_parameters(self) -> Dict[str, torch.Tensor]:
        """Get all trainable Fourier adapter parameters"""
        params = {}
        for name, adapter in self.injected_modules.items():
            for param_name, param in adapter.named_parameters():
                if param.requires_grad:
                    params[f"{name}.{param_name}"] = param
        return params
    
    def freeze_original_parameters(self, model: nn.Module):
        """Freeze all non-Fourier parameters"""
        for name, param in model.named_parameters():
            # Only train Fourier adapter parameters
            if not any(adapter_name in name for adapter_name in self.injected_modules.keys()):
                param.requires_grad = False
            else:
                param.requires_grad = True
    
    def save_adapters(self, path: str):
        """Save only Fourier adapter weights"""
        adapter_state = {}
        for name, adapter in self.injected_modules.items():
            adapter_state[name] = adapter.state_dict()
        
        torch.save({
            'adapters': adapter_state,
            'config': self.config.__dict__
        }, path)
    
    def load_adapters(self, path: str, model: nn.Module):
        """Load Fourier adapter weights"""
        checkpoint = torch.load(path, map_location='cpu')
        adapter_state = checkpoint['adapters']
        
        for name, state_dict in adapter_state.items():
            if name in self.injected_modules:
                self.injected_modules[name].load_state_dict(state_dict)
        
        return model


class FourierLinearWrapper(nn.Module):
    """
    Wrapper that combines original linear layer with Fourier adapter
    
    Forward: y = W @ x + α * Fourier(x)
    """
    
    def __init__(self, original: nn.Linear, adapter: FourierAdapter):
        super().__init__()
        self.original = original
        self.adapter = adapter
        
        # Copy original properties
        self.in_features = original.in_features
        self.out_features = original.out_features
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original linear transformation
        original_output = self.original(x)
        
        # Add Fourier residual
        fourier_residual = self.adapter(x)
        
        return original_output + fourier_residual
    
    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, " + \
               f"fourier_compression={self.adapter.fourier_layer.get_compression_ratio()[0]:.4f}"


def inject_fourier_adapters(
    model: nn.Module,
    config: Optional[FourierAdapterConfig] = None,
    **kwargs
) -> tuple[nn.Module, FourierAdapterInjector]:
    """
    Convenience function to inject Fourier adapters into model
    
    Args:
        model: Target model
        config: Injection configuration
        **kwargs: Additional config parameters
        
    Returns:
        Modified model and injector instance
    """
    if config is None:
        config = FourierAdapterConfig(**kwargs)
    
    injector = FourierAdapterInjector(config)
    modified_model = injector.inject_model(model)
    
    # Freeze original parameters by default
    injector.freeze_original_parameters(modified_model)
    
    return modified_model, injector


def get_fourier_adapter_parameters(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Extract all Fourier adapter parameters from model"""
    params = {}
    for name, module in model.named_modules():
        if isinstance(module, (FourierAdapter, FourierLinearWrapper)):
            for param_name, param in module.adapter.named_parameters():
                if param.requires_grad:
                    params[f"{name}.{param_name}"] = param
    return params


def count_fourier_parameters(model: nn.Module) -> Dict[str, int]:
    """Count parameters in Fourier adapters vs original model"""
    fourier_params = 0
    original_params = 0
    total_params = 0
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        
        if any(fourier_name in name for fourier_name in ['fourier', 'adapter']):
            fourier_params += param_count
        else:
            original_params += param_count
    
    return {
        'total': total_params,
        'fourier': fourier_params,
        'original': original_params,
        'fourier_ratio': fourier_params / total_params if total_params > 0 else 0
    }
