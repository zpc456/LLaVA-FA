"""
Compression utilities for Fourier approximation
Includes sparsification, quantization, and optimization tools
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import numpy as np


def compress_fourier_weights(
    coefficients: torch.Tensor,
    sparsity_ratio: float = 0.5,
    method: str = "magnitude"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compress Fourier coefficients through sparsification
    
    Args:
        coefficients: Coefficient matrix (k_out, k_in)
        sparsity_ratio: Fraction of weights to remove
        method: Sparsification method ("magnitude", "random", "structured")
        
    Returns:
        Compressed coefficients and binary mask
    """
    if method == "magnitude":
        # Keep largest magnitude coefficients
        flat_weights = coefficients.flatten()
        k = int(len(flat_weights) * (1 - sparsity_ratio))
        
        _, indices = torch.topk(flat_weights.abs(), k)
        mask = torch.zeros_like(flat_weights)
        mask[indices] = 1.0
        mask = mask.reshape(coefficients.shape)
        
    elif method == "random":
        # Random sparsification
        mask = torch.rand_like(coefficients) > sparsity_ratio
        
    elif method == "structured":
        # Structured sparsification (remove entire frequency bands)
        k_out, k_in = coefficients.shape
        
        # Calculate importance per frequency
        importance_out = coefficients.norm(dim=1)  # (k_out,)
        importance_in = coefficients.norm(dim=0)   # (k_in,)
        
        # Keep top frequency components
        keep_out = int(k_out * (1 - sparsity_ratio))
        keep_in = int(k_in * (1 - sparsity_ratio))
        
        _, out_indices = torch.topk(importance_out, keep_out)
        _, in_indices = torch.topk(importance_in, keep_in)
        
        mask = torch.zeros_like(coefficients)
        mask[out_indices.unsqueeze(1), in_indices.unsqueeze(0)] = 1.0
    else:
        raise ValueError(f"Unknown sparsification method: {method}")
    
    compressed_coeffs = coefficients * mask
    return compressed_coeffs, mask


def quantize_fourier_weights(
    coefficients: torch.Tensor,
    bits: int = 8,
    method: str = "symmetric"
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Quantize Fourier coefficients
    
    Args:
        coefficients: Coefficient matrix  
        bits: Number of quantization bits
        method: Quantization method ("symmetric", "asymmetric")
        
    Returns:
        Quantized coefficients and quantization parameters
    """
    if method == "symmetric":
        # Symmetric quantization around zero
        max_val = coefficients.abs().max()
        scale = max_val / (2**(bits-1) - 1)
        
        quantized = torch.round(coefficients / scale).clamp(
            -(2**(bits-1)), 2**(bits-1) - 1
        )
        dequantized = quantized * scale
        
        quant_params = {"scale": scale, "zero_point": torch.tensor(0)}
        
    elif method == "asymmetric":
        # Asymmetric quantization  
        min_val = coefficients.min()
        max_val = coefficients.max()
        
        scale = (max_val - min_val) / (2**bits - 1)
        zero_point = torch.round(-min_val / scale).clamp(0, 2**bits - 1)
        
        quantized = torch.round(coefficients / scale + zero_point).clamp(0, 2**bits - 1)
        dequantized = (quantized - zero_point) * scale
        
        quant_params = {"scale": scale, "zero_point": zero_point}
    else:
        raise ValueError(f"Unknown quantization method: {method}")
    
    return dequantized, quant_params


class FourierOptimizer:
    """
    Specialized optimizer for Fourier approximation training
    Applies frequency-aware regularization and updates
    """
    
    def __init__(
        self,
        model: nn.Module,
        frequency_penalty: float = 1e-4,
        low_freq_weight: float = 1.0,
        high_freq_weight: float = 2.0
    ):
        self.model = model
        self.frequency_penalty = frequency_penalty
        self.low_freq_weight = low_freq_weight
        self.high_freq_weight = high_freq_weight
    
    def compute_frequency_regularization(self) -> torch.Tensor:
        """
        Compute frequency-aware regularization loss
        Penalizes high-frequency components more heavily
        """
        reg_loss = torch.tensor(0.0, device=next(self.model.parameters()).device)
        
        for module in self.model.modules():
            if hasattr(module, 'coefficients') and hasattr(module, 'get_frequency_indices'):
                coeffs = module.coefficients
                freq_indices = module.fourier_layer.basis_factory.get_frequency_indices(
                    coeffs.shape[1]
                ).to(coeffs.device)
                
                # Higher frequencies get higher penalties
                freq_weights = self.low_freq_weight + (
                    self.high_freq_weight - self.low_freq_weight
                ) * (freq_indices / freq_indices.max())
                
                # Apply per-frequency penalties
                freq_reg = (coeffs.pow(2) * freq_weights.unsqueeze(0)).sum()
                reg_loss += freq_reg
        
        return self.frequency_penalty * reg_loss
    
    def apply_frequency_dropout(self, dropout_rate: float = 0.1):
        """
        Apply frequency dropout during training
        Randomly masks frequency components to improve robustness
        """
        for module in self.model.modules():
            if hasattr(module, 'gate_in') and hasattr(module, 'gate_out'):
                if self.model.training:
                    # Random frequency dropout
                    dropout_mask_in = (torch.rand_like(module.gate_in) > dropout_rate).float()
                    dropout_mask_out = (torch.rand_like(module.gate_out) > dropout_rate).float()
                    
                    module.gate_in.data *= dropout_mask_in
                    module.gate_out.data *= dropout_mask_out


class FourierAnalyzer:
    """
    Analysis tools for Fourier approximation quality
    """
    
    @staticmethod
    def compute_reconstruction_error(
        original_weight: torch.Tensor,
        fourier_module: nn.Module
    ) -> Dict[str, float]:
        """
        Compute reconstruction error between original and Fourier approximation
        """
        with torch.no_grad():
            # Reconstruct weight from Fourier approximation
            if hasattr(fourier_module, 'to_linear'):
                reconstructed = fourier_module.to_linear().weight
            else:
                # Manual reconstruction
                coeffs = fourier_module.coefficients
                basis_out = fourier_module.basis_out
                basis_in = fourier_module.basis_in
                gate_out = fourier_module.gate_out.unsqueeze(1)
                gate_in = fourier_module.gate_in.unsqueeze(0)
                
                reconstructed = basis_out @ (coeffs * gate_out * gate_in) @ basis_in.T
            
            # Compute various error metrics
            mse = F.mse_loss(reconstructed, original_weight).item()
            mae = F.l1_loss(reconstructed, original_weight).item()
            
            # Relative error
            rel_error = (reconstructed - original_weight).norm().item() / original_weight.norm().item()
            
            # Frobenius norm ratio
            frob_ratio = reconstructed.norm().item() / original_weight.norm().item()
            
            return {
                "mse": mse,
                "mae": mae,
                "relative_error": rel_error,
                "frobenius_ratio": frob_ratio
            }
    
    @staticmethod
    def analyze_frequency_spectrum(
        coefficients: torch.Tensor,
        basis_factory
    ) -> Dict[str, np.ndarray]:
        """
        Analyze frequency spectrum of coefficients
        """
        with torch.no_grad():
            # Get frequency indices
            k_out, k_in = coefficients.shape
            freq_indices_out = basis_factory.get_frequency_indices(k_out).numpy()
            freq_indices_in = basis_factory.get_frequency_indices(k_in).numpy()
            
            # Compute energy per frequency
            energy_out = coefficients.norm(dim=1).cpu().numpy()
            energy_in = coefficients.norm(dim=0).cpu().numpy()
            
            # Cumulative energy
            sorted_indices_out = np.argsort(freq_indices_out)
            sorted_indices_in = np.argsort(freq_indices_in)
            
            cumulative_energy_out = np.cumsum(energy_out[sorted_indices_out])
            cumulative_energy_in = np.cumsum(energy_in[sorted_indices_in])
            
            # Normalize
            cumulative_energy_out /= cumulative_energy_out[-1]
            cumulative_energy_in /= cumulative_energy_in[-1]
            
            return {
                "frequency_indices_out": freq_indices_out,
                "frequency_indices_in": freq_indices_in,
                "energy_out": energy_out,
                "energy_in": energy_in,
                "cumulative_energy_out": cumulative_energy_out,
                "cumulative_energy_in": cumulative_energy_in
            }
    
    @staticmethod
    def compute_compression_metrics(model: nn.Module) -> Dict[str, float]:
        """
        Compute overall compression metrics for model
        """
        total_original_params = 0
        total_compressed_params = 0
        total_active_params = 0
        
        for module in model.modules():
            if hasattr(module, 'get_compression_ratio'):
                ratio, orig, comp = module.get_compression_ratio()
                total_original_params += orig
                total_compressed_params += comp
                
                # Count active parameters (based on gates)
                if hasattr(module, 'gate_in') and hasattr(module, 'gate_out'):
                    active_in = (module.gate_in > 0.5).sum().item()
                    active_out = (module.gate_out > 0.5).sum().item()
                    active_params = active_in * active_out
                    
                    if module.bias is not None:
                        active_params += module.out_features
                        
                    total_active_params += active_params
                else:
                    total_active_params += comp
        
        overall_ratio = total_compressed_params / total_original_params if total_original_params > 0 else 0
        active_ratio = total_active_params / total_original_params if total_original_params > 0 else 0
        
        return {
            "total_original_params": total_original_params,
            "total_compressed_params": total_compressed_params,
            "total_active_params": total_active_params,
            "compression_ratio": overall_ratio,
            "active_ratio": active_ratio,
            "parameter_reduction": 1 - overall_ratio,
            "effective_reduction": 1 - active_ratio
        }


def optimize_fourier_model(
    model: nn.Module,
    calibration_data: torch.utils.data.DataLoader,
    sparsity_target: float = 0.5,
    quantization_bits: int = 8
) -> nn.Module:
    """
    Post-training optimization of Fourier model
    Applies sparsification and quantization for maximum compression
    """
    model.eval()
    
    # Collect calibration statistics
    with torch.no_grad():
        for batch in calibration_data:
            # Forward pass to collect activation statistics
            _ = model(batch)
            break  # One batch is usually sufficient for calibration
    
    # Apply compression to all Fourier modules
    for name, module in model.named_modules():
        if hasattr(module, 'coefficients'):
            # Sparsify coefficients
            compressed_coeffs, mask = compress_fourier_weights(
                module.coefficients, sparsity_target, "magnitude"
            )
            
            # Quantize coefficients
            quantized_coeffs, quant_params = quantize_fourier_weights(
                compressed_coeffs, quantization_bits, "symmetric"
            )
            
            # Update module
            module.coefficients.data.copy_(quantized_coeffs)
            
            # Store quantization parameters
            module.register_buffer('quant_scale', quant_params['scale'])
            module.register_buffer('quant_zero_point', quant_params['zero_point'])
            module.register_buffer('sparsity_mask', mask)
    
    return model
