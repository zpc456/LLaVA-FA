"""
FourierLinear: Core operator for Fourier Approximation
Approximates large weight matrices using fixed frequency domain bases
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math
from .basis_factory import BasisFactory, BasisType


class FourierLinear(nn.Module):
    """
    Fourier approximation of linear layer using fixed frequency domain bases
    
    W ≈ Φ_out @ C @ Φ_in^T
    y = W @ x ≈ Φ_out @ (C @ (Φ_in^T @ x))
    
    Args:
        in_features: Input dimension
        out_features: Output dimension  
        k_in: Number of input frequency components (default: 5% of in_features)
        k_out: Number of output frequency components (default: 5% of out_features)
        basis_type: Type of frequency basis ('dct', 'dst', 'fourier')
        bias: Whether to use bias
        device: Device to place parameters
        dtype: Data type for parameters
        sparse: Whether to use sparse coefficients
        quantize: Whether to quantize coefficients
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        k_in: Optional[int] = None,
        k_out: Optional[int] = None,
        basis_type: BasisType = BasisType.DCT,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        sparse: bool = False,
        quantize: bool = False,
        compression_ratio: float = 0.05
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.basis_type = basis_type
        self.sparse = sparse
        self.quantize = quantize
        
        # Determine frequency dimensions
        if k_in is None:
            k_in = max(1, int(in_features * compression_ratio))
        if k_out is None:
            k_out = max(1, int(out_features * compression_ratio))
            
        self.k_in = k_in
        self.k_out = k_out
        
        # Create basis factory
        self.basis_factory = BasisFactory(basis_type, device=device, dtype=dtype)
        
        # Get frequency domain bases (fixed, not trainable)
        self.register_buffer(
            'basis_in', 
            self.basis_factory.get_basis(in_features, k_in)
        )
        self.register_buffer(
            'basis_out',
            self.basis_factory.get_basis(out_features, k_out) 
        )
        
        # Learnable frequency coefficients
        self.coefficients = nn.Parameter(
            torch.randn(k_out, k_in, device=device, dtype=dtype) * 0.02
        )
        
        # Optional bias
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_features, device=device, dtype=dtype)
            )
        else:
            self.register_parameter('bias', None)
            
        # Frequency gates for scheduling (optional)
        self.gate_in = nn.Parameter(
            torch.ones(k_in, device=device, dtype=dtype)
        )
        self.gate_out = nn.Parameter(
            torch.ones(k_out, device=device, dtype=dtype)
        )
        
        # Quantization parameters
        if quantize:
            self.register_buffer('scale', torch.ones(1, device=device, dtype=dtype))
            self.register_buffer('zero_point', torch.zeros(1, device=device, dtype=torch.int8))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Fourier approximated linear layer
        
        Args:
            x: Input tensor (..., in_features)
            
        Returns:
            Output tensor (..., out_features)
        """
        # Apply frequency domain transformation
        # Step 1: Project input to frequency domain
        t = F.linear(x, self.basis_in.T)  # (..., k_in)
        
        # Apply input frequency gates (for scheduling)
        t = t * self.gate_in
        
        # Step 2: Transform through coefficient matrix
        if self.quantize and self.training:
            # Quantized forward (simulated)
            coeffs = self._quantize_coefficients(self.coefficients)
        else:
            coeffs = self.coefficients
            
        u = F.linear(t, coeffs)  # (..., k_out)
        
        # Apply output frequency gates
        u = u * self.gate_out
        
        # Step 3: Project back to output space
        y = F.linear(u, self.basis_out)  # (..., out_features)
        
        # Add bias if present
        if self.bias is not None:
            y = y + self.bias
            
        return y
    
    def _quantize_coefficients(self, coeffs: torch.Tensor) -> torch.Tensor:
        """Simulate quantization during training"""
        if not hasattr(self, 'scale'):
            return coeffs
            
        # Symmetric quantization
        scale = coeffs.abs().max() / 127.0
        quantized = torch.round(coeffs / scale).clamp(-128, 127)
        return quantized * scale
    
    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        k_in: Optional[int] = None,
        k_out: Optional[int] = None,
        basis_type: BasisType = BasisType.DCT,
        compression_ratio: float = 0.05
    ) -> 'FourierLinear':
        """
        Create FourierLinear from existing Linear layer
        
        Args:
            linear: Existing nn.Linear layer
            k_in: Number of input frequency components
            k_out: Number of output frequency components
            basis_type: Type of frequency basis
            compression_ratio: Compression ratio if k_in/k_out not specified
            
        Returns:
            FourierLinear layer initialized from linear layer
        """
        fourier_linear = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            k_in=k_in,
            k_out=k_out,
            basis_type=basis_type,
            bias=linear.bias is not None,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
            compression_ratio=compression_ratio
        )
        
        # Initialize coefficients from original weight
        with torch.no_grad():
            # Project original weight to frequency domain
            W = linear.weight  # (out_features, in_features)
            coeffs_init = fourier_linear.basis_out.T @ W @ fourier_linear.basis_in
            fourier_linear.coefficients.copy_(coeffs_init)
            
            # Copy bias
            if linear.bias is not None:
                fourier_linear.bias.copy_(linear.bias)
                
        return fourier_linear
    
    def to_linear(self) -> nn.Linear:
        """
        Convert FourierLinear back to regular Linear layer
        Useful for deployment after training
        """
        with torch.no_grad():
            # Reconstruct full weight matrix
            W_approx = self.basis_out @ (
                self.coefficients * self.gate_out.unsqueeze(1)
            ) @ (self.basis_in * self.gate_in).T
            
            linear = nn.Linear(
                self.in_features, 
                self.out_features,
                bias=self.bias is not None,
                device=self.coefficients.device,
                dtype=self.coefficients.dtype
            )
            
            linear.weight.copy_(W_approx)
            if self.bias is not None:
                linear.bias.copy_(self.bias)
                
        return linear
    
    def get_compression_ratio(self) -> Tuple[float, int, int]:
        """
        Calculate actual compression ratio
        
        Returns:
            Compression ratio, original params, compressed params
        """
        original_params = self.in_features * self.out_features
        if self.bias is not None:
            original_params += self.out_features
            
        compressed_params = self.k_in * self.k_out
        if self.bias is not None:
            compressed_params += self.out_features
            
        # Add frequency gates
        compressed_params += self.k_in + self.k_out
        
        ratio = compressed_params / original_params
        
        return ratio, original_params, compressed_params
    
    def apply_frequency_mask(self, mask_in: torch.Tensor, mask_out: torch.Tensor):
        """
        Apply frequency mask for progressive training
        
        Args:
            mask_in: Binary mask for input frequencies (k_in,)
            mask_out: Binary mask for output frequencies (k_out,)
        """
        self.gate_in.data.copy_(mask_in.float())
        self.gate_out.data.copy_(mask_out.float())
    
    def get_frequency_importance(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get importance scores for frequency components
        Useful for adaptive compression and pruning
        
        Returns:
            Input frequency importance, output frequency importance
        """
        with torch.no_grad():
            # L2 norm across coefficients
            input_importance = self.coefficients.norm(dim=0)  # (k_in,)
            output_importance = self.coefficients.norm(dim=1)  # (k_out,)
            
        return input_importance, output_importance
    
    def prune_frequencies(self, sparsity_ratio: float = 0.5):
        """
        Prune least important frequency components
        
        Args:
            sparsity_ratio: Fraction of frequencies to prune
        """
        with torch.no_grad():
            input_importance, output_importance = self.get_frequency_importance()
            
            # Determine pruning thresholds
            k_in_keep = int(self.k_in * (1 - sparsity_ratio))
            k_out_keep = int(self.k_out * (1 - sparsity_ratio))
            
            # Get top-k frequencies
            _, input_indices = input_importance.topk(k_in_keep)
            _, output_indices = output_importance.topk(k_out_keep)
            
            # Create masks
            input_mask = torch.zeros_like(self.gate_in)
            output_mask = torch.zeros_like(self.gate_out)
            
            input_mask[input_indices] = 1.0
            output_mask[output_indices] = 1.0
            
            self.apply_frequency_mask(input_mask, output_mask)
    
    def extra_repr(self) -> str:
        return (f'in_features={self.in_features}, out_features={self.out_features}, '
                f'k_in={self.k_in}, k_out={self.k_out}, '
                f'basis_type={self.basis_type.value}, '
                f'compression_ratio={self.get_compression_ratio()[0]:.4f}')
