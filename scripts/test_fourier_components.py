#!/usr/bin/env python3
"""
Quick test script for LLaVA-FA Fourier components
Tests basic functionality without full model loading
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add LLaVA-FA to path
sys.path.append(str(Path(__file__).parent.parent))

from llavafd.fourier import (
    FourierLinear, BasisFactory, BasisType, FourierAdapter,
    FrequencyScheduler, SchedulerType, compress_fourier_weights,
    quantize_fourier_weights
)


def test_basis_factory():
    """Test BasisFactory functionality"""
    print("🧪 Testing BasisFactory...")
    
    factory = BasisFactory(BasisType.DCT)
    
    # Test basis generation
    basis = factory.get_basis(n_dim=256, k_components=32)
    print(f"✅ Generated DCT basis: {basis.shape}")
    
    # Test orthogonality (should be close to identity)
    orthogonality = torch.mm(basis.T, basis)
    is_orthogonal = torch.allclose(orthogonality, torch.eye(32), atol=1e-6)
    print(f"✅ Basis orthogonality: {'PASS' if is_orthogonal else 'FAIL'}")
    
    # Test 2D basis
    row_basis, col_basis = factory.get_2d_basis(64, 64, 8, 8)
    print(f"✅ 2D basis generation: {row_basis.shape}, {col_basis.shape}")
    
    # Test frequency indices
    freq_indices = factory.get_frequency_indices(32)
    print(f"✅ Frequency indices: {freq_indices[:10]}...")


def test_fourier_linear():
    """Test FourierLinear layer"""
    print("\n🧪 Testing FourierLinear...")
    
    # Create test linear layer
    linear = torch.nn.Linear(512, 256)
    
    # Convert to Fourier
    fourier_linear = FourierLinear.from_linear(
        linear, compression_ratio=0.1, basis_type=BasisType.DCT
    )
    
    print(f"✅ Converted Linear to FourierLinear")
    
    # Test forward pass
    x = torch.randn(8, 32, 512)
    original_output = linear(x)
    fourier_output = fourier_linear(x)
    
    print(f"✅ Forward pass: {original_output.shape} -> {fourier_output.shape}")
    
    # Check compression ratio
    ratio, orig_params, comp_params = fourier_linear.get_compression_ratio()
    print(f"✅ Compression ratio: {ratio:.4f} ({orig_params} -> {comp_params})")
    
    # Test reconstruction error
    mse_error = torch.nn.functional.mse_loss(fourier_output, original_output)
    print(f"✅ Reconstruction MSE: {mse_error.item():.6f}")
    
    # Test conversion back to linear
    reconstructed_linear = fourier_linear.to_linear()
    reconstructed_output = reconstructed_linear(x)
    
    consistency_error = torch.nn.functional.mse_loss(fourier_output, reconstructed_output)
    print(f"✅ Conversion consistency: {consistency_error.item():.6f}")


def test_frequency_scheduler():
    """Test FrequencyScheduler"""
    print("\n🧪 Testing FrequencyScheduler...")
    
    scheduler = FrequencyScheduler(
        scheduler_type=SchedulerType.LINEAR,
        total_steps=1000,
        warmup_steps=100,
        min_components=0.1,
        max_components=0.8
    )
    
    # Test scheduling progression
    ratios = []
    for step in [0, 50, 100, 300, 500, 800, 1000]:
        ratio = scheduler.step(step)
        ratios.append(ratio)
        print(f"  Step {step:4d}: {ratio:.4f}")
    
    print(f"✅ Scheduler progression: {ratios[0]:.3f} -> {ratios[-1]:.3f}")
    
    # Test frequency masks
    mask = scheduler.get_frequency_mask(64, soft=True)
    active_components = (mask > 0.5).sum().item()
    print(f"✅ Frequency mask: {active_components}/64 active components")


def test_compression_utils():
    """Test compression and quantization utilities"""
    print("\n🧪 Testing compression utilities...")
    
    # Create test coefficients
    coeffs = torch.randn(32, 16) * 0.1
    
    # Test sparsification
    sparse_coeffs, mask = compress_fourier_weights(
        coeffs, sparsity_ratio=0.5, method="magnitude"
    )
    
    sparsity = (sparse_coeffs == 0).float().mean().item()
    print(f"✅ Sparsification: {sparsity:.2f} sparsity achieved")
    
    # Test quantization
    quant_coeffs, quant_params = quantize_fourier_weights(
        coeffs, bits=8, method="symmetric"
    )
    
    quant_error = torch.nn.functional.mse_loss(quant_coeffs, coeffs)
    print(f"✅ Quantization: MSE error {quant_error.item():.6f}")
    print(f"   Scale: {quant_params['scale']:.6f}")


def test_fourier_adapter():
    """Test FourierAdapter injection"""
    print("\n🧪 Testing FourierAdapter...")
    
    # Create simple model with linear layers
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = torch.nn.Linear(256, 256)
            self.k_proj = torch.nn.Linear(256, 256)
            self.v_proj = torch.nn.Linear(256, 256)
            self.output_proj = torch.nn.Linear(256, 128)
            
        def forward(self, x):
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)
            return self.output_proj(q + k + v)
    
    model = SimpleModel()
    
    # Test adapter creation
    adapter = FourierAdapter(
        in_features=256,
        out_features=256,
        compression_ratio=0.1,
        basis_type=BasisType.DCT
    )
    
    # Test forward pass
    x = torch.randn(4, 32, 256)
    original_output = model.q_proj(x)
    adapter_output = adapter(x)
    combined_output = original_output + adapter_output
    
    print(f"✅ Adapter forward pass: {adapter_output.shape}")
    print(f"✅ Combined output: {combined_output.shape}")
    
    # Test compression info
    ratio, orig, comp = adapter.fourier_layer.get_compression_ratio()
    print(f"✅ Adapter compression: {ratio:.4f} ({orig} -> {comp})")


def run_performance_benchmark():
    """Run performance comparison"""
    print("\n🚀 Performance Benchmark...")
    
    # Setup
    batch_size, seq_len, hidden_size = 4, 512, 1024
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Original linear layer
    linear = torch.nn.Linear(hidden_size, hidden_size).to(device)
    
    # Fourier approximation
    fourier_linear = FourierLinear(
        hidden_size, hidden_size, compression_ratio=0.05
    ).to(device)
    
    # Input
    x = torch.randn(batch_size, seq_len, hidden_size, device=device)
    
    # Warmup
    for _ in range(10):
        _ = linear(x)
        _ = fourier_linear(x)
    
    # Benchmark
    import time
    
    # Original
    torch.cuda.synchronize() if device == "cuda" else None
    start_time = time.time()
    for _ in range(100):
        _ = linear(x)
    torch.cuda.synchronize() if device == "cuda" else None
    linear_time = time.time() - start_time
    
    # Fourier
    torch.cuda.synchronize() if device == "cuda" else None
    start_time = time.time()
    for _ in range(100):
        _ = fourier_linear(x)
    torch.cuda.synchronize() if device == "cuda" else None
    fourier_time = time.time() - start_time
    
    print(f"✅ Original Linear:  {linear_time:.4f}s")
    print(f"✅ Fourier Linear:   {fourier_time:.4f}s")
    print(f"✅ Speed ratio:      {fourier_time/linear_time:.2f}x")
    
    # Memory usage
    linear_params = sum(p.numel() for p in linear.parameters())
    fourier_params = sum(p.numel() for p in fourier_linear.parameters())
    
    print(f"✅ Parameter ratio:  {fourier_params/linear_params:.4f}x")


def main():
    """Run all tests"""
    print("🎯 LLaVA-FA Fourier Components Test Suite")
    print("=" * 50)
    
    try:
        test_basis_factory()
        test_fourier_linear()  
        test_frequency_scheduler()
        test_compression_utils()
        test_fourier_adapter()
        run_performance_benchmark()
        
        print("\n" + "=" * 50)
        print("🎉 All tests completed successfully!")
        print("✨ LLaVA-FA components are working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
