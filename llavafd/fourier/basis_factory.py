"""
BasisFactory: Generate and cache frequency domain bases
Supports DCT, DST, and Fourier bases for efficient computation
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math
from enum import Enum
import weakref


class BasisType(Enum):
    """Types of frequency domain bases"""
    DCT = "dct"  # Discrete Cosine Transform
    DST = "dst"  # Discrete Sine Transform  
    FOURIER = "fourier"  # Complex Fourier (real/imaginary parts)
    MIXED = "mixed"  # Mixed cosine/sine


class BasisFactory:
    """
    Factory for creating and caching frequency domain bases
    
    Provides efficient generation and caching of orthogonal frequency bases
    optimized for weight matrix approximation in neural networks.
    """
    
    # Global cache shared across instances
    _basis_cache: Dict[Tuple[str, int, int], torch.Tensor] = {}
    _cache_refs: weakref.WeakSet = weakref.WeakSet()
    
    def __init__(
        self, 
        basis_type: BasisType = BasisType.DCT,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        normalize: bool = True
    ):
        """
        Initialize basis factory
        
        Args:
            basis_type: Type of frequency basis to generate
            device: Device to place bases on
            dtype: Data type for bases  
            normalize: Whether to normalize basis vectors
        """
        self.basis_type = basis_type
        self.device = device or torch.device('cpu')
        self.dtype = dtype or torch.float32
        self.normalize = normalize
        
        # Register this instance for cache management
        self._cache_refs.add(self)
    
    def get_basis(self, n_dim: int, k_components: int) -> torch.Tensor:
        """
        Get frequency domain basis matrix
        
        Args:
            n_dim: Dimension of original space
            k_components: Number of frequency components to keep
            
        Returns:
            Basis matrix of shape (n_dim, k_components)
        """
        # Create cache key
        cache_key = (self.basis_type.value, n_dim, k_components)
        
        # Check cache first
        if cache_key in self._basis_cache:
            cached_basis = self._basis_cache[cache_key]
            # Move to correct device/dtype if needed
            if cached_basis.device != self.device or cached_basis.dtype != self.dtype:
                basis = cached_basis.to(device=self.device, dtype=self.dtype)
                self._basis_cache[cache_key] = basis
                return basis
            return cached_basis
        
        # Generate new basis
        basis = self._generate_basis(n_dim, k_components)
        
        # Cache the result
        self._basis_cache[cache_key] = basis
        
        return basis
    
    def _generate_basis(self, n_dim: int, k_components: int) -> torch.Tensor:
        """Generate frequency domain basis matrix"""
        k_components = min(k_components, n_dim)  # Can't have more components than dimensions
        
        if self.basis_type == BasisType.DCT:
            basis = self._generate_dct_basis(n_dim, k_components)
        elif self.basis_type == BasisType.DST:
            basis = self._generate_dst_basis(n_dim, k_components)
        elif self.basis_type == BasisType.FOURIER:
            basis = self._generate_fourier_basis(n_dim, k_components)
        elif self.basis_type == BasisType.MIXED:
            basis = self._generate_mixed_basis(n_dim, k_components)
        else:
            raise ValueError(f"Unsupported basis type: {self.basis_type}")
        
        return basis.to(device=self.device, dtype=self.dtype)
    
    def _generate_dct_basis(self, n_dim: int, k_components: int) -> torch.Tensor:
        """
        Generate DCT-II basis (most common for signal processing)
        
        DCT-II: X_k = sum_{n=0}^{N-1} x_n * cos(π*k*(2n+1)/(2N))
        """
        # Create indices
        n = torch.arange(n_dim, dtype=torch.float32).unsqueeze(1)  # (n_dim, 1)
        k = torch.arange(k_components, dtype=torch.float32).unsqueeze(0)  # (1, k_components)
        
        # DCT-II formula
        basis = torch.cos(math.pi * k * (2 * n + 1) / (2 * n_dim))
        
        # Normalize
        if self.normalize:
            # DC component (k=0) has different normalization
            norm = torch.ones(k_components)
            norm[0] = 1.0 / math.sqrt(2)
            basis = basis * norm.unsqueeze(0) * math.sqrt(2.0 / n_dim)
        
        return basis
    
    def _generate_dst_basis(self, n_dim: int, k_components: int) -> torch.Tensor:
        """
        Generate DST-II basis 
        
        DST-II: X_k = sum_{n=0}^{N-1} x_n * sin(π*k*(2n+1)/(2N))
        """
        # Create indices  
        n = torch.arange(n_dim, dtype=torch.float32).unsqueeze(1)  # (n_dim, 1)
        k = torch.arange(1, k_components + 1, dtype=torch.float32).unsqueeze(0)  # (1, k_components)
        
        # DST-II formula
        basis = torch.sin(math.pi * k * (2 * n + 1) / (2 * n_dim))
        
        # Normalize
        if self.normalize:
            basis = basis * math.sqrt(2.0 / n_dim)
            
        return basis
    
    def _generate_fourier_basis(self, n_dim: int, k_components: int) -> torch.Tensor:
        """
        Generate real-valued Fourier basis
        Uses both cosine and sine components
        """
        # For real-valued output, we use both cos and sin
        # k_components should be even for balanced cos/sin
        k_cos = (k_components + 1) // 2
        k_sin = k_components // 2
        
        bases = []
        
        # Cosine components (including DC)
        if k_cos > 0:
            cos_basis = self._generate_dct_basis(n_dim, k_cos)
            bases.append(cos_basis)
        
        # Sine components (excluding DC which is 0)
        if k_sin > 0:
            sin_basis = self._generate_dst_basis(n_dim, k_sin)
            bases.append(sin_basis)
        
        basis = torch.cat(bases, dim=1)
        
        # Take only requested number of components
        return basis[:, :k_components]
    
    def _generate_mixed_basis(self, n_dim: int, k_components: int) -> torch.Tensor:
        """Generate mixed cosine/sine basis with custom ordering"""
        # Alternate between cosine and sine components
        bases = []
        
        for i in range(k_components):
            if i % 2 == 0:  # Even indices: cosine
                k_idx = i // 2
                if k_idx == 0:
                    # DC component
                    basis_vec = torch.ones(n_dim) / math.sqrt(n_dim)
                else:
                    n = torch.arange(n_dim, dtype=torch.float32)
                    basis_vec = torch.cos(2 * math.pi * k_idx * n / n_dim)
                    if self.normalize:
                        basis_vec = basis_vec / math.sqrt(n_dim / 2)
            else:  # Odd indices: sine
                k_idx = (i + 1) // 2
                n = torch.arange(n_dim, dtype=torch.float32)
                basis_vec = torch.sin(2 * math.pi * k_idx * n / n_dim)
                if self.normalize:
                    basis_vec = basis_vec / math.sqrt(n_dim / 2)
                    
            bases.append(basis_vec.unsqueeze(1))
        
        return torch.cat(bases, dim=1)
    
    def get_2d_basis(
        self, 
        height: int, 
        width: int,
        k_h: int,
        k_w: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get 2D separable basis for blockwise approximation
        
        For approximating matrices reshaped to (height, width),
        returns row and column bases for separable 2D transform.
        
        Args:
            height: Matrix height after reshaping
            width: Matrix width after reshaping  
            k_h: Number of frequency components in height
            k_w: Number of frequency components in width
            
        Returns:
            Row basis (height, k_h), column basis (width, k_w)
        """
        row_basis = self.get_basis(height, k_h)
        col_basis = self.get_basis(width, k_w)
        
        return row_basis, col_basis
    
    def compute_basis_correlation(self, basis1: torch.Tensor, basis2: torch.Tensor) -> torch.Tensor:
        """
        Compute correlation between two basis sets
        Useful for analyzing basis similarity and redundancy
        """
        # Normalize bases
        basis1_norm = F.normalize(basis1, dim=0)
        basis2_norm = F.normalize(basis2, dim=0)
        
        # Compute correlation matrix
        correlation = basis1_norm.T @ basis2_norm
        
        return correlation.abs()
    
    def get_frequency_indices(self, k_components: int) -> torch.Tensor:
        """
        Get frequency indices corresponding to basis components
        Useful for frequency-aware regularization and scheduling
        
        Returns:
            Frequency indices from low to high
        """
        if self.basis_type == BasisType.DCT:
            # DCT frequencies: 0, 1, 2, ..., k-1
            return torch.arange(k_components, dtype=torch.float32)
        elif self.basis_type == BasisType.DST:
            # DST frequencies: 1, 2, 3, ..., k
            return torch.arange(1, k_components + 1, dtype=torch.float32)
        elif self.basis_type == BasisType.FOURIER:
            # Mixed frequencies: 0, 1, 1, 2, 2, 3, 3, ...
            indices = []
            for i in range(k_components):
                if i % 2 == 0:
                    indices.append(i // 2)
                else:
                    indices.append((i + 1) // 2)
            return torch.tensor(indices, dtype=torch.float32)
        elif self.basis_type == BasisType.MIXED:
            # Similar to Fourier
            return self.get_frequency_indices(k_components)
        else:
            return torch.arange(k_components, dtype=torch.float32)
    
    @staticmethod
    def clear_cache():
        """Clear the global basis cache"""
        BasisFactory._basis_cache.clear()
    
    @staticmethod 
    def get_cache_info() -> Dict:
        """Get information about cache usage"""
        return {
            'cache_size': len(BasisFactory._basis_cache),
            'cache_keys': list(BasisFactory._basis_cache.keys()),
            'active_instances': len(BasisFactory._cache_refs)
        }
    
    def __del__(self):
        """Clean up when factory is deleted"""
        # Remove from weak reference set
        try:
            self._cache_refs.discard(self)
        except:
            pass


class BlockwiseBasisFactory(BasisFactory):
    """
    Factory for blockwise/2D frequency domain approximation
    
    For very large weight matrices, applies 2D frequency transform
    by reshaping the matrix and using separable 2D basis.
    """
    
    def __init__(
        self,
        basis_type: BasisType = BasisType.DCT,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        block_size: Tuple[int, int] = (64, 64)
    ):
        super().__init__(basis_type, device, dtype)
        self.block_size = block_size
    
    def get_blockwise_bases(
        self, 
        matrix_shape: Tuple[int, int],
        compression_ratio: float = 0.05
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
        """
        Get bases for blockwise approximation
        
        Args:
            matrix_shape: Shape of matrix to approximate (M, N)
            compression_ratio: Target compression ratio
            
        Returns:
            Row basis, column basis, and block dimensions
        """
        M, N = matrix_shape
        
        # Determine block dimensions
        block_h = min(self.block_size[0], M)
        block_w = min(self.block_size[1], N)
        
        # Calculate frequency components based on compression ratio
        total_block_params = block_h * block_w
        k_h = max(1, int(block_h * math.sqrt(compression_ratio)))
        k_w = max(1, int(block_w * math.sqrt(compression_ratio)))
        
        # Ensure we don't exceed block dimensions
        k_h = min(k_h, block_h)
        k_w = min(k_w, block_w)
        
        # Get 2D bases
        row_basis, col_basis = self.get_2d_basis(block_h, block_w, k_h, k_w)
        
        return row_basis, col_basis, (k_h, k_w)
