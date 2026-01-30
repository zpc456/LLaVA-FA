"""
LLaVA-FA: Learning Fourier Approximation for Compressing Large Multimodal Models

A PyTorch implementation of frequency domain compression for large multimodal models.
Based on the paper: "LLaVA-FA: Learning Fourier Approximation for Compressing Large Multimodal Models"
"""

__version__ = "0.1.0"
__author__ = "LLaVA-FA Team"
__email__ = "contact@llava-fa.org"

from . import fourier
from . import model
from . import train
from . import constants
from . import conversation
from . import mm_utils
from . import utils

__all__ = [
    'fourier',
    'model', 
    'train',
    'constants',
    'conversation',
    'mm_utils',
    'utils'
]
