# LLaVA-FA: Learning Fourier Approximation for Compressing Large Multimodal Models

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-orange" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-Apache%202.0-green" alt="License">
</div>

---

## 🎯 Overview

Large multimodal models (LMMs) have achieved impressive performance on various vision-language tasks, but their substantial computational and memory costs hinder their practical deployment. Existing compression methods often decouple low-rank decomposition and quantization, leading to compounded reconstruction errors, especially in multimodal architectures with cross-modal redundancy. To address this issue, we propose LLaVA-FA, a novel efficient LMM that performs joint low-rank plus quantization approximation in the frequency domain. By leveraging the de-correlation and conjugate symmetry properties of Fourier transform, LLaVA-FA achieves more compact and accurate weight representations. Furthermore, we introduce PolarQuant, a polar-coordinate quantization method tailored for complex matrices, and an optional diagonal calibration (ODC) scheme that eliminates the need for large-scale calibration data. Extensive experimental results demonstrate that our proposed LLaVA-FA outperforms existing efficient multimodal models across multiple benchmarks while maintaining minimal activated parameters and low computational costs, validating its effectiveness as a powerful solution for compressing LMMs.


---

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/your-repo/LLaVA-FA.git
cd LLaVA-FA

# Create conda environment
conda create -n llava-fa python=3.8
conda activate llava-fa

# Install dependencies
pip install -r requirements.txt

# Install flash-attention (optional, for better performance)
pip install flash-attn --no-build-isolation
```

---


## 📚 Citation

If you use LLaVA-FA in your research, please cite:

```bibtex
@article{llava_fa_2026,
  title={LLaVA-FA: Learning Fourier Approximation for Compressing Large Multimodal Models},
  author={Pengcheng Zheng, Chaoning Zhang, Jiarong Mo, GuoHui Li, Jiaquan Zhang, Jiahao Zhang, Sihan Cao, Sheng Zheng, Caiyan Qin, Guoqing Wang, Yang Yang},
  conference={ICLR},
  year={2026}
}
```

---



## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [LLaVA](https://github.com/haotian-liu/LLaVA) for the foundational multimodal framework
- [LLaVA-MoD](https://github.com/shufangxun/LLaVA-MoD) for architectural inspiration

---
