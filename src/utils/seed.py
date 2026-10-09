from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Set deterministic seeds for Python, NumPy, and PyTorch when available.

    cudnn.deterministic alone does not cover everything used by CTGEncoder's
    nn.TransformerEncoderLayer: scaled-dot-product-attention's flash/memory-
    efficient CUDA backends and cuBLAS GEMM reduction order are separate
    nondeterminism sources. Without disabling those, two runs with the same
    seed on GPU can diverge by several hundredths of AUC after 40 epochs of
    training (observed directly when comparing two separate CUDA runs of the
    same config). CUBLAS_WORKSPACE_CONFIG must be set before the first CUDA
    context is created, so set it here as early as possible.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(False)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(False)
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            torch.backends.cuda.enable_math_sdp(True)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
