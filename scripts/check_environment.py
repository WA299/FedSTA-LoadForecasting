"""Verify Python, installed packages and a CUDA forward/backward calculation."""

import importlib.metadata
import platform
import subprocess
import sys
from pathlib import Path

import torch

from src.training.engine import write_json


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.ones((8, 8), device=device, requires_grad=True)
    (x @ x).sum().backward()
    if not torch.isfinite(x.grad).all():
        raise RuntimeError("PyTorch forward/backward verification failed")
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True, check=False).stdout.strip()
    report = {"python": platform.python_version(), "executable": sys.executable,
              "torch": torch.__version__, "torch_cuda": torch.version.cuda,
              "cuda_available": torch.cuda.is_available(), "device": device,
              "gpu": gpu, "forward_backward_passed": True,
              "packages": {name: importlib.metadata.version(name) for name in
                           ("torch", "numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "tqdm", "PyYAML", "pytest")}}
    Path("results/g1").mkdir(parents=True, exist_ok=True)
    write_json("results/g1/environment.json", report)
    print(report, flush=True)


if __name__ == "__main__":
    main()
