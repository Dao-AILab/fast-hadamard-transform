import runpy
import subprocess
from pathlib import Path

import setuptools
import torch.utils.cpp_extension


def capture_nvcc_args(monkeypatch, cuda_arch_list=None):
    captured = {}

    def capture_extension(*args, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(torch.utils.cpp_extension, "CUDA_HOME", "/usr/local/cuda")
    monkeypatch.setattr(torch.utils.cpp_extension, "CUDAExtension", capture_extension)
    monkeypatch.setattr(setuptools, "setup", lambda **kwargs: None)
    monkeypatch.setattr(
        subprocess,
        "check_output",
        lambda *args, **kwargs: "nvcc: NVIDIA Cuda compiler release 13.2, V13.2.51",
    )
    monkeypatch.setenv("FAST_HADAMARD_TRANSFORM_SKIP_CUDA_BUILD", "FALSE")
    if cuda_arch_list is None:
        monkeypatch.delenv("TORCH_CUDA_ARCH_LIST", raising=False)
    else:
        monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", cuda_arch_list)

    setup_path = Path(__file__).parents[1] / "setup.py"
    runpy.run_path(str(setup_path))
    return captured["extra_compile_args"]["nvcc"]


def test_default_cuda_architectures_remain_unchanged(monkeypatch):
    nvcc_args = capture_nvcc_args(monkeypatch)
    gencode = [
        nvcc_args[index + 1]
        for index, arg in enumerate(nvcc_args[:-1])
        if arg == "-gencode"
    ]

    assert gencode == [
        "arch=compute_75,code=sm_75",
        "arch=compute_80,code=sm_80",
        "arch=compute_87,code=sm_87",
        "arch=compute_90,code=sm_90",
        "arch=compute_100,code=sm_100",
        "arch=compute_120,code=sm_120",
        "arch=compute_103,code=sm_103",
        "arch=compute_110,code=sm_110",
        "arch=compute_121,code=sm_121",
    ]


def test_cuda_arch_list_is_left_to_pytorch(monkeypatch):
    nvcc_args = capture_nvcc_args(monkeypatch, "7.5;8.6+PTX")

    assert "-gencode" not in nvcc_args
    assert not any("arch=compute_" in arg for arg in nvcc_args)
    assert set(torch.utils.cpp_extension._get_cuda_arch_flags()) == {
        "-gencode=arch=compute_75,code=sm_75",
        "-gencode=arch=compute_86,code=sm_86",
        "-gencode=arch=compute_86,code=compute_86",
    }
