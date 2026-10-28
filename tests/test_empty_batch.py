import pytest
import torch

from fast_hadamard_transform.fast_hadamard_transform_interface import (
    hadamard_transform,
    hadamard_transform_12N,
    hadamard_transform_20N,
    hadamard_transform_28N,
    hadamard_transform_40N,
)

TRANSFORMS = [
    (hadamard_transform, 137),
    (hadamard_transform_12N, 49),
    (hadamard_transform_20N, 81),
    (hadamard_transform_28N, 113),
    (hadamard_transform_40N, 161),
]


@pytest.mark.parametrize(("transform", "dim"), TRANSFORMS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rank", [2, 3])
def test_empty_batch(transform, dim, dtype, rank):
    shape = (0, dim) if rank == 2 else (2, 0, dim)
    x = torch.empty(shape, device="cuda", dtype=dtype, requires_grad=True)

    out = transform(x)

    assert out.shape == x.shape
    assert out.device == x.device
    assert out.dtype == x.dtype
    assert out.requires_grad
    out.sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert x.grad.device == x.device
    assert x.grad.dtype == x.dtype
    torch.cuda.synchronize()
