# Copyright (C) 2023, Tri Dao.

import math

import pytest
import torch
import torch.nn.functional as F

from fast_hadamard_transform.fast_hadamard_transform_interface import (
    hadamard_transform,
    hadamard_transform_ref,
)


def hadamard_reference(x, scale=1.0):
    dim = x.shape[-1]
    padded_dim = 1 << (dim - 1).bit_length()
    # Accumulate in float64 so the oracle does not share the CUDA kernel's rounding.
    values = F.pad(x.to(torch.float64), (0, padded_dim - dim))
    width = 1
    while width < padded_dim:
        pairs = values.reshape(*x.shape[:-1], -1, 2, width)
        left, right = pairs.unbind(dim=-2)
        values = torch.cat((left + right, left - right), dim=-1)
        width *= 2
    return values.reshape(*x.shape[:-1], padded_dim)[..., :dim].to(x.dtype) * scale


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dim", [1, 8, 17])
def test_hadamard_reference(dim, dtype):
    torch.random.manual_seed(0)
    x = torch.randn(2, 3, dim, dtype=dtype, requires_grad=True)
    x_dense = x.detach().clone().requires_grad_()
    scale = 1 / math.sqrt(dim)
    actual = hadamard_reference(x, scale)
    expected = hadamard_transform_ref(x_dense, scale)
    torch.testing.assert_close(actual, expected)

    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)
    torch.testing.assert_close(x.grad, x_dense.grad)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
# @pytest.mark.parametrize('dtype', [torch.float16])
@pytest.mark.parametrize("dim", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 137, 1024, 2048, 4096, 8192, 16384, 32768])
# @pytest.mark.parametrize("dim", [256])
def test_fast_hadamard_transform(dim, dtype):
    device = "cuda"
    atol = 3e-3 if dtype == torch.float32 else 5e-3
    if dtype == torch.bfloat16:
        atol = 5e-2
    # set seed
    torch.random.manual_seed(0)
    batch_size = 15
    # batch_size = 1
    x = torch.randn(batch_size, dim, device=device, dtype=dtype).requires_grad_()
    x_ref = x.detach().clone().requires_grad_()
    x_pt = x.detach().clone().requires_grad_()
    scale = 1 / math.sqrt(dim)
    out = hadamard_transform(x, scale=scale)
    out_ref = hadamard_reference(x_ref.float(), scale=scale)
    out_pt = hadamard_reference(x_pt, scale=scale)

    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    print(f"Output Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
    print(f"Output Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}")
    assert (out - out_ref).abs().max().item() < 2 * (out_pt - out_ref).abs().max() + atol

    g = torch.randn_like(out)
    out.backward(g)
    out_ref.backward(g)
    out_pt.backward(g)

    print(f"dx max diff: {(x.grad - x_ref.grad).abs().max().item()}")
    print(f"dx Pytorch max diff: {(x_pt.grad - x_ref.grad).abs().max().item()}")
    assert (x.grad - x_ref.grad).abs().max().item() < 2 * (x_pt.grad - x_ref.grad).abs().max() + atol
