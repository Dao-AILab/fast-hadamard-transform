# Copyright (C) 2023, Tri Dao.

import math

import torch
import pytest

from einops import rearrange, repeat

from fast_hadamard_transform.fast_hadamard_transform_interface import hadamard_transform, hadamard_transform_ref


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
# @pytest.mark.parametrize('dtype', [torch.float16])
@pytest.mark.parametrize("dim", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 137, 1024, 2048, 4096, 8192, 16384, 32768])
# @pytest.mark.parametrize("dim", [256])
def test_fast_hadamard_transform(dim, dtype):
    device = "cuda"
    rtol, atol = (3e-4, 3e-3) if dtype == torch.float32 else (3e-3, 5e-3)
    if dtype == torch.bfloat16:
        rtol, atol = 1e-2, 5e-2
    # set seed
    torch.random.manual_seed(0)
    batch_size = 15
    # batch_size = 1
    x = torch.randn(batch_size, dim, device=device, dtype=dtype).requires_grad_()
    x_ref = x.detach().clone().requires_grad_()
    x_pt = x.detach().clone().requires_grad_()
    scale = 1 / math.sqrt(dim)
    out = hadamard_transform(x, scale=scale)
    out_ref = hadamard_transform_ref(x_ref.float(), scale=scale)
    out_pt = hadamard_transform_ref(x_pt, scale=scale)

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
