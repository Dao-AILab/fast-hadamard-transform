import pytest
import torch

from fast_hadamard_transform.fast_hadamard_transform_interface import (
    hadamard_transform,
    hadamard_transform_12N,
    hadamard_transform_20N,
    hadamard_transform_28N,
    hadamard_transform_40N,
)


def test_backward_with_misaligned_gradient():
    x = torch.randn(2, 8, device="cuda", dtype=torch.float16, requires_grad=True)
    grad = torch.randn(2, 12, device="cuda", dtype=x.dtype)[:, :8]
    scale = 0.25
    expected = hadamard_transform(
        grad.clone(memory_format=torch.contiguous_format), scale
    )

    hadamard_transform(x, scale).backward(grad)
    torch.cuda.synchronize()

    torch.testing.assert_close(x.grad, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["strided_rows", "storage_offset", "aligned_rows"])
@pytest.mark.parametrize(
    ("transform", "dim"),
    [
        (hadamard_transform, 8),
        (hadamard_transform, 24),
        (hadamard_transform_12N, 48),
        (hadamard_transform_20N, 80),
        (hadamard_transform_28N, 112),
        (hadamard_transform_40N, 160),
    ],
)
def test_input_alignment(transform, dim, dtype, layout):
    if layout == "storage_offset":
        x = torch.randn(2 * dim + 1, device="cuda", dtype=dtype)[1:].view(2, dim)
    else:
        padding = 1 if layout == "strided_rows" else 8
        x = torch.randn(2, dim + padding, device="cuda", dtype=dtype)[:, :dim]
    scale = 0.25
    expected = transform(x.clone(memory_format=torch.contiguous_format), scale)

    actual = transform(x, scale)
    torch.cuda.synchronize()

    torch.testing.assert_close(actual, expected)
