import torch

from flash_attn.utils.benchmark import benchmark_forward, pytorch_profiler
from fast_hadamard_transform import hadamard_transform
from fast_hadamard_transform.fast_hadamard_transform_interface import hadamard_transform_12N
from fast_hadamard_transform.fast_hadamard_transform_interface import hadamard_transform_20N
from fast_hadamard_transform.fast_hadamard_transform_interface import hadamard_transform_28N


batch_size = 16
seqlen = 2048
dim = 16384
dtype = torch.float16
device = "cuda"

torch.random.manual_seed(0)
x = torch.randn(batch_size, seqlen, dim, dtype=dtype, device=device)
benchmark_forward(hadamard_transform, x, desc="Hadamard transform")
pytorch_profiler(hadamard_transform, x)
benchmark_forward(torch.clone, x, desc="torch.clone")
pytorch_profiler(torch.clone, x)

dim = 12 * 512
x = torch.randn(batch_size, seqlen, dim, dtype=dtype, device=device)
benchmark_forward(hadamard_transform_12N, x, 1.0, desc="Hadamard transform 12N")
pytorch_profiler(hadamard_transform_12N, x, 1.0)

dim = 20 * 512
x = torch.randn(batch_size, seqlen, dim, dtype=dtype, device=device)
benchmark_forward(hadamard_transform_20N, x, 1.0, desc="Hadamard transform 20N")
pytorch_profiler(hadamard_transform_20N, x, 1.0)

dim = 28 * 512
x = torch.randn(batch_size, seqlen, dim, dtype=dtype, device=device)
benchmark_forward(hadamard_transform_28N, x, 1.0, desc="Hadamard transform 28N")
pytorch_profiler(hadamard_transform_28N, x, 1.0)
