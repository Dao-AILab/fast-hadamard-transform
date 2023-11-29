import torch

from flash_attn.utils.benchmark import benchmark_forward, pytorch_profiler
from fast_hadamard_transform import hadamard_transform


batch_size = 16
seqlen = 2048
dim = 8192
dtype = torch.float16
device = "cuda"

torch.random.manual_seed(0)
x = torch.randn(batch_size, seqlen, dim, dtype=dtype, device=device)
benchmark_forward(hadamard_transform, x, desc="Hadamard transform")
pytorch_profiler(hadamard_transform, x)
benchmark_forward(torch.clone, x, desc="torch.clone")
pytorch_profiler(torch.clone, x)
