#pragma once

#if !defined(USE_MUSA) && !defined(USE_ROCM)
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#define WARP_SIZE 32
#define MIN(A, B) std::min((A), (B))
#elif defined(USE_MUSA)
#include <musa_bf16.h>
#include <musa_fp16.h>

#define WARP_SIZE 32
#define MIN(A, B) std::min((A), (B))
#define C10_CUDA_CHECK C10_MUSA_CHECK
#define C10_CUDA_KERNEL_LAUNCH_CHECK C10_MUSA_KERNEL_LAUNCH_CHECK
#define cudaFuncAttributeMaxDynamicSharedMemorySize musaFuncAttributeMaxDynamicSharedMemorySize
#define cudaFuncSetAttribute musaFuncSetAttribute
#define cudaStream_t musaStream_t

#include "torch_musa/csrc/core/MUSAGuard.h"
#include "torch_musa/csrc/core/MUSAStream.h"
namespace at {
namespace cuda {
#ifdef USE_MUSA
using CUDAGuard = at::musa::MUSAGuard;
inline at::musa::MUSAStream getCurrentCUDAStream() {
    return at::musa::getCurrentMUSAStream();
}
#endif
} // namespace cuda
} // namespace at
#elif defined(USE_ROCM)
#define WARP_SIZE 64
#define MIN(A, B) ((A) < (B) ? (A) : (B))
#define __shfl_xor_sync(MASK, X, OFFSET) __shfl_xor(X, OFFSET)
#endif // !defined(USE_MUSA) && !defined(USE_ROCM)
