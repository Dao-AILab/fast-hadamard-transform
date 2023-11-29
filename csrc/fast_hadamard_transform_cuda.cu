/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

// #pragma once

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/cuda/CUDAException.h>  // For C10_CUDA_CHECK and C10_CUDA_KERNEL_LAUNCH_CHECK

#include "fast_hadamard_transform.h"
#include "fast_hadamard_transform_common.h"
#include "static_switch.h"

#define FULL_MASK 0xffffffff

// https://stackoverflow.com/questions/35311711/whats-the-right-way-to-compute-integral-base-2-logarithms-at-compile-time
constexpr int cilog2(int val) { return val > 0 ? 1 + cilog2(val >> 1) : -1; }

template<int kNThreads_, int kLogN_, typename input_t_>
struct fast_hadamard_transform_kernel_traits {
    using input_t = input_t_;
    static constexpr int kNThreads = kNThreads_;
    static constexpr int kLogN = kLogN_;
    static constexpr int N = 1 << kLogN;
    static constexpr int kNBytes = sizeof(input_t);
    static_assert(kNBytes == 2 || kNBytes == 4);
    static constexpr int kNElts = kNBytes == 4 ? 4 : 8;
    // It's possible that we need to do 2 rounds of exchange if input_t is 16 bits
    // (since then we'd have 8 values of float, and each round we can exchange 4 floats).
    static constexpr int kNExchangePerVec = sizeof(float) / sizeof(input_t);
    using vec_t = typename BytesToType<kNBytes * kNElts>::Type;
    static constexpr int kNChunks = N / (kNElts * kNThreads);
    // We don't want to use more than 32 KB of shared memory.
    static constexpr int kSmemExchangeSize = std::min(N * 4, 32 * 1024);
    static constexpr int kNExchangeRounds = N * 4 / kSmemExchangeSize;
    static constexpr int kSmemSize = kSmemExchangeSize;
};


template<int kLogN, int kNChunks>
__device__ __forceinline__ void hadamard_mult_thread(float x[kNChunks][1 << kLogN]) {
    constexpr int N = 1 << kLogN;
    #pragma unroll
    for (int i = 0; i < kLogN; ++i) {
        const int stride = 1 << i;
        #pragma unroll
        for (int j = 0; j < N / 2; ++j) {
            const int lo = j & (stride - 1);
            const int idx = (j - lo) * 2 + lo;
            #pragma unroll
            for (int c = 0; c < kNChunks; ++c) {
                const float a = x[c][idx];
                const float b = x[c][idx + stride];
                x[c][idx] = a + b;
                x[c][idx + stride] = a - b;
            }
        }
    }
}

template<int kLogWarpSize, int kStepStart, int kNChunks, int kNItems>
__device__ __forceinline__ void hadamard_mult_warp(float x[kNChunks][kNItems]) {
    constexpr int N = 1 << kLogWarpSize;
    int lane_id = threadIdx.x % N;
    #pragma unroll
    for (int step = kStepStart; step < kLogWarpSize; ++step) {
        const int lane_mask = 1 << step;
        const float sign = (lane_id & lane_mask) ? -1.f : 1.f;
        #pragma unroll
        for (int c = 0; c < kNChunks; ++c) {
            #pragma unroll
            for (int i = 0; i < kNItems; ++i) {
                float x_val_other = __shfl_xor_sync(FULL_MASK, x[c][i], lane_mask);
                x[c][i] = sign * x[c][i] + x_val_other;
            }
        }
    }
}

template<typename Ktraits>
__global__ __launch_bounds__(Ktraits::kNThreads)
void fast_hadamard_transform_kernel(HadamardParamsBase params) {
    constexpr int kLogN = Ktraits::kLogN;
    constexpr int kNThreads = Ktraits::kNThreads;
    constexpr int kNElts = Ktraits::kNElts;
    constexpr int kNExchangePerVec = Ktraits::kNExchangePerVec;
    constexpr int kNExchangeRounds = Ktraits::kNExchangeRounds;
    constexpr int kNChunks = Ktraits::kNChunks;
    using input_t = typename Ktraits::input_t;
    using vec_t = typename Ktraits::vec_t;

    constexpr int kLogNElts = cilog2(Ktraits::kNElts);
    static_assert(1 << kLogNElts == kNElts, "kNElts must be a power of 2");
    constexpr int kWarpSize = std::min(kNThreads, 32);
    constexpr int kLogWarpSize = cilog2(kWarpSize);
    static_assert(1 << kLogWarpSize == kWarpSize, "Warp size must be a power of 2");
    constexpr int kNWarps = kNThreads / kWarpSize;
    constexpr int kLogNWarps = cilog2(kNWarps);
    static_assert(1 << kLogNWarps == kNWarps, "kNWarps must be a power of 2");
    constexpr int kLogNChunks = cilog2(kNChunks);
    static_assert(1 << kLogNChunks == kNChunks, "kNChunks must be a power of 2");
    constexpr int kLoadsPerExchange = Ktraits::kSmemExchangeSize / (sizeof(vec_t) * kNThreads);
    static_assert(kLoadsPerExchange * sizeof(vec_t) * kNThreads == Ktraits::kSmemExchangeSize, "kSmemExchangeSize should be a power of 2");
    static_assert(kNExchangeRounds * kLoadsPerExchange * sizeof(vec_t) == kNChunks * kNElts * sizeof(float));

    constexpr int kChunksPerExchange = Ktraits::kSmemExchangeSize / (sizeof(vec_t) * kNExchangePerVec * kNThreads);
    static_assert(kChunksPerExchange * sizeof(vec_t) * kNExchangePerVec * kNThreads == Ktraits::kSmemExchangeSize);
    constexpr int kNExchanges = kNChunks / kChunksPerExchange;
    static_assert(kNExchanges * kChunksPerExchange == kNChunks);

    // Shared memory.
    extern __shared__ char smem_[];
    vec_t *smem_exchange = reinterpret_cast<vec_t *>(smem_);

    const int tidx = threadIdx.x;
    const int batch_id = blockIdx.x;
    input_t *x = reinterpret_cast<input_t *>(params.x_ptr) + batch_id * params.x_batch_stride;
    input_t *out = reinterpret_cast<input_t *>(params.out_ptr) + batch_id * params.out_batch_stride;

    input_t x_vals_load[kNChunks][kNElts] = {0};
    #pragma unroll
    for (int c = 0; c < kNChunks; ++c) {
        if ((c * kNThreads + tidx) * kNElts < params.dim) {
            reinterpret_cast<vec_t*>(x_vals_load)[c] = reinterpret_cast<const vec_t*>(x)[c * kNThreads + tidx];
        }
    }
    float x_vals[kNChunks][kNElts];
    #pragma unroll
    for (int c = 0; c < kNChunks; ++c) {
        #pragma unroll
        for (int i = 0; i < kNElts; ++i) { x_vals[c][i] = float(x_vals_load[c][i]); }
    }

    hadamard_mult_thread<kLogNElts, kNChunks>(x_vals);
    hadamard_mult_warp<kLogWarpSize, 0, kNChunks, kNElts>(x_vals);

    if constexpr (kNWarps > 1) {
        // Exchange data between warps.
        const int warp_id = tidx / kWarpSize;
        const int lane_id = tidx % kWarpSize;
        const int row_t = tidx % kNWarps;
        const int col_t = tidx / kNWarps;
        // We use the XOR trick (new_col = col ^ row) to avoid / reduce smem bank conflicts.
        #pragma unroll
        for (int c0 = 0; c0 < kNChunks / kChunksPerExchange; ++c0) {
            __syncthreads();
            #pragma unroll
            for (int c1 = 0; c1 < kChunksPerExchange; ++c1) {
                #pragma unroll
                for (int r = 0; r < kNExchangePerVec; ++r) {
                    smem_exchange[(c1 * kNExchangePerVec + r) * kNThreads + warp_id * kWarpSize + lane_id ^ warp_id] = reinterpret_cast<vec_t*>(x_vals[c0 * kChunksPerExchange + c1])[r];
                }
            }
            __syncthreads();
            #pragma unroll
            for (int c1 = 0; c1 < kChunksPerExchange; ++c1) {
                #pragma unroll
                for (int r = 0; r < kNExchangePerVec; ++r) {
                    reinterpret_cast<vec_t*>(x_vals[c0 * kChunksPerExchange + c1])[r] = smem_exchange[(c1 * kNExchangePerVec + r) * kNThreads + row_t * kWarpSize + col_t ^ row_t];
                }
            }
        }
        hadamard_mult_warp<kLogNWarps, 0, kNChunks, kNElts>(x_vals);
        #pragma unroll
        for (int c0 = 0; c0 < kNChunks / kChunksPerExchange; ++c0) {
            __syncthreads();
            #pragma unroll
            for (int c1 = 0; c1 < kChunksPerExchange; ++c1) {
                #pragma unroll
                for (int r = 0; r < kNExchangePerVec; ++r) {
                    smem_exchange[(c1 * kNExchangePerVec + r) * kNThreads + row_t * kWarpSize + col_t ^ row_t] = reinterpret_cast<vec_t*>(x_vals[c0 * kChunksPerExchange + c1])[r];
                }
            }
            __syncthreads();
            #pragma unroll
            for (int c1 = 0; c1 < kChunksPerExchange; ++c1) {
                #pragma unroll
                for (int r = 0; r < kNExchangePerVec; ++r) {
                    reinterpret_cast<vec_t*>(x_vals[c0 * kChunksPerExchange + c1])[r] = smem_exchange[(c1 * kNExchangePerVec + r) * kNThreads + warp_id * kWarpSize + lane_id ^ warp_id];
                }
            }
        }
    }

    if constexpr (kNChunks > 1) {
        float x_vals_transposed[kNElts][kNChunks];
        #pragma unroll
        for (int c = 0; c < kNChunks; ++c) {
            #pragma unroll
            for (int i = 0; i < kNElts; ++i) { x_vals_transposed[i][c] = x_vals[c][i]; }
        }
        hadamard_mult_thread<kLogNChunks, kNElts>(x_vals_transposed);
        #pragma unroll
        for (int c = 0; c < kNChunks; ++c) {
            #pragma unroll
            for (int i = 0; i < kNElts; ++i) { x_vals[c][i] = x_vals_transposed[i][c]; }
        }
    }

    input_t out_vals_store[kNChunks][kNElts];
    #pragma unroll
    for (int c = 0; c < kNChunks; ++c) {
        #pragma unroll
        for (int i = 0; i < kNElts; ++i) { out_vals_store[c][i] = x_vals[c][i] * params.scale; }
    }
    #pragma unroll
    for (int c = 0; c < kNChunks; ++c) {
        if ((c * kNThreads + tidx) * kNElts < params.dim) {
            reinterpret_cast<vec_t*>(out)[c * kNThreads + tidx] = reinterpret_cast<vec_t*>(out_vals_store)[c];
        }
    }
}

template<int kNThreads, int kLogN, typename input_t>
void fast_hadamard_transform_fwd_launch(HadamardParamsBase &params, cudaStream_t stream) {
    using Ktraits = fast_hadamard_transform_kernel_traits<kNThreads, kLogN, input_t>;
    constexpr int kSmemSize = Ktraits::kSmemSize;
    dim3 grid(params.batch);
    auto kernel = &fast_hadamard_transform_kernel<Ktraits>;
    if (kSmemSize >= 48 * 1024) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
        }
    kernel<<<grid, Ktraits::kNThreads, kSmemSize, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<typename input_t>
void fast_hadamard_transform_cuda(HadamardParamsBase &params, cudaStream_t stream) {
    if (params.log_dim == 3) {
        fast_hadamard_transform_fwd_launch<1, 3, input_t>(params, stream);
    } else if (params.log_dim == 4) {
        fast_hadamard_transform_fwd_launch<2, 4, input_t>(params, stream);
    } else if (params.log_dim == 5) {
        fast_hadamard_transform_fwd_launch<4, 5, input_t>(params, stream);
    } else if (params.log_dim == 6) {
        fast_hadamard_transform_fwd_launch<8, 6, input_t>(params, stream);
    } else if (params.log_dim == 7) {
        fast_hadamard_transform_fwd_launch<16, 7, input_t>(params, stream);
    } else if (params.log_dim == 8) {
        fast_hadamard_transform_fwd_launch<32, 8, input_t>(params, stream);
    } else if (params.log_dim == 9) {
        fast_hadamard_transform_fwd_launch<32, 9, input_t>(params, stream);
    } else if (params.log_dim == 10) {
        fast_hadamard_transform_fwd_launch<128, 10, input_t>(params, stream);
    } else if (params.log_dim == 11) {
        fast_hadamard_transform_fwd_launch<256, 11, input_t>(params, stream);
    } else if (params.log_dim == 12) {
        fast_hadamard_transform_fwd_launch<256, 12, input_t>(params, stream);
    } else if (params.log_dim == 13) {
        fast_hadamard_transform_fwd_launch<256, 13, input_t>(params, stream);
    } else if (params.log_dim == 14) {
        fast_hadamard_transform_fwd_launch<256, 14, input_t>(params, stream);
    } else if (params.log_dim == 15) {
        fast_hadamard_transform_fwd_launch<256, 15, input_t>(params, stream);
    }
}

template void fast_hadamard_transform_cuda<float>(HadamardParamsBase &params, cudaStream_t stream);
template void fast_hadamard_transform_cuda<at::Half>(HadamardParamsBase &params, cudaStream_t stream);
template void fast_hadamard_transform_cuda<at::BFloat16>(HadamardParamsBase &params, cudaStream_t stream);