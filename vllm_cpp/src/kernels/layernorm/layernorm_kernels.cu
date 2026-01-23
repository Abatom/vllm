// SPDX-License-Identifier: Apache-2.0
// vLLM C++ Implementation - LayerNorm CUDA Kernels
// Adapted from vLLM csrc/layernorm_kernels.cu

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <torch/torch.h>
#include <ATen/cuda/CUDAContext.h>

#include <cmath>

namespace vllm {
namespace kernels {

// Warp reduce sum
template<typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block reduce sum
template<typename T>
__device__ __forceinline__ T block_reduce_sum(T val, T* shared) {
    const int lane = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;
    
    val = warp_reduce_sum(val);
    
    if (lane == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();
    
    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : static_cast<T>(0);
    
    if (warp_id == 0) {
        val = warp_reduce_sum(val);
    }
    
    return val;
}

// RMS Norm kernel
template<typename scalar_t>
__global__ void rms_norm_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const float epsilon,
    const int num_tokens,
    const int hidden_size
) {
    extern __shared__ float s_variance[];
    
    const int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;
    
    const int input_offset = token_idx * hidden_size;
    
    // Compute variance
    float variance = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float x = static_cast<float>(input[input_offset + i]);
        variance += x * x;
    }
    
    variance = block_reduce_sum(variance, s_variance);
    
    if (threadIdx.x == 0) {
        s_variance[0] = rsqrtf(variance / hidden_size + epsilon);
    }
    __syncthreads();
    
    float rsqrt_var = s_variance[0];
    
    // Apply normalization
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float x = static_cast<float>(input[input_offset + i]);
        float w = static_cast<float>(weight[i]);
        output[input_offset + i] = static_cast<scalar_t>(x * rsqrt_var * w);
    }
}

// Fused Add + RMS Norm kernel
template<typename scalar_t>
__global__ void fused_add_rms_norm_kernel(
    scalar_t* __restrict__ input,
    scalar_t* __restrict__ residual,
    const scalar_t* __restrict__ weight,
    const float epsilon,
    const int num_tokens,
    const int hidden_size
) {
    extern __shared__ float s_variance[];
    
    const int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;
    
    const int offset = token_idx * hidden_size;
    
    // First pass: add residual and compute variance
    float variance = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float x = static_cast<float>(input[offset + i]);
        float r = static_cast<float>(residual[offset + i]);
        float sum = x + r;
        residual[offset + i] = static_cast<scalar_t>(sum);
        variance += sum * sum;
    }
    
    variance = block_reduce_sum(variance, s_variance);
    
    if (threadIdx.x == 0) {
        s_variance[0] = rsqrtf(variance / hidden_size + epsilon);
    }
    __syncthreads();
    
    float rsqrt_var = s_variance[0];
    
    // Second pass: apply normalization
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float x = static_cast<float>(residual[offset + i]);
        float w = static_cast<float>(weight[i]);
        input[offset + i] = static_cast<scalar_t>(x * rsqrt_var * w);
    }
}

// Dispatch functions
void rms_norm(
    torch::Tensor& output,
    const torch::Tensor& input,
    const torch::Tensor& weight,
    float epsilon
) {
    int num_tokens = input.numel() / input.size(-1);
    int hidden_size = input.size(-1);
    
    dim3 grid(num_tokens);
    dim3 block(std::min(hidden_size, 1024));
    int shared_mem_size = (block.x / 32 + 1) * sizeof(float);
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        input.scalar_type(), "rms_norm_kernel", ([&] {
            rms_norm_kernel<scalar_t><<<grid, block, shared_mem_size, stream>>>(
                output.data_ptr<scalar_t>(),
                input.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(),
                epsilon,
                num_tokens,
                hidden_size
            );
        })
    );
}

void fused_add_rms_norm(
    torch::Tensor& input,
    torch::Tensor& residual,
    const torch::Tensor& weight,
    float epsilon
) {
    int num_tokens = input.numel() / input.size(-1);
    int hidden_size = input.size(-1);
    
    dim3 grid(num_tokens);
    dim3 block(std::min(hidden_size, 1024));
    int shared_mem_size = (block.x / 32 + 1) * sizeof(float);
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        input.scalar_type(), "fused_add_rms_norm_kernel", ([&] {
            fused_add_rms_norm_kernel<scalar_t><<<grid, block, shared_mem_size, stream>>>(
                input.data_ptr<scalar_t>(),
                residual.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(),
                epsilon,
                num_tokens,
                hidden_size
            );
        })
    );
}

}  // namespace kernels
}  // namespace vllm
