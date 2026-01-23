// SPDX-License-Identifier: Apache-2.0
// vLLM C++ Implementation - Activation CUDA Kernels
// Adapted from vLLM csrc/activation_kernels.cu

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <torch/torch.h>
#include <ATen/cuda/CUDAContext.h>

#include <cmath>

namespace vllm {
namespace kernels {

// SiLU activation: x * sigmoid(x)
template<typename T>
__device__ __forceinline__ T silu(const T& x) {
    return x / (static_cast<T>(1.0f) + exp(-x));
}

template<>
__device__ __forceinline__ __half silu(const __half& x) {
    float fx = __half2float(x);
    return __float2half(fx / (1.0f + expf(-fx)));
}

template<>
__device__ __forceinline__ __nv_bfloat16 silu(const __nv_bfloat16& x) {
    float fx = __bfloat162float(x);
    return __float2bfloat16(fx / (1.0f + expf(-fx)));
}

// GELU activation (tanh approximation)
template<typename T>
__device__ __forceinline__ T gelu_tanh(const T& x) {
    const float kBeta = 0.7978845608f;  // sqrt(2/pi)
    const float kKappa = 0.044715f;
    float fx = static_cast<float>(x);
    float y = fx + kKappa * fx * fx * fx;
    return static_cast<T>(0.5f * fx * (1.0f + tanhf(kBeta * y)));
}

// SiLU and Mul kernel
template<typename scalar_t>
__global__ void silu_and_mul_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int num_tokens,
    const int hidden_size
) {
    const int token_idx = blockIdx.x;
    const int idx = threadIdx.x;
    
    if (token_idx >= num_tokens) return;
    
    const int input_offset = token_idx * hidden_size * 2;
    const int output_offset = token_idx * hidden_size;
    
    for (int i = idx; i < hidden_size; i += blockDim.x) {
        scalar_t gate = input[input_offset + i];
        scalar_t up = input[input_offset + hidden_size + i];
        output[output_offset + i] = silu(gate) * up;
    }
}

// GELU and Mul kernel
template<typename scalar_t>
__global__ void gelu_and_mul_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int num_tokens,
    const int hidden_size
) {
    const int token_idx = blockIdx.x;
    const int idx = threadIdx.x;
    
    if (token_idx >= num_tokens) return;
    
    const int input_offset = token_idx * hidden_size * 2;
    const int output_offset = token_idx * hidden_size;
    
    for (int i = idx; i < hidden_size; i += blockDim.x) {
        scalar_t gate = input[input_offset + i];
        scalar_t up = input[input_offset + hidden_size + i];
        output[output_offset + i] = gelu_tanh(gate) * up;
    }
}

// FATReLU and Mul kernel
template<typename scalar_t>
__global__ void fatrelu_and_mul_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const float threshold,
    const int num_tokens,
    const int hidden_size
) {
    const int token_idx = blockIdx.x;
    const int idx = threadIdx.x;
    
    if (token_idx >= num_tokens) return;
    
    const int input_offset = token_idx * hidden_size * 2;
    const int output_offset = token_idx * hidden_size;
    
    for (int i = idx; i < hidden_size; i += blockDim.x) {
        float gate = static_cast<float>(input[input_offset + i]);
        scalar_t up = input[input_offset + hidden_size + i];
        float relu_out = fmaxf(gate, 0.0f);
        relu_out = (relu_out > threshold) ? relu_out : 0.0f;
        output[output_offset + i] = static_cast<scalar_t>(relu_out) * up;
    }
}

// Dispatch functions
void silu_and_mul(
    torch::Tensor& output,
    const torch::Tensor& input
) {
    int num_tokens = input.size(0);
    int hidden_size = input.size(-1) / 2;
    
    dim3 grid(num_tokens);
    dim3 block(std::min(hidden_size, 1024));
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        input.scalar_type(), "silu_and_mul_kernel", ([&] {
            silu_and_mul_kernel<scalar_t><<<grid, block, 0, stream>>>(
                output.data_ptr<scalar_t>(),
                input.data_ptr<scalar_t>(),
                num_tokens,
                hidden_size
            );
        })
    );
}

void gelu_and_mul(
    torch::Tensor& output,
    const torch::Tensor& input
) {
    int num_tokens = input.size(0);
    int hidden_size = input.size(-1) / 2;
    
    dim3 grid(num_tokens);
    dim3 block(std::min(hidden_size, 1024));
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        input.scalar_type(), "gelu_and_mul_kernel", ([&] {
            gelu_and_mul_kernel<scalar_t><<<grid, block, 0, stream>>>(
                output.data_ptr<scalar_t>(),
                input.data_ptr<scalar_t>(),
                num_tokens,
                hidden_size
            );
        })
    );
}

void fatrelu_and_mul(
    torch::Tensor& output,
    const torch::Tensor& input,
    float threshold
) {
    int num_tokens = input.size(0);
    int hidden_size = input.size(-1) / 2;
    
    dim3 grid(num_tokens);
    dim3 block(std::min(hidden_size, 1024));
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        input.scalar_type(), "fatrelu_and_mul_kernel", ([&] {
            fatrelu_and_mul_kernel<scalar_t><<<grid, block, 0, stream>>>(
                output.data_ptr<scalar_t>(),
                input.data_ptr<scalar_t>(),
                threshold,
                num_tokens,
                hidden_size
            );
        })
    );
}

}  // namespace kernels
}  // namespace vllm
