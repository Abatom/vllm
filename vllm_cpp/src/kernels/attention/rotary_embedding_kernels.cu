// SPDX-License-Identifier: Apache-2.0
// vLLM C++ Implementation - Rotary Embedding CUDA Kernels
// Adapted from vLLM csrc/pos_encoding_kernels.cu

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <torch/torch.h>
#include <ATen/cuda/CUDAContext.h>

#include <cmath>

namespace vllm {
namespace kernels {

// Rotary embedding kernel (NeoX style)
template<typename scalar_t>
__global__ void rotary_embedding_neox_kernel(
    const int32_t* __restrict__ positions,
    scalar_t* __restrict__ query,
    scalar_t* __restrict__ key,
    const scalar_t* __restrict__ cos_sin_cache,
    const int head_size,
    const int rotary_dim,
    const int num_tokens,
    const int num_heads,
    const int num_kv_heads,
    const int q_stride,
    const int k_stride
) {
    const int token_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    
    if (token_idx >= num_tokens) return;
    
    const int pos = positions[token_idx];
    const int half_rotary_dim = rotary_dim / 2;
    
    // Get cos and sin from cache
    const scalar_t* cache = cos_sin_cache + pos * rotary_dim;
    const scalar_t* cos_ptr = cache;
    const scalar_t* sin_ptr = cache + half_rotary_dim;
    
    // Apply to query
    if (head_idx < num_heads) {
        scalar_t* q = query + token_idx * q_stride + head_idx * head_size;
        
        for (int i = threadIdx.x; i < half_rotary_dim; i += blockDim.x) {
            float cos_val = static_cast<float>(cos_ptr[i]);
            float sin_val = static_cast<float>(sin_ptr[i]);
            
            float q0 = static_cast<float>(q[i]);
            float q1 = static_cast<float>(q[i + half_rotary_dim]);
            
            q[i] = static_cast<scalar_t>(q0 * cos_val - q1 * sin_val);
            q[i + half_rotary_dim] = static_cast<scalar_t>(q1 * cos_val + q0 * sin_val);
        }
    }
    
    // Apply to key
    if (head_idx < num_kv_heads) {
        scalar_t* k = key + token_idx * k_stride + head_idx * head_size;
        
        for (int i = threadIdx.x; i < half_rotary_dim; i += blockDim.x) {
            float cos_val = static_cast<float>(cos_ptr[i]);
            float sin_val = static_cast<float>(sin_ptr[i]);
            
            float k0 = static_cast<float>(k[i]);
            float k1 = static_cast<float>(k[i + half_rotary_dim]);
            
            k[i] = static_cast<scalar_t>(k0 * cos_val - k1 * sin_val);
            k[i + half_rotary_dim] = static_cast<scalar_t>(k1 * cos_val + k0 * sin_val);
        }
    }
}

// Rotary embedding kernel (GPT-J style)
template<typename scalar_t>
__global__ void rotary_embedding_gptj_kernel(
    const int32_t* __restrict__ positions,
    scalar_t* __restrict__ query,
    scalar_t* __restrict__ key,
    const scalar_t* __restrict__ cos_sin_cache,
    const int head_size,
    const int rotary_dim,
    const int num_tokens,
    const int num_heads,
    const int num_kv_heads,
    const int q_stride,
    const int k_stride
) {
    const int token_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    
    if (token_idx >= num_tokens) return;
    
    const int pos = positions[token_idx];
    const int half_rotary_dim = rotary_dim / 2;
    
    // Get cos and sin from cache
    const scalar_t* cache = cos_sin_cache + pos * rotary_dim;
    const scalar_t* cos_ptr = cache;
    const scalar_t* sin_ptr = cache + half_rotary_dim;
    
    // Apply to query
    if (head_idx < num_heads) {
        scalar_t* q = query + token_idx * q_stride + head_idx * head_size;
        
        for (int i = threadIdx.x; i < half_rotary_dim; i += blockDim.x) {
            float cos_val = static_cast<float>(cos_ptr[i]);
            float sin_val = static_cast<float>(sin_ptr[i]);
            
            int src_idx = 2 * i;
            float q0 = static_cast<float>(q[src_idx]);
            float q1 = static_cast<float>(q[src_idx + 1]);
            
            q[src_idx] = static_cast<scalar_t>(q0 * cos_val - q1 * sin_val);
            q[src_idx + 1] = static_cast<scalar_t>(q0 * sin_val + q1 * cos_val);
        }
    }
    
    // Apply to key
    if (head_idx < num_kv_heads) {
        scalar_t* k = key + token_idx * k_stride + head_idx * head_size;
        
        for (int i = threadIdx.x; i < half_rotary_dim; i += blockDim.x) {
            float cos_val = static_cast<float>(cos_ptr[i]);
            float sin_val = static_cast<float>(sin_ptr[i]);
            
            int src_idx = 2 * i;
            float k0 = static_cast<float>(k[src_idx]);
            float k1 = static_cast<float>(k[src_idx + 1]);
            
            k[src_idx] = static_cast<scalar_t>(k0 * cos_val - k1 * sin_val);
            k[src_idx + 1] = static_cast<scalar_t>(k0 * sin_val + k1 * cos_val);
        }
    }
}

// Dispatch function
void rotary_embedding(
    const torch::Tensor& positions,
    torch::Tensor& query,
    torch::Tensor& key,
    int head_size,
    const torch::Tensor& cos_sin_cache,
    bool is_neox
) {
    int num_tokens = positions.size(0);
    int rotary_dim = cos_sin_cache.size(-1);
    
    // Infer head counts from tensor shapes
    int q_stride = query.size(-1);
    int k_stride = key.size(-1);
    int num_heads = q_stride / head_size;
    int num_kv_heads = k_stride / head_size;
    
    dim3 grid(num_tokens, std::max(num_heads, num_kv_heads));
    dim3 block(std::min(rotary_dim / 2, 256));
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    if (is_neox) {
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16,
            query.scalar_type(), "rotary_embedding_neox_kernel", ([&] {
                rotary_embedding_neox_kernel<scalar_t><<<grid, block, 0, stream>>>(
                    positions.data_ptr<int32_t>(),
                    query.data_ptr<scalar_t>(),
                    key.data_ptr<scalar_t>(),
                    cos_sin_cache.data_ptr<scalar_t>(),
                    head_size,
                    rotary_dim,
                    num_tokens,
                    num_heads,
                    num_kv_heads,
                    q_stride,
                    k_stride
                );
            })
        );
    } else {
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16,
            query.scalar_type(), "rotary_embedding_gptj_kernel", ([&] {
                rotary_embedding_gptj_kernel<scalar_t><<<grid, block, 0, stream>>>(
                    positions.data_ptr<int32_t>(),
                    query.data_ptr<scalar_t>(),
                    key.data_ptr<scalar_t>(),
                    cos_sin_cache.data_ptr<scalar_t>(),
                    head_size,
                    rotary_dim,
                    num_tokens,
                    num_heads,
                    num_kv_heads,
                    q_stride,
                    k_stride
                );
            })
        );
    }
}

}  // namespace kernels
}  // namespace vllm
