// SPDX-License-Identifier: Apache-2.0
// vLLM C++ Implementation - KV Cache CUDA Kernels
// Adapted from vLLM csrc/cache_kernels.cu

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <torch/torch.h>
#include <ATen/cuda/CUDAContext.h>

namespace vllm {
namespace kernels {

// Reshape and cache kernel
template<typename scalar_t>
__global__ void reshape_and_cache_kernel(
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    scalar_t* __restrict__ key_cache,
    scalar_t* __restrict__ value_cache,
    const int32_t* __restrict__ slot_mapping,
    const int num_tokens,
    const int num_heads,
    const int head_size,
    const int block_size,
    const int key_stride,
    const int value_stride
) {
    const int token_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    
    if (token_idx >= num_tokens) return;
    
    const int slot = slot_mapping[token_idx];
    if (slot < 0) return;  // Padding token
    
    const int block_idx = slot / block_size;
    const int block_offset = slot % block_size;
    
    // Key layout: [num_blocks, block_size, num_heads, head_size]
    const int cache_offset = block_idx * block_size * num_heads * head_size +
                             block_offset * num_heads * head_size +
                             head_idx * head_size;
    
    // Input layout: [num_tokens, num_heads * head_size] or [num_tokens, num_heads, head_size]
    const int input_key_offset = token_idx * key_stride + head_idx * head_size;
    const int input_value_offset = token_idx * value_stride + head_idx * head_size;
    
    for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
        key_cache[cache_offset + i] = key[input_key_offset + i];
        value_cache[cache_offset + i] = value[input_value_offset + i];
    }
}

// Flash-style reshape and cache kernel
// Cache layout: [num_blocks, 2, block_size, num_heads, head_size]
template<typename scalar_t>
__global__ void reshape_and_cache_flash_kernel(
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    scalar_t* __restrict__ kv_cache,
    const int32_t* __restrict__ slot_mapping,
    const int num_tokens,
    const int num_heads,
    const int head_size,
    const int block_size
) {
    const int token_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    
    if (token_idx >= num_tokens) return;
    
    const int slot = slot_mapping[token_idx];
    if (slot < 0) return;
    
    const int block_idx = slot / block_size;
    const int block_offset = slot % block_size;
    
    // Cache layout: [num_blocks, 2, block_size, num_heads, head_size]
    const int block_stride = 2 * block_size * num_heads * head_size;
    const int key_cache_offset = block_idx * block_stride +
                                  0 * block_size * num_heads * head_size +  // Key is at index 0
                                  block_offset * num_heads * head_size +
                                  head_idx * head_size;
    const int value_cache_offset = block_idx * block_stride +
                                    1 * block_size * num_heads * head_size +  // Value is at index 1
                                    block_offset * num_heads * head_size +
                                    head_idx * head_size;
    
    const int input_offset = token_idx * num_heads * head_size + head_idx * head_size;
    
    for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
        kv_cache[key_cache_offset + i] = key[input_offset + i];
        kv_cache[value_cache_offset + i] = value[input_offset + i];
    }
}

// Swap blocks kernel
template<typename scalar_t>
__global__ void swap_blocks_kernel(
    scalar_t* __restrict__ src,
    scalar_t* __restrict__ dst,
    const int64_t* __restrict__ block_mapping,
    const int num_pairs,
    const int block_size_bytes
) {
    const int pair_idx = blockIdx.x;
    if (pair_idx >= num_pairs) return;
    
    const int src_block = block_mapping[pair_idx * 2];
    const int dst_block = block_mapping[pair_idx * 2 + 1];
    
    const int block_offset_src = src_block * block_size_bytes;
    const int block_offset_dst = dst_block * block_size_bytes;
    
    // Copy in chunks of sizeof(scalar_t)
    const int num_elements = block_size_bytes / sizeof(scalar_t);
    
    for (int i = threadIdx.x; i < num_elements; i += blockDim.x) {
        dst[block_offset_dst / sizeof(scalar_t) + i] = 
            src[block_offset_src / sizeof(scalar_t) + i];
    }
}

// Dispatch functions
void reshape_and_cache(
    const torch::Tensor& key,
    const torch::Tensor& value,
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    const torch::Tensor& slot_mapping
) {
    int num_tokens = key.size(0);
    int num_heads = key_cache.size(2);
    int head_size = key_cache.size(3);
    int block_size = key_cache.size(1);
    
    int key_stride = key.size(-1);
    int value_stride = value.size(-1);
    
    dim3 grid(num_tokens, num_heads);
    dim3 block(std::min(head_size, 256));
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        key.scalar_type(), "reshape_and_cache_kernel", ([&] {
            reshape_and_cache_kernel<scalar_t><<<grid, block, 0, stream>>>(
                key.data_ptr<scalar_t>(),
                value.data_ptr<scalar_t>(),
                key_cache.data_ptr<scalar_t>(),
                value_cache.data_ptr<scalar_t>(),
                slot_mapping.data_ptr<int32_t>(),
                num_tokens,
                num_heads,
                head_size,
                block_size,
                key_stride,
                value_stride
            );
        })
    );
}

void reshape_and_cache_flash(
    const torch::Tensor& key,
    const torch::Tensor& value,
    torch::Tensor& kv_cache,
    const torch::Tensor& slot_mapping
) {
    int num_tokens = key.size(0);
    int num_heads = kv_cache.size(3);
    int head_size = kv_cache.size(4);
    int block_size = kv_cache.size(2);
    
    dim3 grid(num_tokens, num_heads);
    dim3 block(std::min(head_size, 256));
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        key.scalar_type(), "reshape_and_cache_flash_kernel", ([&] {
            reshape_and_cache_flash_kernel<scalar_t><<<grid, block, 0, stream>>>(
                key.data_ptr<scalar_t>(),
                value.data_ptr<scalar_t>(),
                kv_cache.data_ptr<scalar_t>(),
                slot_mapping.data_ptr<int32_t>(),
                num_tokens,
                num_heads,
                head_size,
                block_size
            );
        })
    );
}

void swap_blocks(
    torch::Tensor& src,
    torch::Tensor& dst,
    int block_size_in_bytes,
    const torch::Tensor& block_mapping
) {
    int num_pairs = block_mapping.size(0) / 2;
    
    dim3 grid(num_pairs);
    dim3 block(256);
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        src.scalar_type(), "swap_blocks_kernel", ([&] {
            swap_blocks_kernel<scalar_t><<<grid, block, 0, stream>>>(
                src.data_ptr<scalar_t>(),
                dst.data_ptr<scalar_t>(),
                block_mapping.data_ptr<int64_t>(),
                num_pairs,
                block_size_in_bytes
            );
        })
    );
}

}  // namespace kernels
}  // namespace vllm
