// SPDX-License-Identifier: Apache-2.0
// vLLM C++ Implementation - Attention Layer
#pragma once

#include "vllm/common.h"
#include "vllm/core/config.h"
#include "vllm/core/kv_cache.h"
#include "vllm/model_executor/layers/linear.h"
#include "vllm/model_executor/layers/rotary_embedding.h"

namespace vllm {

// Forward declarations for CUDA attention kernels
void paged_attention_v1_cuda(
    torch::Tensor& output,
    const torch::Tensor& query,
    const torch::Tensor& key_cache,
    const torch::Tensor& value_cache,
    int num_kv_heads,
    float scale,
    const torch::Tensor& block_tables,
    const torch::Tensor& seq_lens,
    int block_size,
    int max_seq_len
);

void paged_attention_v2_cuda(
    torch::Tensor& output,
    torch::Tensor& exp_sums,
    torch::Tensor& max_logits,
    torch::Tensor& tmp_out,
    const torch::Tensor& query,
    const torch::Tensor& key_cache,
    const torch::Tensor& value_cache,
    int num_kv_heads,
    float scale,
    const torch::Tensor& block_tables,
    const torch::Tensor& seq_lens,
    int block_size,
    int max_seq_len
);

void reshape_and_cache_cuda(
    const torch::Tensor& key,
    const torch::Tensor& value,
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    const torch::Tensor& slot_mapping
);

// Attention types
enum class AttentionType {
    DECODER,         // Causal attention for decoder
    ENCODER_ONLY,    // Bidirectional attention for encoder
    ENCODER_DECODER  // Cross attention
};

// Attention metadata for a batch
struct AttentionMetadata {
    // Sequence lengths for each request
    torch::Tensor seq_lens;
    
    // Block tables for paged attention
    torch::Tensor block_tables;
    
    // Slot mapping for KV cache
    torch::Tensor slot_mapping;
    
    // Maximum sequence length in batch
    int max_seq_len = 0;
    
    // Maximum number of blocks
    int max_num_blocks = 0;
    
    // Is this a prefill or decode step?
    bool is_prefill = true;
    
    // Number of prefill tokens
    int num_prefill_tokens = 0;
    
    // Number of decode tokens
    int num_decode_tokens = 0;
    
    // Query start locations (for variable length sequences)
    torch::Tensor query_start_loc;
    
    // Context lengths
    torch::Tensor context_lens;
};

// Attention layer implementation
class Attention {
public:
    Attention(
        int num_heads,
        int head_dim,
        float scale,
        int num_kv_heads = -1,
        const CacheConfig* cache_config = nullptr,
        AttentionType attn_type = AttentionType::DECODER
    ) : num_heads_(num_heads),
        head_dim_(head_dim),
        scale_(scale),
        num_kv_heads_(num_kv_heads > 0 ? num_kv_heads : num_heads),
        attn_type_(attn_type) {
        
        if (cache_config) {
            block_size_ = cache_config->block_size;
        }
    }
    
    // Forward pass with KV cache
    torch::Tensor forward(
        const torch::Tensor& query,
        const torch::Tensor& key,
        const torch::Tensor& value,
        LayerKVCache* kv_cache,
        const AttentionMetadata& attn_metadata
    ) const {
        int batch_size = query.size(0);
        torch::Tensor output = torch::empty_like(query);
        
        // Cache K and V
        if (kv_cache && attn_metadata.slot_mapping.defined()) {
            reshape_and_cache_cuda(
                key, value,
                kv_cache->key_cache(), kv_cache->value_cache(),
                attn_metadata.slot_mapping
            );
        }
        
        if (attn_metadata.is_prefill) {
            // Prefill: use flash attention or standard attention
            output = prefill_attention(query, key, value, attn_metadata);
        } else {
            // Decode: use paged attention
            output = decode_attention(query, kv_cache, attn_metadata);
        }
        
        return output;
    }
    
    // Simple forward without KV cache (for testing)
    torch::Tensor forward(
        const torch::Tensor& query,
        const torch::Tensor& key,
        const torch::Tensor& value
    ) const {
        // Reshape for attention
        auto batch_size = query.size(0);
        auto q = query.view({batch_size, -1, num_heads_, head_dim_}).transpose(1, 2);
        auto k = key.view({batch_size, -1, num_kv_heads_, head_dim_}).transpose(1, 2);
        auto v = value.view({batch_size, -1, num_kv_heads_, head_dim_}).transpose(1, 2);
        
        // Repeat KV heads if needed (for GQA)
        if (num_kv_heads_ < num_heads_) {
            int repeat = num_heads_ / num_kv_heads_;
            k = k.repeat_interleave(repeat, 1);
            v = v.repeat_interleave(repeat, 1);
        }
        
        // Scaled dot product attention
        auto attn_output = torch::scaled_dot_product_attention(
            q, k, v, 
            /*attn_mask=*/torch::Tensor(),
            /*dropout_p=*/0.0,
            /*is_causal=*/(attn_type_ == AttentionType::DECODER)
        );
        
        // Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
                                 .view({batch_size, -1, num_heads_ * head_dim_});
        
        return attn_output;
    }
    
private:
    torch::Tensor prefill_attention(
        const torch::Tensor& query,
        const torch::Tensor& key,
        const torch::Tensor& value,
        const AttentionMetadata& attn_metadata
    ) const {
        // Use flash attention for prefill
        auto batch_size = query.size(0);
        auto q = query.view({batch_size, -1, num_heads_, head_dim_}).transpose(1, 2);
        auto k = key.view({batch_size, -1, num_kv_heads_, head_dim_}).transpose(1, 2);
        auto v = value.view({batch_size, -1, num_kv_heads_, head_dim_}).transpose(1, 2);
        
        // Repeat KV for GQA
        if (num_kv_heads_ < num_heads_) {
            int repeat = num_heads_ / num_kv_heads_;
            k = k.repeat_interleave(repeat, 1);
            v = v.repeat_interleave(repeat, 1);
        }
        
        auto attn_output = torch::scaled_dot_product_attention(
            q, k, v,
            torch::Tensor(),
            0.0,
            (attn_type_ == AttentionType::DECODER)
        );
        
        return attn_output.transpose(1, 2).contiguous()
                         .view({batch_size, -1, num_heads_ * head_dim_});
    }
    
    torch::Tensor decode_attention(
        const torch::Tensor& query,
        LayerKVCache* kv_cache,
        const AttentionMetadata& attn_metadata
    ) const {
        torch::Tensor output = torch::empty_like(query);
        
        // Use paged attention for decode
        paged_attention_v1_cuda(
            output,
            query,
            kv_cache->key_cache(),
            kv_cache->value_cache(),
            num_kv_heads_,
            scale_,
            attn_metadata.block_tables,
            attn_metadata.seq_lens,
            block_size_,
            attn_metadata.max_seq_len
        );
        
        return output;
    }
    
    int num_heads_;
    int head_dim_;
    float scale_;
    int num_kv_heads_;
    int block_size_ = 16;
    AttentionType attn_type_;
};

// Fallback implementations
inline void reshape_and_cache_cuda(
    const torch::Tensor& key,
    const torch::Tensor& value,
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    const torch::Tensor& slot_mapping
) {
    // Scatter key and value to cache based on slot mapping
    auto num_tokens = key.size(0);
    for (int64_t i = 0; i < num_tokens; ++i) {
        int slot = slot_mapping[i].item<int>();
        if (slot >= 0) {
            int block_idx = slot / key_cache.size(1);
            int block_offset = slot % key_cache.size(1);
            key_cache[block_idx][block_offset].copy_(key[i]);
            value_cache[block_idx][block_offset].copy_(value[i]);
        }
    }
}

inline void paged_attention_v1_cuda(
    torch::Tensor& output,
    const torch::Tensor& query,
    const torch::Tensor& key_cache,
    const torch::Tensor& value_cache,
    int num_kv_heads,
    float scale,
    const torch::Tensor& block_tables,
    const torch::Tensor& seq_lens,
    int block_size,
    int max_seq_len
) {
    // This is a simplified fallback - real implementation uses CUDA kernels
    // For actual usage, this would call the paged_attention kernel from csrc/
    output.zero_();
}

inline void paged_attention_v2_cuda(
    torch::Tensor& output,
    torch::Tensor& exp_sums,
    torch::Tensor& max_logits,
    torch::Tensor& tmp_out,
    const torch::Tensor& query,
    const torch::Tensor& key_cache,
    const torch::Tensor& value_cache,
    int num_kv_heads,
    float scale,
    const torch::Tensor& block_tables,
    const torch::Tensor& seq_lens,
    int block_size,
    int max_seq_len
) {
    // Fallback implementation
    output.zero_();
}

}  // namespace vllm
