// SPDX-License-Identifier: Apache-2.0
// vLLM C++ Implementation - Rotary Position Embedding
#pragma once

#include "vllm/common.h"

namespace vllm {

// Forward declaration for CUDA kernel
void rotary_embedding_cuda(
    const torch::Tensor& positions,
    torch::Tensor& query,
    torch::Tensor& key,
    int head_size,
    const torch::Tensor& cos_sin_cache,
    bool is_neox
);

// Rotary Position Embedding
class RotaryEmbedding {
public:
    RotaryEmbedding(
        int head_dim,
        int max_position = 131072,
        float rope_theta = 10000.0f,
        bool is_neox = true,
        float partial_rotary_factor = 1.0f,
        DataType dtype = DataType::kFloat16
    ) : head_dim_(head_dim),
        max_position_(max_position),
        rope_theta_(rope_theta),
        is_neox_(is_neox),
        partial_rotary_factor_(partial_rotary_factor),
        dtype_(dtype) {
        
        rotary_dim_ = static_cast<int>(head_dim * partial_rotary_factor);
        initialize();
    }
    
    void initialize() {
        // Compute inverse frequencies
        int half_dim = rotary_dim_ / 2;
        std::vector<float> inv_freq(half_dim);
        for (int i = 0; i < half_dim; ++i) {
            inv_freq[i] = 1.0f / std::pow(rope_theta_, static_cast<float>(2 * i) / rotary_dim_);
        }
        
        // Build cos/sin cache
        auto options = torch::TensorOptions()
            .dtype(to_torch_dtype(dtype_))
            .device(torch::kCUDA);
        
        // Create position indices
        auto pos_tensor = torch::arange(0, max_position_, torch::kFloat32);
        auto inv_freq_tensor = torch::from_blob(
            inv_freq.data(), {half_dim}, torch::kFloat32
        ).clone();
        
        // Compute angles: pos * inv_freq
        auto freqs = torch::outer(pos_tensor, inv_freq_tensor);
        
        // Compute cos and sin
        auto cos = torch::cos(freqs);
        auto sin = torch::sin(freqs);
        
        // Stack cos and sin: [max_position, rotary_dim]
        cos_sin_cache_ = torch::cat({cos, sin}, -1).to(options);
    }
    
    // Apply rotary embedding to query and key
    std::tuple<torch::Tensor, torch::Tensor> forward(
        const torch::Tensor& positions,
        torch::Tensor query,
        torch::Tensor key
    ) const {
        rotary_embedding_cuda(positions, query, key, head_dim_, cos_sin_cache_, is_neox_);
        return std::make_tuple(query, key);
    }
    
    int head_dim() const { return head_dim_; }
    int rotary_dim() const { return rotary_dim_; }
    
    const torch::Tensor& cos_sin_cache() const { return cos_sin_cache_; }
    
private:
    int head_dim_;
    int rotary_dim_;
    int max_position_;
    float rope_theta_;
    bool is_neox_;
    float partial_rotary_factor_;
    DataType dtype_;
    torch::Tensor cos_sin_cache_;
};

// Fallback implementation
inline void rotary_embedding_cuda(
    const torch::Tensor& positions,
    torch::Tensor& query,
    torch::Tensor& key,
    int head_size,
    const torch::Tensor& cos_sin_cache,
    bool is_neox
) {
    // Get cos and sin from cache based on positions
    int rotary_dim = cos_sin_cache.size(-1);
    int half_dim = rotary_dim / 2;
    
    // Index into cache
    auto cos = cos_sin_cache.index_select(0, positions.flatten()).narrow(-1, 0, half_dim);
    auto sin = cos_sin_cache.index_select(0, positions.flatten()).narrow(-1, half_dim, half_dim);
    
    // Reshape for broadcasting
    auto q_shape = query.sizes().vec();
    auto batch_size = q_shape[0];
    
    // Apply rotary embedding
    auto apply_rotary = [&](torch::Tensor& x) {
        auto x_shape = x.sizes().vec();
        auto num_heads = x_shape[x_shape.size() - 1] / head_size;
        x = x.view({batch_size, -1, head_size});
        
        auto x_rot = x.narrow(-1, 0, half_dim);
        auto x_pass = x.narrow(-1, half_dim, head_size - half_dim);
        
        // Rotate
        if (is_neox) {
            auto x1 = x_rot.narrow(-1, 0, half_dim / 2);
            auto x2 = x_rot.narrow(-1, half_dim / 2, half_dim / 2);
            
            cos = cos.view({batch_size, 1, half_dim});
            sin = sin.view({batch_size, 1, half_dim});
            
            auto rot_half = torch::cat({-x2, x1}, -1);
            x_rot = x_rot * cos + rot_half * sin;
        }
        
        x = torch::cat({x_rot, x_pass}, -1);
        x = x.view(x_shape);
    };
    
    apply_rotary(query);
    apply_rotary(key);
}

// Factory function
inline std::unique_ptr<RotaryEmbedding> get_rope(
    int head_dim,
    int max_position = 131072,
    float rope_theta = 10000.0f,
    bool is_neox = true,
    float partial_rotary_factor = 1.0f,
    DataType dtype = DataType::kFloat16
) {
    return std::make_unique<RotaryEmbedding>(
        head_dim, max_position, rope_theta, is_neox, partial_rotary_factor, dtype
    );
}

}  // namespace vllm
