// SPDX-License-Identifier: Apache-2.0
// vLLM C++ Implementation - Layer Normalization
#pragma once

#include "vllm/common.h"

namespace vllm {

// Forward declarations for CUDA kernels
void rms_norm_cuda(
    torch::Tensor& output,
    const torch::Tensor& input,
    const torch::Tensor& weight,
    float epsilon
);

void fused_add_rms_norm_cuda(
    torch::Tensor& input,
    torch::Tensor& residual,
    const torch::Tensor& weight,
    float epsilon
);

// RMS Normalization Layer
class RMSNorm {
public:
    RMSNorm(int hidden_size, float eps = 1e-6f, DataType dtype = DataType::kFloat16)
        : hidden_size_(hidden_size), eps_(eps), dtype_(dtype) {
        initialize();
    }
    
    void initialize() {
        auto options = torch::TensorOptions()
            .dtype(to_torch_dtype(dtype_))
            .device(torch::kCUDA);
        
        weight_ = torch::ones({hidden_size_}, options);
    }
    
    // Standard forward
    torch::Tensor forward(const torch::Tensor& input) const {
        torch::Tensor output = torch::empty_like(input);
        rms_norm_cuda(output, input, weight_, eps_);
        return output;
    }
    
    // Fused add and normalize (in-place on residual)
    std::tuple<torch::Tensor, torch::Tensor> forward(
        const torch::Tensor& input,
        torch::Tensor& residual
    ) const {
        torch::Tensor hidden = input.clone();
        fused_add_rms_norm_cuda(hidden, residual, weight_, eps_);
        return std::make_tuple(hidden, residual);
    }
    
    void load_weight(const torch::Tensor& weight) {
        weight_.copy_(weight);
    }
    
    torch::Tensor& weight() { return weight_; }
    const torch::Tensor& weight() const { return weight_; }
    
private:
    int hidden_size_;
    float eps_;
    DataType dtype_;
    torch::Tensor weight_;
};

// Fallback CPU implementations (for reference)
inline void rms_norm_cuda(
    torch::Tensor& output,
    const torch::Tensor& input,
    const torch::Tensor& weight,
    float epsilon
) {
    // Compute RMS: sqrt(mean(x^2) + eps)
    auto variance = input.pow(2).mean(-1, true);
    auto rsqrt = torch::rsqrt(variance + epsilon);
    output = input * rsqrt * weight;
}

inline void fused_add_rms_norm_cuda(
    torch::Tensor& input,
    torch::Tensor& residual,
    const torch::Tensor& weight,
    float epsilon
) {
    // Add residual
    residual = residual + input;
    
    // RMS norm
    auto variance = residual.pow(2).mean(-1, true);
    auto rsqrt = torch::rsqrt(variance + epsilon);
    input = residual * rsqrt * weight;
}

}  // namespace vllm
