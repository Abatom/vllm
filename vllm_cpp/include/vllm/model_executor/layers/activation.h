// SPDX-License-Identifier: Apache-2.0
// vLLM C++ Implementation - Activation Functions
#pragma once

#include "vllm/common.h"

namespace vllm {

// Forward declarations for CUDA kernels
void silu_and_mul_cuda(torch::Tensor& output, const torch::Tensor& input);
void gelu_and_mul_cuda(torch::Tensor& output, const torch::Tensor& input);
void gelu_tanh_and_mul_cuda(torch::Tensor& output, const torch::Tensor& input);
void fatrelu_and_mul_cuda(torch::Tensor& output, const torch::Tensor& input, float threshold);

// Base activation interface
class ActivationBase {
public:
    virtual ~ActivationBase() = default;
    virtual torch::Tensor forward(const torch::Tensor& input) const = 0;
};

// SiLU (Swish) and Mul - Used in Qwen2 and many LLMs
// Takes input of shape [..., 2 * hidden_size] and outputs [..., hidden_size]
// gate = input[..., :hidden_size], up = input[..., hidden_size:]
// output = silu(gate) * up
class SiluAndMul : public ActivationBase {
public:
    torch::Tensor forward(const torch::Tensor& input) const override {
        int64_t hidden_size = input.size(-1) / 2;
        torch::Tensor output = torch::empty({input.sizes().vec()}, input.options());
        output = output.narrow(-1, 0, hidden_size);
        
        // Use CUDA kernel if available
        silu_and_mul_cuda(output, input);
        return output;
    }
};

// GELU and Mul - Used in some models
class GeluAndMul : public ActivationBase {
public:
    torch::Tensor forward(const torch::Tensor& input) const override {
        int64_t hidden_size = input.size(-1) / 2;
        torch::Tensor output = torch::empty_like(input.narrow(-1, 0, hidden_size));
        gelu_and_mul_cuda(output, input);
        return output;
    }
};

// GELU Tanh approximation and Mul
class GeluTanhAndMul : public ActivationBase {
public:
    torch::Tensor forward(const torch::Tensor& input) const override {
        int64_t hidden_size = input.size(-1) / 2;
        torch::Tensor output = torch::empty_like(input.narrow(-1, 0, hidden_size));
        gelu_tanh_and_mul_cuda(output, input);
        return output;
    }
};

// FATReLU and Mul - Used in MiniCPM
class FatreluAndMul : public ActivationBase {
public:
    explicit FatreluAndMul(float threshold = 0.0f) : threshold_(threshold) {}
    
    torch::Tensor forward(const torch::Tensor& input) const override {
        int64_t hidden_size = input.size(-1) / 2;
        torch::Tensor output = torch::empty_like(input.narrow(-1, 0, hidden_size));
        fatrelu_and_mul_cuda(output, input, threshold_);
        return output;
    }
    
private:
    float threshold_;
};

// Fallback implementations
inline void silu_and_mul_cuda(torch::Tensor& output, const torch::Tensor& input) {
    int64_t hidden_size = input.size(-1) / 2;
    auto gate = input.narrow(-1, 0, hidden_size);
    auto up = input.narrow(-1, hidden_size, hidden_size);
    output = torch::silu(gate) * up;
}

inline void gelu_and_mul_cuda(torch::Tensor& output, const torch::Tensor& input) {
    int64_t hidden_size = input.size(-1) / 2;
    auto gate = input.narrow(-1, 0, hidden_size);
    auto up = input.narrow(-1, hidden_size, hidden_size);
    output = torch::gelu(gate) * up;
}

inline void gelu_tanh_and_mul_cuda(torch::Tensor& output, const torch::Tensor& input) {
    int64_t hidden_size = input.size(-1) / 2;
    auto gate = input.narrow(-1, 0, hidden_size);
    auto up = input.narrow(-1, hidden_size, hidden_size);
    output = torch::gelu(gate, "tanh") * up;
}

inline void fatrelu_and_mul_cuda(torch::Tensor& output, const torch::Tensor& input, float threshold) {
    int64_t hidden_size = input.size(-1) / 2;
    auto gate = input.narrow(-1, 0, hidden_size);
    auto up = input.narrow(-1, hidden_size, hidden_size);
    auto relu_out = torch::relu(gate);
    // Apply threshold
    auto mask = relu_out > threshold;
    output = torch::where(mask, relu_out, torch::zeros_like(relu_out)) * up;
}

// Factory function to create activation by name
inline std::unique_ptr<ActivationBase> create_activation(const std::string& name, float param = 0.0f) {
    if (name == "silu" || name == "swish") {
        return std::make_unique<SiluAndMul>();
    } else if (name == "gelu") {
        return std::make_unique<GeluAndMul>();
    } else if (name == "gelu_tanh") {
        return std::make_unique<GeluTanhAndMul>();
    } else if (name == "fatrelu") {
        return std::make_unique<FatreluAndMul>(param);
    }
    throw std::runtime_error("Unknown activation: " + name);
}

}  // namespace vllm
