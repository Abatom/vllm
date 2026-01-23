// SPDX-License-Identifier: Apache-2.0
// vLLM C++ - Layer Tests (Simple, without GTest)

#include <iostream>
#include <cassert>
#include <cmath>

#include "vllm/common.h"
#include "vllm/model_executor/layers/linear.h"
#include "vllm/model_executor/layers/layernorm.h"
#include "vllm/model_executor/layers/activation.h"
#include "vllm/model_executor/layers/rotary_embedding.h"

using namespace vllm;

#define TEST_ASSERT(cond, msg) \
    if (!(cond)) { \
        std::cerr << "FAILED: " << msg << std::endl; \
        return false; \
    }

bool test_linear_layer() {
    std::cout << "Testing Linear layer..." << std::endl;
    
    int in_features = 256;
    int out_features = 512;
    int batch_size = 4;
    
    Linear linear(in_features, out_features, true, DataType::kFloat16);
    
    auto input = torch::randn({batch_size, in_features}, 
        torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA));
    
    auto output = linear.forward(input);
    
    TEST_ASSERT(output.size(0) == batch_size, "Batch size mismatch");
    TEST_ASSERT(output.size(1) == out_features, "Output features mismatch");
    
    std::cout << "  Linear layer: PASSED" << std::endl;
    return true;
}

bool test_rms_norm() {
    std::cout << "Testing RMSNorm layer..." << std::endl;
    
    int hidden_size = 256;
    int batch_size = 4;
    float eps = 1e-6f;
    
    RMSNorm norm(hidden_size, eps, DataType::kFloat16);
    
    auto input = torch::randn({batch_size, hidden_size},
        torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA));
    
    auto output = norm.forward(input);
    
    TEST_ASSERT(output.sizes() == input.sizes(), "Shape mismatch");
    
    // Check that output is normalized (variance should be close to 1)
    auto output_f32 = output.to(torch::kFloat32);
    auto variance = (output_f32 * output_f32).mean(-1);
    float mean_var = variance.mean().item<float>();
    
    TEST_ASSERT(std::abs(mean_var - 1.0f) < 0.5f, "Variance not normalized");
    
    std::cout << "  RMSNorm layer: PASSED" << std::endl;
    return true;
}

bool test_silu_activation() {
    std::cout << "Testing SiLU activation..." << std::endl;
    
    int hidden_size = 256;
    int batch_size = 4;
    
    SiluAndMul activation;
    
    // Input should be 2x hidden_size (gate and up combined)
    auto input = torch::randn({batch_size, hidden_size * 2},
        torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA));
    
    auto output = activation.forward(input);
    
    TEST_ASSERT(output.size(0) == batch_size, "Batch size mismatch");
    TEST_ASSERT(output.size(1) == hidden_size, "Hidden size mismatch");
    
    std::cout << "  SiLU activation: PASSED" << std::endl;
    return true;
}

bool test_rotary_embedding() {
    std::cout << "Testing Rotary Embedding..." << std::endl;
    
    int head_dim = 64;
    int max_position = 4096;
    float rope_theta = 10000.0f;
    
    RotaryEmbedding rope(head_dim, max_position, rope_theta, true, 1.0f, DataType::kFloat16);
    
    int batch_size = 4;
    int num_heads = 8;
    
    auto positions = torch::arange(0, batch_size, torch::kInt32).to(torch::kCUDA);
    auto query = torch::randn({batch_size, num_heads * head_dim},
        torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA));
    auto key = torch::randn({batch_size, num_heads * head_dim},
        torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA));
    
    auto [q_out, k_out] = rope.forward(positions, query, key);
    
    TEST_ASSERT(q_out.sizes() == query.sizes(), "Query shape mismatch");
    TEST_ASSERT(k_out.sizes() == key.sizes(), "Key shape mismatch");
    
    std::cout << "  Rotary Embedding: PASSED" << std::endl;
    return true;
}

bool test_mlp() {
    std::cout << "Testing MLP layer..." << std::endl;
    
    int hidden_size = 256;
    int intermediate_size = 512;
    int batch_size = 4;
    
    MLP mlp(hidden_size, intermediate_size, "silu", 0.0f, 1, DataType::kFloat16);
    
    auto input = torch::randn({batch_size, hidden_size},
        torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA));
    
    auto output = mlp.forward(input);
    
    TEST_ASSERT(output.sizes() == input.sizes(), "Shape mismatch");
    
    std::cout << "  MLP layer: PASSED" << std::endl;
    return true;
}

int main() {
    std::cout << "vLLM C++ Layer Tests" << std::endl;
    std::cout << "====================" << std::endl;
    std::cout << std::endl;
    
    // Initialize CUDA
    torch::cuda::set_device(0);
    
    int passed = 0;
    int failed = 0;
    
    if (test_linear_layer()) passed++; else failed++;
    if (test_rms_norm()) passed++; else failed++;
    if (test_silu_activation()) passed++; else failed++;
    if (test_rotary_embedding()) passed++; else failed++;
    if (test_mlp()) passed++; else failed++;
    
    std::cout << std::endl;
    std::cout << "Results: " << passed << " passed, " << failed << " failed" << std::endl;
    
    return (failed > 0) ? 1 : 0;
}
