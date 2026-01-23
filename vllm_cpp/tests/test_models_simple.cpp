// SPDX-License-Identifier: Apache-2.0
// vLLM C++ - Model Tests (Simple, without GTest)

#include <iostream>
#include <cassert>

#include "vllm/common.h"
#include "vllm/core/config.h"
#include "vllm/core/kv_cache.h"
#include "vllm/model_executor/models/qwen2.h"
#include "vllm/model_executor/models/mimo_v2_flash.h"

using namespace vllm;

#define TEST_ASSERT(cond, msg) \
    if (!(cond)) { \
        std::cerr << "FAILED: " << msg << std::endl; \
        return false; \
    }

bool test_qwen2_decoder_layer() {
    std::cout << "Testing Qwen2 Decoder Layer..." << std::endl;
    
    Qwen2Config config;
    config.hidden_size = 256;
    config.intermediate_size = 512;
    config.num_attention_heads = 8;
    config.num_key_value_heads = 4;
    config.head_dim = 32;
    config.rms_norm_eps = 1e-6f;
    config.rope_theta = 10000.0f;
    config.hidden_act = "silu";
    config.dtype = DataType::kFloat16;
    
    int batch_size = 2;
    int seq_len = 4;
    
    Qwen2DecoderLayer layer(config, 0, nullptr, 1, "model");
    
    auto positions = torch::arange(0, seq_len, torch::kInt32)
        .unsqueeze(0).expand({batch_size, seq_len}).contiguous()
        .view({-1}).to(torch::kCUDA);
    
    auto hidden = torch::randn({batch_size * seq_len, config.hidden_size},
        torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA));
    
    torch::Tensor residual;
    auto [output, new_residual] = layer.forward(positions, hidden, residual);
    
    TEST_ASSERT(output.size(0) == batch_size * seq_len, "Batch size mismatch");
    TEST_ASSERT(output.size(1) == config.hidden_size, "Hidden size mismatch");
    
    std::cout << "  Qwen2 Decoder Layer: PASSED" << std::endl;
    return true;
}

bool test_qwen2_model() {
    std::cout << "Testing Qwen2 Model..." << std::endl;
    
    // Small config for testing
    Qwen2Config config;
    config.hidden_size = 256;
    config.intermediate_size = 512;
    config.num_hidden_layers = 4;
    config.num_attention_heads = 8;
    config.num_key_value_heads = 4;
    config.vocab_size = 1000;
    config.max_position_embeddings = 512;
    config.head_dim = 32;
    config.rms_norm_eps = 1e-6f;
    config.rope_theta = 10000.0f;
    config.hidden_act = "silu";
    config.dtype = DataType::kFloat16;
    
    CacheConfig cache_config;
    cache_config.block_size = 16;
    
    Qwen2ForCausalLM model(config, &cache_config, 1);
    
    int batch_size = 2;
    int seq_len = 4;
    
    auto input_ids = torch::randint(0, config.vocab_size, {batch_size * seq_len},
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    
    auto positions = torch::arange(0, seq_len, torch::kInt32)
        .unsqueeze(0).expand({batch_size, seq_len}).contiguous()
        .view({-1}).to(torch::kCUDA);
    
    auto hidden = model.forward(input_ids, positions);
    
    TEST_ASSERT(hidden.size(0) == batch_size * seq_len, "Batch size mismatch");
    TEST_ASSERT(hidden.size(1) == config.hidden_size, "Hidden size mismatch");
    
    auto logits = model.compute_logits(hidden);
    
    TEST_ASSERT(logits.size(0) == batch_size * seq_len, "Logits batch size mismatch");
    TEST_ASSERT(logits.size(1) == config.vocab_size, "Vocab size mismatch");
    
    std::cout << "  Qwen2 Model: PASSED" << std::endl;
    return true;
}

bool test_mimo_v2_flash_decoder_layer() {
    std::cout << "Testing MiMoV2Flash Decoder Layer..." << std::endl;
    
    MiMoV2FlashConfig config;
    config.hidden_size = 256;
    config.intermediate_size = 512;
    config.moe_intermediate_size = 256;
    config.num_attention_heads = 8;
    config.num_key_value_heads = 4;
    config.head_dim = 32;
    config.v_head_dim = 32;
    config.layernorm_epsilon = 1e-6f;
    config.rope_theta = 10000.0f;
    config.hidden_act = "silu";
    config.n_routed_experts = 8;
    config.num_experts_per_tok = 2;
    config.dtype = DataType::kFloat16;
    
    config.hybrid_layer_pattern = {0};  // Normal attention
    config.moe_layer_freq = {true};     // MoE layer
    
    int batch_size = 2;
    int seq_len = 4;
    
    MiMoV2FlashDecoderLayer layer(config, 0, nullptr, 1, "model");
    
    auto positions = torch::arange(0, seq_len, torch::kInt32)
        .unsqueeze(0).expand({batch_size, seq_len}).contiguous()
        .view({-1}).to(torch::kCUDA);
    
    auto hidden = torch::randn({batch_size * seq_len, config.hidden_size},
        torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA));
    
    torch::Tensor residual;
    auto [output, new_residual] = layer.forward(positions, hidden, residual);
    
    TEST_ASSERT(output.size(0) == batch_size * seq_len, "Batch size mismatch");
    TEST_ASSERT(output.size(1) == config.hidden_size, "Hidden size mismatch");
    TEST_ASSERT(layer.is_moe_layer(), "Should be MoE layer");
    
    std::cout << "  MiMoV2Flash Decoder Layer: PASSED" << std::endl;
    return true;
}

bool test_mimo_v2_flash_model() {
    std::cout << "Testing MiMoV2Flash Model..." << std::endl;
    
    // Small config for testing
    MiMoV2FlashConfig config;
    config.hidden_size = 256;
    config.intermediate_size = 512;
    config.moe_intermediate_size = 256;
    config.num_hidden_layers = 4;
    config.num_attention_heads = 8;
    config.num_key_value_heads = 4;
    config.vocab_size = 1000;
    config.max_position_embeddings = 512;
    config.head_dim = 32;
    config.v_head_dim = 32;
    config.layernorm_epsilon = 1e-6f;
    config.rope_theta = 10000.0f;
    config.hidden_act = "silu";
    config.n_routed_experts = 8;
    config.num_experts_per_tok = 2;
    config.dtype = DataType::kFloat16;
    
    // Set up patterns
    config.hybrid_layer_pattern = {0, 0, 1, 0};  // Layer 2 uses SWA
    config.moe_layer_freq = {false, true, true, true};  // First layer is dense
    
    config.swa_num_attention_heads = 8;
    config.swa_num_key_value_heads = 4;
    config.swa_head_dim = 32;
    config.swa_v_head_dim = 16;
    
    CacheConfig cache_config;
    cache_config.block_size = 16;
    
    MiMoV2FlashForCausalLM model(config, &cache_config, 1);
    
    int batch_size = 2;
    int seq_len = 4;
    
    auto input_ids = torch::randint(0, config.vocab_size, {batch_size * seq_len},
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    
    auto positions = torch::arange(0, seq_len, torch::kInt32)
        .unsqueeze(0).expand({batch_size, seq_len}).contiguous()
        .view({-1}).to(torch::kCUDA);
    
    auto hidden = model.forward(input_ids, positions);
    
    TEST_ASSERT(hidden.size(0) == batch_size * seq_len, "Batch size mismatch");
    TEST_ASSERT(hidden.size(1) == config.hidden_size, "Hidden size mismatch");
    
    auto logits = model.compute_logits(hidden);
    
    TEST_ASSERT(logits.size(0) == batch_size * seq_len, "Logits batch size mismatch");
    TEST_ASSERT(logits.size(1) == config.vocab_size, "Vocab size mismatch");
    
    std::cout << "  MiMoV2Flash Model: PASSED" << std::endl;
    return true;
}

int main() {
    std::cout << "vLLM C++ Model Tests" << std::endl;
    std::cout << "====================" << std::endl;
    std::cout << std::endl;
    
    // Initialize CUDA
    torch::cuda::set_device(0);
    
    int passed = 0;
    int failed = 0;
    
    if (test_qwen2_decoder_layer()) passed++; else failed++;
    if (test_qwen2_model()) passed++; else failed++;
    if (test_mimo_v2_flash_decoder_layer()) passed++; else failed++;
    if (test_mimo_v2_flash_model()) passed++; else failed++;
    
    std::cout << std::endl;
    std::cout << "Results: " << passed << " passed, " << failed << " failed" << std::endl;
    
    return (failed > 0) ? 1 : 0;
}
