// SPDX-License-Identifier: Apache-2.0
// vLLM C++ - Qwen2 Model Example

#include <iostream>
#include <vector>

#include "vllm/common.h"
#include "vllm/core/config.h"
#include "vllm/core/kv_cache.h"
#include "vllm/model_executor/models/qwen2.h"

using namespace vllm;

int main(int argc, char** argv) {
    std::cout << "vLLM C++ Qwen2 Model Example" << std::endl;
    std::cout << "============================" << std::endl;
    
    try {
        // Initialize CUDA
        torch::cuda::set_device(0);
        
        // Create Qwen2 configuration
        Qwen2Config config;
        config.hidden_size = 4096;
        config.intermediate_size = 11008;
        config.num_hidden_layers = 32;
        config.num_attention_heads = 32;
        config.num_key_value_heads = 8;  // GQA with 8 KV heads
        config.vocab_size = 151936;
        config.max_position_embeddings = 32768;
        config.rms_norm_eps = 1e-6f;
        config.rope_theta = 1000000.0f;
        config.hidden_act = "silu";
        config.dtype = DataType::kFloat16;
        config.compute_derived();
        
        std::cout << "Creating Qwen2 model..." << std::endl;
        std::cout << "  Hidden size: " << config.hidden_size << std::endl;
        std::cout << "  Num layers: " << config.num_hidden_layers << std::endl;
        std::cout << "  Num heads: " << config.num_attention_heads << std::endl;
        std::cout << "  Num KV heads: " << config.num_key_value_heads << std::endl;
        std::cout << "  Head dim: " << config.head_dim << std::endl;
        std::cout << std::endl;
        
        // Create model
        CacheConfig cache_config;
        cache_config.block_size = 16;
        
        Qwen2ForCausalLM model(config, &cache_config, /*tp_size=*/1);
        
        std::cout << "Model created successfully!" << std::endl;
        
        // Create sample input
        int batch_size = 2;
        int seq_len = 10;
        
        auto input_options = torch::TensorOptions()
            .dtype(torch::kInt32)
            .device(torch::kCUDA);
        
        auto input_ids = torch::randint(0, config.vocab_size, {batch_size, seq_len}, input_options);
        auto positions = torch::arange(0, seq_len, input_options).unsqueeze(0).expand({batch_size, seq_len});
        
        std::cout << "Input shape: [" << batch_size << ", " << seq_len << "]" << std::endl;
        
        // Forward pass
        std::cout << "Running forward pass..." << std::endl;
        
        auto hidden_states = model.forward(
            input_ids.view({-1}),  // Flatten
            positions.view({-1})
        );
        
        std::cout << "Hidden states shape: ";
        for (int i = 0; i < hidden_states.dim(); ++i) {
            std::cout << hidden_states.size(i);
            if (i < hidden_states.dim() - 1) std::cout << " x ";
        }
        std::cout << std::endl;
        
        // Compute logits
        auto logits = model.compute_logits(hidden_states);
        
        std::cout << "Logits shape: ";
        for (int i = 0; i < logits.dim(); ++i) {
            std::cout << logits.size(i);
            if (i < logits.dim() - 1) std::cout << " x ";
        }
        std::cout << std::endl;
        
        // Get top predictions
        auto [top_values, top_indices] = torch::topk(logits, 5, -1);
        
        std::cout << std::endl;
        std::cout << "Top 5 predictions for first token:" << std::endl;
        auto indices = top_indices[0].cpu();
        auto values = top_values[0].cpu();
        for (int i = 0; i < 5; ++i) {
            std::cout << "  Token " << indices[i].item<int>() 
                     << ": logit = " << values[i].item<float>() << std::endl;
        }
        
        std::cout << std::endl;
        std::cout << "Qwen2 example complete!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
