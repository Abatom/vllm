// SPDX-License-Identifier: Apache-2.0
// vLLM C++ - MiMoV2Flash Model Example

#include <iostream>
#include <vector>

#include "vllm/common.h"
#include "vllm/core/config.h"
#include "vllm/core/kv_cache.h"
#include "vllm/model_executor/models/mimo_v2_flash.h"

using namespace vllm;

int main(int argc, char** argv) {
    std::cout << "vLLM C++ MiMoV2Flash Model Example" << std::endl;
    std::cout << "===================================" << std::endl;
    
    try {
        // Initialize CUDA
        torch::cuda::set_device(0);
        
        // Create MiMoV2Flash configuration
        MiMoV2FlashConfig config;
        config.hidden_size = 4096;
        config.intermediate_size = 11008;
        config.moe_intermediate_size = 2816;
        config.num_hidden_layers = 32;
        config.num_attention_heads = 32;
        config.num_key_value_heads = 8;
        config.vocab_size = 102400;
        config.max_position_embeddings = 32768;
        config.layernorm_epsilon = 1e-6f;
        config.rope_theta = 10000.0f;
        config.hidden_act = "silu";
        config.head_dim = 128;
        config.v_head_dim = 128;
        
        // MoE configuration
        config.n_routed_experts = 64;
        config.num_experts_per_tok = 8;
        config.n_group = 8;
        config.topk_group = 4;
        config.norm_topk_prob = true;
        
        // Sliding window configuration
        config.sliding_window_size = 4096;
        config.swa_num_attention_heads = 32;
        config.swa_num_key_value_heads = 8;
        config.swa_head_dim = 128;
        config.swa_v_head_dim = 64;
        
        config.dtype = DataType::kFloat16;
        
        // Set up layer patterns
        config.hybrid_layer_pattern.resize(config.num_hidden_layers, 0);
        config.moe_layer_freq.resize(config.num_hidden_layers, true);
        config.moe_layer_freq[0] = false;  // First layer is dense
        
        // Set some layers to use sliding window attention
        for (int i = 1; i < config.num_hidden_layers; i += 4) {
            config.hybrid_layer_pattern[i] = 1;  // SWA layer
        }
        
        config.compute_derived();
        
        std::cout << "Creating MiMoV2Flash model..." << std::endl;
        std::cout << "  Hidden size: " << config.hidden_size << std::endl;
        std::cout << "  Num layers: " << config.num_hidden_layers << std::endl;
        std::cout << "  Num experts: " << config.n_routed_experts << std::endl;
        std::cout << "  Top-k experts: " << config.num_experts_per_tok << std::endl;
        std::cout << "  Head dim: " << config.head_dim << std::endl;
        std::cout << "  V head dim: " << config.v_head_dim << std::endl;
        std::cout << std::endl;
        
        // Count layer types
        int num_swa_layers = 0;
        int num_moe_layers = 0;
        for (int i = 0; i < config.num_hidden_layers; ++i) {
            if (config.hybrid_layer_pattern[i] == 1) num_swa_layers++;
            if (config.moe_layer_freq[i]) num_moe_layers++;
        }
        std::cout << "  SWA layers: " << num_swa_layers << std::endl;
        std::cout << "  MoE layers: " << num_moe_layers << std::endl;
        std::cout << "  Dense layers: " << (config.num_hidden_layers - num_moe_layers) << std::endl;
        std::cout << std::endl;
        
        // Create model
        CacheConfig cache_config;
        cache_config.block_size = 16;
        
        MiMoV2FlashForCausalLM model(config, &cache_config, /*tp_size=*/1);
        
        std::cout << "Model created successfully!" << std::endl;
        
        // Create sample input
        int batch_size = 2;
        int seq_len = 8;
        
        auto input_options = torch::TensorOptions()
            .dtype(torch::kInt32)
            .device(torch::kCUDA);
        
        auto input_ids = torch::randint(0, config.vocab_size, {batch_size, seq_len}, input_options);
        auto positions = torch::arange(0, seq_len, input_options).unsqueeze(0).expand({batch_size, seq_len});
        
        std::cout << "Input shape: [" << batch_size << ", " << seq_len << "]" << std::endl;
        
        // Forward pass
        std::cout << "Running forward pass..." << std::endl;
        
        auto hidden_states = model.forward(
            input_ids.view({-1}),
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
        std::cout << "MiMoV2Flash example complete!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
