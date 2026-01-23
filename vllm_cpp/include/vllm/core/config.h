// SPDX-License-Identifier: Apache-2.0
// vLLM C++ Implementation - Configuration Classes
#pragma once

#include "vllm/common.h"
#include <nlohmann/json.hpp>

namespace vllm {

// Model configuration
struct ModelConfig {
    std::string model_name;
    std::string model_path;
    int hidden_size = 4096;
    int intermediate_size = 11008;
    int num_hidden_layers = 32;
    int num_attention_heads = 32;
    int num_key_value_heads = 32;  // For GQA models
    int vocab_size = 32000;
    int max_position_embeddings = 4096;
    float rms_norm_eps = 1e-6f;
    float rope_theta = 10000.0f;
    int head_dim = 0;  // Computed if 0: hidden_size / num_attention_heads
    std::string hidden_act = "silu";
    bool tie_word_embeddings = false;
    DataType dtype = DataType::kFloat16;
    std::string quantization;  // Empty for no quantization
    
    // MoE parameters (for MiMoV2Flash)
    int n_routed_experts = 0;
    int num_experts_per_tok = 0;
    int moe_intermediate_size = 0;
    bool norm_topk_prob = true;
    
    // Compute derived values
    void compute_derived() {
        if (head_dim == 0) {
            head_dim = hidden_size / num_attention_heads;
        }
    }
    
    // Load from HuggingFace config.json
    static ModelConfig from_json(const std::string& json_path);
    
    // Serialize to JSON
    nlohmann::json to_json() const;
};

// Cache configuration
struct CacheConfig {
    int block_size = 16;
    float gpu_memory_utilization = 0.9f;
    int num_gpu_blocks = 0;  // Auto-computed if 0
    std::string cache_dtype = "auto";  // "auto", "fp16", "fp8"
    bool enable_prefix_caching = true;
    
    int get_num_kv_heads(const ModelConfig& model_config) const {
        return model_config.num_key_value_heads;
    }
    
    size_t get_block_size_bytes(const ModelConfig& model_config) const {
        // 2 for K and V, num_layers, head_dim, dtype_size
        int num_kv_heads = get_num_kv_heads(model_config);
        int head_dim = model_config.head_dim;
        int num_layers = model_config.num_hidden_layers;
        size_t dtype_sz = (cache_dtype == "fp8") ? 1 : 2;  // fp8 or fp16
        return 2 * num_layers * num_kv_heads * head_dim * block_size * dtype_sz;
    }
};

// Parallel configuration
struct ParallelConfig {
    int tensor_parallel_size = 1;
    int pipeline_parallel_size = 1;
    int data_parallel_size = 1;
    
    int world_size() const {
        return tensor_parallel_size * pipeline_parallel_size * data_parallel_size;
    }
};

// Scheduler configuration
struct SchedulerConfig {
    int max_num_seqs = 256;
    int max_num_batched_tokens = 8192;
    int max_model_len = 0;  // Use model's max_position_embeddings if 0
    bool enable_chunked_prefill = true;
    int max_num_partial_prefills = 1;
    float delay_factor = 0.0f;
    bool enable_prefix_caching = true;
};

// Speculative decoding configuration
struct SpecDecodeConfig {
    bool enabled = false;
    std::string draft_model_path;
    int num_speculative_tokens = 5;
    float acceptance_threshold = 0.0f;
};

// Main vLLM configuration
struct VllmConfig {
    ModelConfig model_config;
    CacheConfig cache_config;
    ParallelConfig parallel_config;
    SchedulerConfig scheduler_config;
    SpecDecodeConfig spec_decode_config;
    
    int device_id = 0;
    std::string device_type = "cuda";
    bool enforce_eager = false;  // Disable CUDA graphs
    int max_seq_len_to_capture = 8192;  // Max seq len for CUDA graphs
    
    static VllmConfig create_default(const std::string& model_path);
};

// Implementation of ModelConfig::from_json
inline ModelConfig ModelConfig::from_json(const std::string& json_path) {
    // TODO: Implement JSON parsing
    ModelConfig config;
    // This would read from config.json
    return config;
}

inline nlohmann::json ModelConfig::to_json() const {
    nlohmann::json j;
    j["model_name"] = model_name;
    j["hidden_size"] = hidden_size;
    j["intermediate_size"] = intermediate_size;
    j["num_hidden_layers"] = num_hidden_layers;
    j["num_attention_heads"] = num_attention_heads;
    j["num_key_value_heads"] = num_key_value_heads;
    j["vocab_size"] = vocab_size;
    j["max_position_embeddings"] = max_position_embeddings;
    j["rms_norm_eps"] = rms_norm_eps;
    j["rope_theta"] = rope_theta;
    j["hidden_act"] = hidden_act;
    j["tie_word_embeddings"] = tie_word_embeddings;
    return j;
}

inline VllmConfig VllmConfig::create_default(const std::string& model_path) {
    VllmConfig config;
    config.model_config.model_path = model_path;
    config.model_config.compute_derived();
    return config;
}

}  // namespace vllm
