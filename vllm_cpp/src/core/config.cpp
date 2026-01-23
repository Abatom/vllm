// SPDX-License-Identifier: Apache-2.0
// vLLM C++ Implementation - Configuration Implementation

#include "vllm/core/config.h"
#include <fstream>
#include <iostream>

namespace vllm {

#if VLLM_HAS_JSON
// Load model config from HuggingFace config.json
ModelConfig load_model_config_from_json(const std::string& json_path) {
    ModelConfig config;
    
    std::ifstream file(json_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open config file: " + json_path);
    }
    
    nlohmann::json j;
    file >> j;
    
    // Parse common fields
    if (j.contains("hidden_size")) {
        config.hidden_size = j["hidden_size"].get<int>();
    }
    if (j.contains("intermediate_size")) {
        config.intermediate_size = j["intermediate_size"].get<int>();
    }
    if (j.contains("num_hidden_layers")) {
        config.num_hidden_layers = j["num_hidden_layers"].get<int>();
    }
    if (j.contains("num_attention_heads")) {
        config.num_attention_heads = j["num_attention_heads"].get<int>();
    }
    if (j.contains("num_key_value_heads")) {
        config.num_key_value_heads = j["num_key_value_heads"].get<int>();
    } else {
        config.num_key_value_heads = config.num_attention_heads;
    }
    if (j.contains("vocab_size")) {
        config.vocab_size = j["vocab_size"].get<int>();
    }
    if (j.contains("max_position_embeddings")) {
        config.max_position_embeddings = j["max_position_embeddings"].get<int>();
    }
    if (j.contains("rms_norm_eps")) {
        config.rms_norm_eps = j["rms_norm_eps"].get<float>();
    }
    if (j.contains("rope_theta")) {
        config.rope_theta = j["rope_theta"].get<float>();
    }
    if (j.contains("hidden_act")) {
        config.hidden_act = j["hidden_act"].get<std::string>();
    }
    if (j.contains("tie_word_embeddings")) {
        config.tie_word_embeddings = j["tie_word_embeddings"].get<bool>();
    }
    
    // MoE fields
    if (j.contains("num_experts") || j.contains("n_routed_experts")) {
        config.n_routed_experts = j.value("n_routed_experts", j.value("num_experts", 0));
    }
    if (j.contains("num_experts_per_tok")) {
        config.num_experts_per_tok = j["num_experts_per_tok"].get<int>();
    }
    if (j.contains("moe_intermediate_size")) {
        config.moe_intermediate_size = j["moe_intermediate_size"].get<int>();
    }
    
    // Get model type from architectures
    if (j.contains("architectures") && !j["architectures"].empty()) {
        std::string arch = j["architectures"][0].get<std::string>();
        if (arch.find("Qwen2") != std::string::npos) {
            config.model_name = "qwen2";
        } else if (arch.find("MiMo") != std::string::npos) {
            config.model_name = "mimo_v2_flash";
        }
    }
    
    config.compute_derived();
    return config;
}
#else
// Stub when nlohmann/json is not available
ModelConfig load_model_config_from_json(const std::string& json_path) {
    throw std::runtime_error("JSON support not available. Please install nlohmann/json library.");
}
#endif

}  // namespace vllm
