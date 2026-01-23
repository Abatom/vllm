// SPDX-License-Identifier: Apache-2.0
// vLLM C++ Implementation - MiMoV2Flash Model (Mixture of Experts)
#pragma once

#include "vllm/common.h"
#include "vllm/core/config.h"
#include "vllm/core/kv_cache.h"
#include "vllm/model_executor/layers/linear.h"
#include "vllm/model_executor/layers/layernorm.h"
#include "vllm/model_executor/layers/activation.h"
#include "vllm/model_executor/layers/rotary_embedding.h"
#include "vllm/model_executor/layers/attention.h"
#include "vllm/model_executor/layers/mlp.h"

#include <memory>
#include <vector>

namespace vllm {

// MiMoV2Flash Model Configuration
struct MiMoV2FlashConfig {
    int hidden_size = 4096;
    int intermediate_size = 11008;
    int moe_intermediate_size = 2816;
    int num_hidden_layers = 32;
    int num_attention_heads = 32;
    int num_key_value_heads = 8;
    int vocab_size = 102400;
    int max_position_embeddings = 32768;
    float layernorm_epsilon = 1e-6f;
    float rope_theta = 10000.0f;
    std::string hidden_act = "silu";
    int head_dim = 128;
    int v_head_dim = 128;  // May differ from head_dim
    float attention_value_scale = 1.0f;
    bool attention_bias = false;
    float partial_rotary_factor = 1.0f;
    
    // MoE parameters
    int n_routed_experts = 64;
    int num_experts_per_tok = 8;
    int n_group = 8;
    int topk_group = 4;
    bool norm_topk_prob = true;
    
    // Sliding window attention parameters
    int sliding_window_size = 4096;
    int swa_num_attention_heads = 32;
    int swa_num_key_value_heads = 8;
    int swa_head_dim = 128;
    int swa_v_head_dim = 64;
    float swa_rope_theta = 10000.0f;
    bool add_swa_attention_sink_bias = false;
    
    // Hybrid layer pattern (0 = normal attention, 1 = sliding window)
    std::vector<int> hybrid_layer_pattern;
    
    // MoE layer frequency pattern
    std::vector<bool> moe_layer_freq;
    
    DataType dtype = DataType::kFloat16;
    
    void compute_derived() {
        // Initialize patterns if empty
        if (hybrid_layer_pattern.empty()) {
            hybrid_layer_pattern.resize(num_hidden_layers, 0);
        }
        if (moe_layer_freq.empty()) {
            moe_layer_freq.resize(num_hidden_layers, true);
            // First layer typically not MoE
            if (!moe_layer_freq.empty()) {
                moe_layer_freq[0] = false;
            }
        }
    }
    
    static MiMoV2FlashConfig from_model_config(const ModelConfig& config) {
        MiMoV2FlashConfig mimo_config;
        mimo_config.hidden_size = config.hidden_size;
        mimo_config.intermediate_size = config.intermediate_size;
        mimo_config.moe_intermediate_size = config.moe_intermediate_size > 0 ? 
                                            config.moe_intermediate_size : config.intermediate_size;
        mimo_config.num_hidden_layers = config.num_hidden_layers;
        mimo_config.num_attention_heads = config.num_attention_heads;
        mimo_config.num_key_value_heads = config.num_key_value_heads;
        mimo_config.vocab_size = config.vocab_size;
        mimo_config.max_position_embeddings = config.max_position_embeddings;
        mimo_config.layernorm_epsilon = config.rms_norm_eps;
        mimo_config.rope_theta = config.rope_theta;
        mimo_config.hidden_act = config.hidden_act;
        mimo_config.n_routed_experts = config.n_routed_experts;
        mimo_config.num_experts_per_tok = config.num_experts_per_tok;
        mimo_config.dtype = config.dtype;
        mimo_config.compute_derived();
        return mimo_config;
    }
};

// MiMoV2 Attention Module
class MiMoV2Attention {
public:
    MiMoV2Attention(
        const MiMoV2FlashConfig& config,
        int layer_idx,
        bool is_sliding_window = false,
        const CacheConfig* cache_config = nullptr,
        int tp_size = 1,
        const std::string& prefix = ""
    ) : config_(config), layer_idx_(layer_idx), 
        is_sliding_window_(is_sliding_window), tp_size_(tp_size), prefix_(prefix) {
        
        // Use SWA or normal attention parameters
        int num_heads = is_sliding_window ? config.swa_num_attention_heads : config.num_attention_heads;
        int num_kv_heads = is_sliding_window ? config.swa_num_key_value_heads : config.num_key_value_heads;
        int head_dim = is_sliding_window ? config.swa_head_dim : config.head_dim;
        int v_head_dim = is_sliding_window ? config.swa_v_head_dim : config.v_head_dim;
        float rope_theta = is_sliding_window ? config.swa_rope_theta : config.rope_theta;
        
        num_heads_ = num_heads / tp_size;
        num_kv_heads_ = std::max(1, num_kv_heads / tp_size);
        head_dim_ = head_dim;
        v_head_dim_ = v_head_dim;
        
        q_size_ = num_heads_ * head_dim;
        k_size_ = num_kv_heads_ * head_dim;
        v_size_ = num_kv_heads_ * v_head_dim;
        
        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        
        // QKV projection with different V head dim
        qkv_proj_ = std::make_unique<QKVParallelLinear>(
            config.hidden_size,
            head_dim,
            num_heads,
            num_kv_heads,
            config.attention_bias,
            tp_size,
            config.dtype
        );
        
        // Output projection
        o_proj_ = std::make_unique<RowParallelLinear>(
            num_heads * v_head_dim,
            config.hidden_size,
            /*bias=*/false,
            tp_size,
            /*reduce_results=*/true,
            config.dtype
        );
        
        // Rotary embedding
        rotary_emb_ = std::make_unique<RotaryEmbedding>(
            head_dim,
            config.max_position_embeddings,
            rope_theta,
            /*is_neox=*/true,
            config.partial_rotary_factor,
            config.dtype
        );
        
        // Attention
        attn_ = std::make_unique<Attention>(
            num_heads_,
            head_dim,
            scale,
            num_kv_heads_,
            cache_config
        );
        
        // Attention sink bias (for sliding window attention)
        if (is_sliding_window && config.add_swa_attention_sink_bias) {
            auto options = torch::TensorOptions()
                .dtype(to_torch_dtype(config.dtype))
                .device(torch::kCUDA);
            attention_sink_bias_ = torch::zeros({num_heads_}, options);
        }
        
        v_scale_ = config.attention_value_scale;
    }
    
    torch::Tensor forward(
        const torch::Tensor& positions,
        const torch::Tensor& hidden_states,
        LayerKVCache* kv_cache = nullptr,
        const AttentionMetadata* attn_metadata = nullptr
    ) {
        // QKV projection
        auto [q, k, v] = qkv_proj_->forward(hidden_states);
        
        // Apply rotary embedding
        std::tie(q, k) = rotary_emb_->forward(positions, q, k);
        
        // Apply V scale
        if (v_scale_ != 1.0f) {
            v = v * v_scale_;
        }
        
        // Pad V if v_head_dim differs from head_dim
        if (v_head_dim_ < head_dim_) {
            auto v_shape = v.sizes().vec();
            v = v.view({-1, num_kv_heads_, v_head_dim_});
            v = torch::nn::functional::pad(
                v, 
                torch::nn::functional::PadFuncOptions({0, head_dim_ - v_head_dim_})
            );
            v = v.view({v_shape[0], num_kv_heads_ * head_dim_});
        }
        
        // Attention
        torch::Tensor attn_output;
        if (kv_cache && attn_metadata) {
            attn_output = attn_->forward(q, k, v, kv_cache, *attn_metadata);
        } else {
            attn_output = attn_->forward(q, k, v);
        }
        
        // Truncate output if v_head_dim differs
        if (v_head_dim_ < head_dim_) {
            attn_output = attn_output.view({-1, num_heads_, head_dim_})
                                     .narrow(-1, 0, v_head_dim_)
                                     .reshape({-1, num_heads_ * v_head_dim_});
        }
        
        // Output projection
        auto output = o_proj_->forward(attn_output);
        
        return output;
    }
    
private:
    MiMoV2FlashConfig config_;
    int layer_idx_;
    bool is_sliding_window_;
    int tp_size_;
    std::string prefix_;
    int num_heads_;
    int num_kv_heads_;
    int head_dim_;
    int v_head_dim_;
    int q_size_;
    int k_size_;
    int v_size_;
    float v_scale_;
    
    std::unique_ptr<QKVParallelLinear> qkv_proj_;
    std::unique_ptr<RowParallelLinear> o_proj_;
    std::unique_ptr<RotaryEmbedding> rotary_emb_;
    std::unique_ptr<Attention> attn_;
    torch::Tensor attention_sink_bias_;
};

// MiMoV2 MLP (Dense, non-MoE)
class MiMoV2MLP {
public:
    MiMoV2MLP(
        const MiMoV2FlashConfig& config,
        int tp_size = 1,
        const std::string& prefix = ""
    ) : mlp_(
            config.hidden_size,
            config.intermediate_size,
            config.hidden_act,
            0.0f,
            tp_size,
            config.dtype
        ) {}
    
    torch::Tensor forward(const torch::Tensor& x) {
        return mlp_.forward(x);
    }
    
private:
    MLP mlp_;
};

// MiMoV2 MoE Layer
class MiMoV2MoE {
public:
    MiMoV2MoE(
        const MiMoV2FlashConfig& config,
        int tp_size = 1,
        const std::string& prefix = ""
    ) : moe_(
            config.hidden_size,
            config.moe_intermediate_size,
            config.n_routed_experts,
            config.num_experts_per_tok,
            config.norm_topk_prob,
            tp_size,
            config.dtype
        ), config_(config), tp_size_(tp_size) {
        
        // E-score correction bias
        auto options = torch::TensorOptions()
            .dtype(torch::kFloat32)  // Router typically uses float32
            .device(torch::kCUDA);
        e_score_correction_bias_ = torch::zeros({config.n_routed_experts}, options);
    }
    
    torch::Tensor forward(const torch::Tensor& hidden_states) {
        return moe_.forward(hidden_states);
    }
    
    void load_gate_weight(const torch::Tensor& weight) {
        moe_.load_gate_weight(weight);
    }
    
    void load_e_score_bias(const torch::Tensor& bias) {
        e_score_correction_bias_.copy_(bias);
    }
    
private:
    MoE moe_;
    MiMoV2FlashConfig config_;
    int tp_size_;
    torch::Tensor e_score_correction_bias_;
};

// MiMoV2Flash Decoder Layer
class MiMoV2FlashDecoderLayer {
public:
    MiMoV2FlashDecoderLayer(
        const MiMoV2FlashConfig& config,
        int layer_idx,
        const CacheConfig* cache_config = nullptr,
        int tp_size = 1,
        const std::string& prefix = ""
    ) : config_(config), layer_idx_(layer_idx), prefix_(prefix) {
        
        // Determine if this is a sliding window attention layer
        bool is_swa = false;
        if (layer_idx < static_cast<int>(config.hybrid_layer_pattern.size())) {
            is_swa = (config.hybrid_layer_pattern[layer_idx] == 1);
        }
        
        // Self attention
        self_attn_ = std::make_unique<MiMoV2Attention>(
            config, layer_idx, is_swa, cache_config, tp_size, 
            prefix + ".self_attn"
        );
        
        // Determine if this is an MoE layer
        bool is_moe = false;
        if (layer_idx < static_cast<int>(config.moe_layer_freq.size())) {
            is_moe = config.moe_layer_freq[layer_idx];
        }
        is_moe_layer_ = is_moe;
        
        // MLP (MoE or dense)
        if (is_moe) {
            moe_ = std::make_unique<MiMoV2MoE>(config, tp_size, prefix + ".mlp");
        } else {
            mlp_ = std::make_unique<MiMoV2MLP>(config, tp_size, prefix + ".mlp");
        }
        
        // Layer norms
        input_layernorm_ = std::make_unique<RMSNorm>(
            config.hidden_size, config.layernorm_epsilon, config.dtype
        );
        post_attention_layernorm_ = std::make_unique<RMSNorm>(
            config.hidden_size, config.layernorm_epsilon, config.dtype
        );
    }
    
    std::tuple<torch::Tensor, torch::Tensor> forward(
        const torch::Tensor& positions,
        torch::Tensor hidden_states,
        torch::Tensor residual,
        LayerKVCache* kv_cache = nullptr,
        const AttentionMetadata* attn_metadata = nullptr
    ) {
        // Self attention with residual
        if (!residual.defined()) {
            residual = hidden_states;
            hidden_states = input_layernorm_->forward(hidden_states);
        } else {
            std::tie(hidden_states, residual) = input_layernorm_->forward(hidden_states, residual);
        }
        
        hidden_states = self_attn_->forward(positions, hidden_states, kv_cache, attn_metadata);
        
        // MLP/MoE with residual
        std::tie(hidden_states, residual) = post_attention_layernorm_->forward(hidden_states, residual);
        
        if (is_moe_layer_) {
            hidden_states = moe_->forward(hidden_states);
        } else {
            hidden_states = mlp_->forward(hidden_states);
        }
        
        return std::make_tuple(hidden_states, residual);
    }
    
    bool is_moe_layer() const { return is_moe_layer_; }
    
private:
    MiMoV2FlashConfig config_;
    int layer_idx_;
    std::string prefix_;
    bool is_moe_layer_ = false;
    
    std::unique_ptr<MiMoV2Attention> self_attn_;
    std::unique_ptr<MiMoV2MLP> mlp_;
    std::unique_ptr<MiMoV2MoE> moe_;
    std::unique_ptr<RMSNorm> input_layernorm_;
    std::unique_ptr<RMSNorm> post_attention_layernorm_;
};

// MiMoV2 Model (transformer backbone)
class MiMoV2Model {
public:
    MiMoV2Model(
        const MiMoV2FlashConfig& config,
        const CacheConfig* cache_config = nullptr,
        int tp_size = 1,
        const std::string& prefix = "model"
    ) : config_(config), prefix_(prefix) {
        
        auto options = torch::TensorOptions()
            .dtype(to_torch_dtype(config.dtype))
            .device(torch::kCUDA);
        
        // Token embedding
        embed_tokens_ = torch::empty({config.vocab_size, config.hidden_size}, options);
        
        // Decoder layers
        for (int i = 0; i < config.num_hidden_layers; ++i) {
            layers_.push_back(std::make_unique<MiMoV2FlashDecoderLayer>(
                config, i, cache_config, tp_size, prefix
            ));
        }
        
        // Final layer norm
        norm_ = std::make_unique<RMSNorm>(config.hidden_size, config.layernorm_epsilon, config.dtype);
    }
    
    torch::Tensor forward(
        const torch::Tensor& input_ids,
        const torch::Tensor& positions,
        KVCacheManager* kv_cache_manager = nullptr,
        const AttentionMetadata* attn_metadata = nullptr,
        const torch::Tensor& inputs_embeds = torch::Tensor()
    ) {
        // Embedding
        torch::Tensor hidden_states;
        if (inputs_embeds.defined()) {
            hidden_states = inputs_embeds;
        } else {
            hidden_states = torch::embedding(embed_tokens_, input_ids);
        }
        
        torch::Tensor residual;
        
        // Apply decoder layers
        for (int i = 0; i < static_cast<int>(layers_.size()); ++i) {
            LayerKVCache* layer_cache = nullptr;
            if (kv_cache_manager) {
                layer_cache = kv_cache_manager->get_layer_cache(i);
            }
            
            std::tie(hidden_states, residual) = layers_[i]->forward(
                positions, hidden_states, residual, layer_cache, attn_metadata
            );
        }
        
        // Final norm
        std::tie(hidden_states, residual) = norm_->forward(hidden_states, residual);
        
        return hidden_states;
    }
    
    void load_embed_tokens(const torch::Tensor& weight) {
        embed_tokens_.copy_(weight);
    }
    
    torch::Tensor& embed_tokens() { return embed_tokens_; }
    std::vector<std::unique_ptr<MiMoV2FlashDecoderLayer>>& layers() { return layers_; }
    RMSNorm* norm() { return norm_.get(); }
    
private:
    MiMoV2FlashConfig config_;
    std::string prefix_;
    torch::Tensor embed_tokens_;
    std::vector<std::unique_ptr<MiMoV2FlashDecoderLayer>> layers_;
    std::unique_ptr<RMSNorm> norm_;
};

// MiMoV2Flash For Causal LM
class MiMoV2FlashForCausalLM {
public:
    MiMoV2FlashForCausalLM(
        const MiMoV2FlashConfig& config,
        const CacheConfig* cache_config = nullptr,
        int tp_size = 1
    ) : config_(config) {
        
        // Transformer model
        model_ = std::make_unique<MiMoV2Model>(config, cache_config, tp_size);
        
        // LM head
        auto options = torch::TensorOptions()
            .dtype(to_torch_dtype(config.dtype))
            .device(torch::kCUDA);
        
        lm_head_weight_ = torch::empty({config.vocab_size, config.hidden_size}, options);
    }
    
    torch::Tensor forward(
        const torch::Tensor& input_ids,
        const torch::Tensor& positions,
        KVCacheManager* kv_cache_manager = nullptr,
        const AttentionMetadata* attn_metadata = nullptr,
        const torch::Tensor& inputs_embeds = torch::Tensor()
    ) {
        return model_->forward(input_ids, positions, kv_cache_manager, attn_metadata, inputs_embeds);
    }
    
    torch::Tensor compute_logits(const torch::Tensor& hidden_states) {
        return torch::matmul(hidden_states, lm_head_weight_.t());
    }
    
    void load_weights(const std::unordered_map<std::string, torch::Tensor>& weights) {
        for (const auto& [name, weight] : weights) {
            if (name.find("embed_tokens") != std::string::npos) {
                model_->load_embed_tokens(weight);
            } else if (name.find("lm_head") != std::string::npos) {
                lm_head_weight_.copy_(weight);
            } else if (name.find("norm.weight") != std::string::npos &&
                       name.find("layernorm") == std::string::npos) {
                model_->norm()->load_weight(weight);
            }
        }
    }
    
    MiMoV2Model* model() { return model_.get(); }
    const MiMoV2FlashConfig& config() const { return config_; }
    
private:
    MiMoV2FlashConfig config_;
    std::unique_ptr<MiMoV2Model> model_;
    torch::Tensor lm_head_weight_;
};

}  // namespace vllm
