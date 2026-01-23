// SPDX-License-Identifier: Apache-2.0
// vLLM C++ Implementation - MLP Layers
#pragma once

#include "vllm/common.h"
#include "vllm/core/config.h"
#include "vllm/model_executor/layers/linear.h"
#include "vllm/model_executor/layers/activation.h"

namespace vllm {

// Standard MLP with gate/up projection and down projection
class MLP {
public:
    MLP(
        int hidden_size,
        int intermediate_size,
        const std::string& hidden_act = "silu",
        float hidden_act_param = 0.0f,
        int tp_size = 1,
        DataType dtype = DataType::kFloat16
    ) : hidden_size_(hidden_size),
        intermediate_size_(intermediate_size),
        tp_size_(tp_size),
        dtype_(dtype) {
        
        // Gate and up projections merged
        gate_up_proj_ = std::make_unique<MergedColumnParallelLinear>(
            hidden_size,
            std::vector<int>{intermediate_size, intermediate_size},
            /*bias=*/false,
            tp_size,
            dtype
        );
        
        // Down projection
        down_proj_ = std::make_unique<RowParallelLinear>(
            intermediate_size,
            hidden_size,
            /*bias=*/false,
            tp_size,
            /*reduce_results=*/true,
            dtype
        );
        
        // Activation function
        act_fn_ = create_activation(hidden_act, hidden_act_param);
    }
    
    torch::Tensor forward(const torch::Tensor& x) const {
        // gate_up: [batch, seq, 2 * intermediate_size]
        auto gate_up = gate_up_proj_->forward(x);
        
        // Apply activation (SiLU + mul)
        auto hidden = act_fn_->forward(gate_up);
        
        // Down projection
        auto output = down_proj_->forward(hidden);
        
        return output;
    }
    
    // Weight loading
    void load_gate_weight(const torch::Tensor& weight) {
        gate_up_proj_->load_shard_weight(weight, 0);
    }
    
    void load_up_weight(const torch::Tensor& weight) {
        gate_up_proj_->load_shard_weight(weight, 1);
    }
    
    void load_down_weight(const torch::Tensor& weight) {
        down_proj_->load_weight(weight);
    }
    
    MergedColumnParallelLinear* gate_up_proj() { return gate_up_proj_.get(); }
    RowParallelLinear* down_proj() { return down_proj_.get(); }
    
private:
    int hidden_size_;
    int intermediate_size_;
    int tp_size_;
    DataType dtype_;
    std::unique_ptr<MergedColumnParallelLinear> gate_up_proj_;
    std::unique_ptr<RowParallelLinear> down_proj_;
    std::unique_ptr<ActivationBase> act_fn_;
};

// MoE (Mixture of Experts) Layer
class MoE {
public:
    MoE(
        int hidden_size,
        int intermediate_size,
        int num_experts,
        int top_k,
        bool normalize_topk = true,
        int tp_size = 1,
        DataType dtype = DataType::kFloat16
    ) : hidden_size_(hidden_size),
        intermediate_size_(intermediate_size),
        num_experts_(num_experts),
        top_k_(top_k),
        normalize_topk_(normalize_topk),
        tp_size_(tp_size),
        dtype_(dtype) {
        
        initialize();
    }
    
    void initialize() {
        auto options = torch::TensorOptions()
            .dtype(to_torch_dtype(dtype_))
            .device(torch::kCUDA);
        
        // Gate/router
        gate_weight_ = torch::empty({num_experts_, hidden_size_}, options);
        
        // Expert weights (fused for efficiency)
        int local_intermediate = intermediate_size_ / tp_size_;
        
        // ws: [num_experts, 2 * local_intermediate, hidden_size] - gate and up combined
        ws_ = torch::empty({num_experts_, 2 * local_intermediate, hidden_size_}, options);
        
        // w2s: [num_experts, hidden_size, local_intermediate] - down projection
        w2s_ = torch::empty({num_experts_, hidden_size_, local_intermediate}, options);
    }
    
    torch::Tensor forward(const torch::Tensor& hidden_states) const {
        auto shape = hidden_states.sizes().vec();
        int64_t num_tokens = hidden_states.numel() / hidden_size_;
        auto x = hidden_states.view({num_tokens, hidden_size_});
        
        // Compute router logits
        auto router_logits = torch::matmul(x, gate_weight_.t());
        
        // Top-k routing
        auto [topk_weights, topk_ids] = torch::topk(router_logits, top_k_, /*dim=*/-1);
        
        if (normalize_topk_) {
            topk_weights = torch::softmax(topk_weights, -1);
        }
        
        // Compute expert outputs (simplified - real impl uses fused kernels)
        torch::Tensor output = torch::zeros_like(x);
        
        for (int k = 0; k < top_k_; ++k) {
            auto expert_ids = topk_ids.select(-1, k);
            auto weights = topk_weights.select(-1, k).unsqueeze(-1);
            
            for (int e = 0; e < num_experts_; ++e) {
                auto mask = (expert_ids == e);
                if (!mask.any().item<bool>()) continue;
                
                auto x_e = x.index({mask});
                
                // Gate + Up
                auto gate_up = torch::matmul(x_e, ws_[e].t());
                int64_t half = gate_up.size(-1) / 2;
                auto gate = gate_up.narrow(-1, 0, half);
                auto up = gate_up.narrow(-1, half, half);
                
                // SiLU activation
                auto hidden = torch::silu(gate) * up;
                
                // Down projection
                auto expert_out = torch::matmul(hidden, w2s_[e].t());
                
                // Add weighted output
                output.index_put_({mask}, output.index({mask}) + expert_out * weights.index({mask}));
            }
        }
        
        return output.view(shape);
    }
    
    // Weight loading
    void load_gate_weight(const torch::Tensor& weight) {
        gate_weight_.copy_(weight);
    }
    
    void load_expert_weight(const torch::Tensor& weight, const std::string& name, int expert_id) {
        int local_intermediate = intermediate_size_ / tp_size_;
        
        if (name == "w1" || name == "gate_proj") {
            // Gate projection: first half of ws
            ws_[expert_id].narrow(0, 0, local_intermediate).copy_(weight);
        } else if (name == "w3" || name == "up_proj") {
            // Up projection: second half of ws
            ws_[expert_id].narrow(0, local_intermediate, local_intermediate).copy_(weight);
        } else if (name == "w2" || name == "down_proj") {
            // Down projection
            w2s_[expert_id].copy_(weight);
        }
    }
    
private:
    int hidden_size_;
    int intermediate_size_;
    int num_experts_;
    int top_k_;
    bool normalize_topk_;
    int tp_size_;
    DataType dtype_;
    
    torch::Tensor gate_weight_;
    torch::Tensor ws_;   // Combined gate + up weights
    torch::Tensor w2s_;  // Down weights
};

}  // namespace vllm
