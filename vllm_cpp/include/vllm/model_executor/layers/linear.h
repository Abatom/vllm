// SPDX-License-Identifier: Apache-2.0
// vLLM C++ Implementation - Linear Layers
#pragma once

#include "vllm/common.h"
#include "vllm/core/config.h"

namespace vllm {

// Base linear layer
class Linear {
public:
    Linear(int in_features, int out_features, bool bias = true, DataType dtype = DataType::kFloat16)
        : in_features_(in_features), out_features_(out_features), has_bias_(bias), dtype_(dtype) {
        initialize();
    }
    
    void initialize() {
        auto options = torch::TensorOptions()
            .dtype(to_torch_dtype(dtype_))
            .device(torch::kCUDA);
        
        weight_ = torch::empty({out_features_, in_features_}, options);
        if (has_bias_) {
            bias_ = torch::empty({out_features_}, options);
        }
    }
    
    torch::Tensor forward(const torch::Tensor& input) const {
        torch::Tensor output = torch::linear(input, weight_, has_bias_ ? bias_ : torch::Tensor());
        return output;
    }
    
    // Weight loading
    void load_weight(const torch::Tensor& weight) {
        weight_.copy_(weight);
    }
    
    void load_bias(const torch::Tensor& bias) {
        if (has_bias_) {
            bias_.copy_(bias);
        }
    }
    
    torch::Tensor& weight() { return weight_; }
    const torch::Tensor& weight() const { return weight_; }
    torch::Tensor& bias() { return bias_; }
    const torch::Tensor& bias() const { return bias_; }
    
protected:
    int in_features_;
    int out_features_;
    bool has_bias_;
    DataType dtype_;
    torch::Tensor weight_;
    torch::Tensor bias_;
};

// Column parallel linear (for tensor parallelism)
class ColumnParallelLinear : public Linear {
public:
    ColumnParallelLinear(
        int in_features,
        int out_features,
        bool bias = true,
        int tp_size = 1,
        DataType dtype = DataType::kFloat16
    ) : Linear(in_features, out_features / tp_size, bias, dtype),
        total_out_features_(out_features),
        tp_size_(tp_size) {}
    
    // For loading sharded weights
    void load_shard(const torch::Tensor& weight, int shard_id) {
        int shard_size = out_features_;
        int start = shard_id * shard_size;
        weight_.copy_(weight.slice(0, start, start + shard_size));
    }
    
private:
    int total_out_features_;
    int tp_size_;
};

// Row parallel linear (for tensor parallelism)
class RowParallelLinear : public Linear {
public:
    RowParallelLinear(
        int in_features,
        int out_features,
        bool bias = true,
        int tp_size = 1,
        bool reduce_results = true,
        DataType dtype = DataType::kFloat16
    ) : Linear(in_features / tp_size, out_features, bias, dtype),
        total_in_features_(in_features),
        tp_size_(tp_size),
        reduce_results_(reduce_results) {}
    
    torch::Tensor forward(const torch::Tensor& input) const {
        torch::Tensor output = Linear::forward(input);
        // TODO: Add all-reduce for tensor parallelism if reduce_results_
        return output;
    }
    
private:
    int total_in_features_;
    int tp_size_;
    bool reduce_results_;
};

// Merged column parallel linear (for Q, K, V projections or gate+up)
class MergedColumnParallelLinear {
public:
    MergedColumnParallelLinear(
        int in_features,
        const std::vector<int>& out_features_list,
        bool bias = true,
        int tp_size = 1,
        DataType dtype = DataType::kFloat16
    ) : in_features_(in_features),
        out_features_list_(out_features_list),
        has_bias_(bias),
        tp_size_(tp_size),
        dtype_(dtype) {
        
        total_out_features_ = 0;
        for (int out : out_features_list) {
            total_out_features_ += out / tp_size;
        }
        
        initialize();
    }
    
    void initialize() {
        auto options = torch::TensorOptions()
            .dtype(to_torch_dtype(dtype_))
            .device(torch::kCUDA);
        
        weight_ = torch::empty({total_out_features_, in_features_}, options);
        if (has_bias_) {
            bias_ = torch::empty({total_out_features_}, options);
        }
    }
    
    torch::Tensor forward(const torch::Tensor& input) const {
        return torch::linear(input, weight_, has_bias_ ? bias_ : torch::Tensor());
    }
    
    // Load weights for a specific shard (e.g., "q", "k", "v" or 0, 1 for gate/up)
    template<typename ShardId>
    void load_shard_weight(const torch::Tensor& weight, ShardId shard_id) {
        int idx = get_shard_index(shard_id);
        int start = 0;
        for (int i = 0; i < idx; ++i) {
            start += out_features_list_[i] / tp_size_;
        }
        int size = out_features_list_[idx] / tp_size_;
        weight_.slice(0, start, start + size).copy_(weight);
    }
    
    torch::Tensor& weight() { return weight_; }
    const torch::Tensor& weight() const { return weight_; }
    
private:
    int get_shard_index(const std::string& shard_id) const {
        if (shard_id == "q") return 0;
        if (shard_id == "k") return 1;
        if (shard_id == "v") return 2;
        return 0;
    }
    
    int get_shard_index(int shard_id) const {
        return shard_id;
    }
    
    int in_features_;
    std::vector<int> out_features_list_;
    int total_out_features_;
    bool has_bias_;
    int tp_size_;
    DataType dtype_;
    torch::Tensor weight_;
    torch::Tensor bias_;
};

// QKV parallel linear (specialized for attention)
class QKVParallelLinear {
public:
    QKVParallelLinear(
        int hidden_size,
        int head_dim,
        int total_num_heads,
        int total_num_kv_heads,
        bool bias = true,
        int tp_size = 1,
        DataType dtype = DataType::kFloat16
    ) : hidden_size_(hidden_size),
        head_dim_(head_dim),
        total_num_heads_(total_num_heads),
        total_num_kv_heads_(total_num_kv_heads),
        has_bias_(bias),
        tp_size_(tp_size),
        dtype_(dtype) {
        
        num_heads_ = total_num_heads / tp_size;
        num_kv_heads_ = std::max(1, total_num_kv_heads / tp_size);
        
        q_size_ = num_heads_ * head_dim;
        kv_size_ = num_kv_heads_ * head_dim;
        total_out_features_ = q_size_ + 2 * kv_size_;
        
        initialize();
    }
    
    void initialize() {
        auto options = torch::TensorOptions()
            .dtype(to_torch_dtype(dtype_))
            .device(torch::kCUDA);
        
        weight_ = torch::empty({total_out_features_, hidden_size_}, options);
        if (has_bias_) {
            bias_ = torch::empty({total_out_features_}, options);
        }
    }
    
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(
        const torch::Tensor& input
    ) const {
        torch::Tensor qkv = torch::linear(input, weight_, has_bias_ ? bias_ : torch::Tensor());
        
        auto splits = qkv.split({q_size_, kv_size_, kv_size_}, -1);
        return std::make_tuple(splits[0], splits[1], splits[2]);
    }
    
    torch::Tensor& weight() { return weight_; }
    const torch::Tensor& weight() const { return weight_; }
    
    int q_size() const { return q_size_; }
    int kv_size() const { return kv_size_; }
    int num_heads() const { return num_heads_; }
    int num_kv_heads() const { return num_kv_heads_; }
    
private:
    int hidden_size_;
    int head_dim_;
    int total_num_heads_;
    int total_num_kv_heads_;
    int num_heads_;
    int num_kv_heads_;
    int q_size_;
    int kv_size_;
    int total_out_features_;
    bool has_bias_;
    int tp_size_;
    DataType dtype_;
    torch::Tensor weight_;
    torch::Tensor bias_;
};

}  // namespace vllm
