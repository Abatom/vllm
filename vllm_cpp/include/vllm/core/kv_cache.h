// SPDX-License-Identifier: Apache-2.0
// vLLM C++ Implementation - KV Cache Management
#pragma once

#include "vllm/common.h"
#include "vllm/core/config.h"
#include "vllm/core/tensor.h"

#include <deque>
#include <set>

namespace vllm {

// Block of KV cache
struct KVBlock {
    int block_id;
    int ref_count = 0;
    bool is_allocated = false;
    
    void acquire() { ++ref_count; }
    void release() { --ref_count; }
    bool is_free() const { return ref_count == 0 && !is_allocated; }
};

// Block table for a single sequence
class BlockTable {
public:
    BlockTable() = default;
    
    void add_block(int block_id) {
        block_ids_.push_back(block_id);
    }
    
    void pop_block() {
        if (!block_ids_.empty()) {
            block_ids_.pop_back();
        }
    }
    
    int get_block(int idx) const {
        return block_ids_[idx];
    }
    
    int num_blocks() const {
        return static_cast<int>(block_ids_.size());
    }
    
    const std::vector<int>& blocks() const {
        return block_ids_;
    }
    
    void clear() {
        block_ids_.clear();
    }
    
    // Convert to tensor for kernel
    torch::Tensor to_tensor(int device_id = 0) const {
        auto options = torch::TensorOptions()
            .dtype(torch::kInt32)
            .device(torch::kCUDA, device_id);
        auto tensor = torch::empty({static_cast<int64_t>(block_ids_.size())}, options);
        
        // Copy block IDs to GPU
        std::vector<int32_t> ids(block_ids_.begin(), block_ids_.end());
        cudaMemcpy(tensor.data_ptr<int32_t>(), ids.data(),
                   ids.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
        return tensor;
    }
    
private:
    std::vector<int> block_ids_;
};

// Block pool allocator
class BlockPool {
public:
    BlockPool(int num_blocks) : num_blocks_(num_blocks) {
        for (int i = 0; i < num_blocks; ++i) {
            free_blocks_.insert(i);
        }
    }
    
    // Allocate a block
    int allocate() {
        if (free_blocks_.empty()) {
            return -1;  // No free blocks
        }
        int block_id = *free_blocks_.begin();
        free_blocks_.erase(free_blocks_.begin());
        return block_id;
    }
    
    // Free a block
    void free(int block_id) {
        free_blocks_.insert(block_id);
    }
    
    // Get number of free blocks
    int num_free_blocks() const {
        return static_cast<int>(free_blocks_.size());
    }
    
    // Get total number of blocks
    int total_blocks() const {
        return num_blocks_;
    }
    
    // Check if a block can be allocated
    bool can_allocate(int num_blocks = 1) const {
        return free_blocks_.size() >= static_cast<size_t>(num_blocks);
    }
    
private:
    int num_blocks_;
    std::set<int> free_blocks_;
};

// KV Cache for a single layer
class LayerKVCache {
public:
    LayerKVCache(
        int num_blocks,
        int block_size,
        int num_kv_heads,
        int head_dim,
        DataType dtype,
        int device_id = 0
    ) : num_blocks_(num_blocks),
        block_size_(block_size),
        num_kv_heads_(num_kv_heads),
        head_dim_(head_dim),
        dtype_(dtype),
        device_id_(device_id) {
        allocate();
    }
    
    void allocate() {
        auto options = torch::TensorOptions()
            .dtype(to_torch_dtype(dtype_))
            .device(torch::kCUDA, device_id_);
        
        // Shape: [num_blocks, block_size, num_kv_heads, head_dim]
        key_cache_ = torch::zeros({num_blocks_, block_size_, num_kv_heads_, head_dim_}, options);
        value_cache_ = torch::zeros({num_blocks_, block_size_, num_kv_heads_, head_dim_}, options);
    }
    
    torch::Tensor& key_cache() { return key_cache_; }
    torch::Tensor& value_cache() { return value_cache_; }
    const torch::Tensor& key_cache() const { return key_cache_; }
    const torch::Tensor& value_cache() const { return value_cache_; }
    
private:
    int num_blocks_;
    int block_size_;
    int num_kv_heads_;
    int head_dim_;
    DataType dtype_;
    int device_id_;
    torch::Tensor key_cache_;
    torch::Tensor value_cache_;
};

// KV Cache Manager - manages all layer caches
class KVCacheManager {
public:
    KVCacheManager(const ModelConfig& model_config, const CacheConfig& cache_config, int device_id = 0)
        : model_config_(model_config),
          cache_config_(cache_config),
          device_id_(device_id) {
        initialize();
    }
    
    void initialize() {
        // Compute number of blocks based on GPU memory if not specified
        int num_blocks = cache_config_.num_gpu_blocks;
        if (num_blocks == 0) {
            num_blocks = compute_num_blocks();
        }
        
        // Create block pool
        block_pool_ = std::make_unique<BlockPool>(num_blocks);
        
        // Create per-layer caches
        DataType cache_dtype = DataType::kFloat16;
        if (cache_config_.cache_dtype == "fp8") {
            cache_dtype = DataType::kFP8E4M3;
        }
        
        for (int i = 0; i < model_config_.num_hidden_layers; ++i) {
            layer_caches_.push_back(std::make_unique<LayerKVCache>(
                num_blocks,
                cache_config_.block_size,
                model_config_.num_key_value_heads,
                model_config_.head_dim,
                cache_dtype,
                device_id_
            ));
        }
        
        num_blocks_ = num_blocks;
    }
    
    int compute_num_blocks() {
        // Get available GPU memory
        size_t free_mem, total_mem;
        VLLM_CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
        
        // Use configured utilization
        size_t usable_mem = static_cast<size_t>(total_mem * cache_config_.gpu_memory_utilization);
        
        // Estimate memory for model weights (rough estimate)
        size_t model_mem = model_config_.num_hidden_layers * 
                           model_config_.hidden_size * 
                           model_config_.intermediate_size * 2 * 
                           dtype_size(model_config_.dtype);
        
        // Available for KV cache
        size_t cache_mem = (usable_mem > model_mem) ? (usable_mem - model_mem) : (usable_mem / 2);
        
        // Compute block size in bytes
        size_t block_bytes = cache_config_.get_block_size_bytes(model_config_);
        
        return static_cast<int>(cache_mem / block_bytes);
    }
    
    // Allocate blocks for a sequence
    bool allocate_blocks(int seq_id, int num_tokens) {
        int num_blocks_needed = (num_tokens + cache_config_.block_size - 1) / cache_config_.block_size;
        
        if (!block_pool_->can_allocate(num_blocks_needed)) {
            return false;
        }
        
        BlockTable table;
        for (int i = 0; i < num_blocks_needed; ++i) {
            int block_id = block_pool_->allocate();
            if (block_id < 0) {
                // Rollback
                for (int j = 0; j < table.num_blocks(); ++j) {
                    block_pool_->free(table.get_block(j));
                }
                return false;
            }
            table.add_block(block_id);
        }
        
        block_tables_[seq_id] = std::move(table);
        return true;
    }
    
    // Free blocks for a sequence
    void free_blocks(int seq_id) {
        auto it = block_tables_.find(seq_id);
        if (it != block_tables_.end()) {
            for (int block_id : it->second.blocks()) {
                block_pool_->free(block_id);
            }
            block_tables_.erase(it);
        }
    }
    
    // Get block table for a sequence
    const BlockTable* get_block_table(int seq_id) const {
        auto it = block_tables_.find(seq_id);
        if (it != block_tables_.end()) {
            return &it->second;
        }
        return nullptr;
    }
    
    // Access layer caches
    LayerKVCache* get_layer_cache(int layer_idx) {
        return layer_caches_[layer_idx].get();
    }
    
    // Get number of free blocks
    int num_free_blocks() const {
        return block_pool_->num_free_blocks();
    }
    
    int num_total_blocks() const {
        return num_blocks_;
    }
    
private:
    ModelConfig model_config_;
    CacheConfig cache_config_;
    int device_id_;
    int num_blocks_ = 0;
    std::unique_ptr<BlockPool> block_pool_;
    std::vector<std::unique_ptr<LayerKVCache>> layer_caches_;
    std::unordered_map<int, BlockTable> block_tables_;
};

}  // namespace vllm
