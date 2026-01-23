// SPDX-License-Identifier: Apache-2.0
// vLLM C++ Implementation - Scheduler (V1)
#pragma once

#include "vllm/common.h"
#include "vllm/core/config.h"
#include "vllm/core/kv_cache.h"
#include "vllm/engine/request.h"
#include "vllm/model_executor/layers/attention.h"

#include <queue>
#include <deque>
#include <unordered_map>
#include <memory>
#include <algorithm>

namespace vllm {

// Scheduler output
struct SchedulerOutput {
    // Requests scheduled for this step
    std::vector<std::shared_ptr<Request>> scheduled_requests;
    
    // Number of prefill and decode tokens
    int num_prefill_tokens = 0;
    int num_decode_tokens = 0;
    
    // Preempted requests
    std::vector<std::shared_ptr<Request>> preempted_requests;
    
    // Finished requests
    std::vector<std::shared_ptr<Request>> finished_requests;
    
    // Attention metadata for this batch
    AttentionMetadata attention_metadata;
    
    bool is_empty() const {
        return scheduled_requests.empty();
    }
    
    int num_scheduled_tokens() const {
        return num_prefill_tokens + num_decode_tokens;
    }
};

// Scheduler class
class Scheduler {
public:
    Scheduler(
        const SchedulerConfig& config,
        const CacheConfig& cache_config,
        KVCacheManager* kv_cache_manager
    ) : config_(config),
        cache_config_(cache_config),
        kv_cache_manager_(kv_cache_manager) {
        
        next_seq_id_ = 0;
    }
    
    // Add a new request
    void add_request(std::shared_ptr<Request> request) {
        // Assign sequence ID
        request->set_seq_id(next_seq_id_++);
        
        // Add to waiting queue
        waiting_queue_.push_back(request);
    }
    
    // Abort a request
    void abort_request(const std::string& request_id) {
        // Check waiting queue
        for (auto it = waiting_queue_.begin(); it != waiting_queue_.end(); ++it) {
            if ((*it)->request_id() == request_id) {
                (*it)->set_status(RequestStatus::FINISHED_ABORT);
                waiting_queue_.erase(it);
                return;
            }
        }
        
        // Check running requests
        auto it = running_requests_.find(request_id);
        if (it != running_requests_.end()) {
            it->second->set_status(RequestStatus::FINISHED_ABORT);
            // Free KV cache blocks
            kv_cache_manager_->free_blocks(it->second->seq_id());
            running_requests_.erase(it);
        }
    }
    
    // Schedule next batch
    SchedulerOutput schedule() {
        SchedulerOutput output;
        
        int budget_tokens = config_.max_num_batched_tokens;
        int budget_seqs = config_.max_num_seqs;
        
        // First, handle running requests (decode tokens)
        for (auto& [req_id, request] : running_requests_) {
            if (output.scheduled_requests.size() >= static_cast<size_t>(budget_seqs)) {
                break;
            }
            
            // Each decode request uses 1 token
            if (budget_tokens >= 1) {
                output.scheduled_requests.push_back(request);
                output.num_decode_tokens += 1;
                budget_tokens -= 1;
                request->mark_scheduled();
            }
        }
        
        // Then, try to admit new requests from waiting queue
        while (!waiting_queue_.empty() && 
               output.scheduled_requests.size() < static_cast<size_t>(budget_seqs)) {
            
            auto request = waiting_queue_.front();
            int num_tokens = request->num_prompt_tokens() - request->num_cached_tokens();
            
            // Check if we have budget
            if (num_tokens > budget_tokens) {
                // Check if chunked prefill is enabled
                if (config_.enable_chunked_prefill && budget_tokens > 0) {
                    // Schedule partial prefill
                    num_tokens = budget_tokens;
                } else {
                    break;
                }
            }
            
            // Try to allocate KV cache blocks
            int total_tokens = request->num_tokens();
            if (!kv_cache_manager_->allocate_blocks(request->seq_id(), total_tokens)) {
                // Not enough blocks, try preemption or wait
                break;
            }
            
            waiting_queue_.pop_front();
            output.scheduled_requests.push_back(request);
            output.num_prefill_tokens += num_tokens;
            budget_tokens -= num_tokens;
            
            request->set_status(RequestStatus::PREFILLING);
            request->mark_scheduled();
            running_requests_[request->request_id()] = request;
        }
        
        // Build attention metadata
        if (!output.is_empty()) {
            build_attention_metadata(output);
        }
        
        return output;
    }
    
    // Update after a step
    void update_after_step(const std::vector<RequestOutput>& outputs) {
        for (const auto& req_output : outputs) {
            auto it = running_requests_.find(req_output.request_id);
            if (it == running_requests_.end()) continue;
            
            auto& request = it->second;
            
            // Append new tokens
            for (int token_id : req_output.new_token_ids) {
                request->append_token(token_id);
            }
            
            // Check if finished
            if (req_output.finished || request->should_stop()) {
                if (request->num_output_tokens() >= request->sampling_params().max_tokens) {
                    request->set_status(RequestStatus::FINISHED_LENGTH);
                } else {
                    request->set_status(RequestStatus::FINISHED_STOPPED);
                }
                
                // Free KV cache
                kv_cache_manager_->free_blocks(request->seq_id());
                running_requests_.erase(it);
            } else {
                // Transition to decoding
                request->set_status(RequestStatus::DECODING);
            }
        }
    }
    
    // Get number of waiting requests
    int num_waiting() const {
        return static_cast<int>(waiting_queue_.size());
    }
    
    // Get number of running requests
    int num_running() const {
        return static_cast<int>(running_requests_.size());
    }
    
    // Get all running requests
    const std::unordered_map<std::string, std::shared_ptr<Request>>& running_requests() const {
        return running_requests_;
    }
    
    // Check if empty
    bool is_empty() const {
        return waiting_queue_.empty() && running_requests_.empty();
    }
    
private:
    void build_attention_metadata(SchedulerOutput& output) {
        int batch_size = static_cast<int>(output.scheduled_requests.size());
        
        auto int_options = torch::TensorOptions()
            .dtype(torch::kInt32)
            .device(torch::kCUDA);
        
        // Compute sequence lengths and build block tables
        std::vector<int> seq_lens;
        std::vector<int> context_lens;
        int max_seq_len = 0;
        int max_num_blocks = 0;
        
        for (const auto& request : output.scheduled_requests) {
            int seq_len = request->num_tokens();
            seq_lens.push_back(seq_len);
            context_lens.push_back(request->num_prompt_tokens());
            max_seq_len = std::max(max_seq_len, seq_len);
            
            const auto* block_table = kv_cache_manager_->get_block_table(request->seq_id());
            if (block_table) {
                max_num_blocks = std::max(max_num_blocks, block_table->num_blocks());
            }
        }
        
        // Create tensors
        output.attention_metadata.seq_lens = torch::from_blob(
            seq_lens.data(), {batch_size}, torch::kInt32
        ).to(torch::kCUDA).clone();
        
        output.attention_metadata.context_lens = torch::from_blob(
            context_lens.data(), {batch_size}, torch::kInt32
        ).to(torch::kCUDA).clone();
        
        output.attention_metadata.max_seq_len = max_seq_len;
        output.attention_metadata.max_num_blocks = max_num_blocks;
        output.attention_metadata.is_prefill = (output.num_prefill_tokens > 0);
        output.attention_metadata.num_prefill_tokens = output.num_prefill_tokens;
        output.attention_metadata.num_decode_tokens = output.num_decode_tokens;
        
        // Build block tables tensor
        if (max_num_blocks > 0) {
            output.attention_metadata.block_tables = torch::zeros(
                {batch_size, max_num_blocks}, int_options
            );
            
            for (int i = 0; i < batch_size; ++i) {
                const auto* block_table = kv_cache_manager_->get_block_table(
                    output.scheduled_requests[i]->seq_id()
                );
                if (block_table) {
                    auto table_tensor = block_table->to_tensor();
                    output.attention_metadata.block_tables[i].narrow(
                        0, 0, block_table->num_blocks()
                    ).copy_(table_tensor);
                }
            }
        }
        
        // Build slot mapping
        int total_tokens = output.num_scheduled_tokens();
        std::vector<int> slot_mapping(total_tokens);
        int slot_idx = 0;
        
        for (const auto& request : output.scheduled_requests) {
            const auto* block_table = kv_cache_manager_->get_block_table(request->seq_id());
            int num_tokens = (output.attention_metadata.is_prefill && 
                             request->status() == RequestStatus::PREFILLING) ?
                            request->num_prompt_tokens() : 1;
            
            for (int i = 0; i < num_tokens; ++i) {
                int token_pos = request->num_tokens() - num_tokens + i;
                if (block_table) {
                    int block_idx = token_pos / cache_config_.block_size;
                    int block_offset = token_pos % cache_config_.block_size;
                    slot_mapping[slot_idx++] = 
                        block_table->get_block(block_idx) * cache_config_.block_size + block_offset;
                } else {
                    slot_mapping[slot_idx++] = -1;
                }
            }
        }
        
        output.attention_metadata.slot_mapping = torch::from_blob(
            slot_mapping.data(), {total_tokens}, torch::kInt32
        ).to(torch::kCUDA).clone();
    }
    
    SchedulerConfig config_;
    CacheConfig cache_config_;
    KVCacheManager* kv_cache_manager_;
    
    std::deque<std::shared_ptr<Request>> waiting_queue_;
    std::unordered_map<std::string, std::shared_ptr<Request>> running_requests_;
    int next_seq_id_;
};

}  // namespace vllm
