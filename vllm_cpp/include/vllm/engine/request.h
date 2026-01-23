// SPDX-License-Identifier: Apache-2.0
// vLLM C++ Implementation - Request Definition
#pragma once

#include "vllm/common.h"

#include <chrono>
#include <string>
#include <vector>
#include <optional>

namespace vllm {

// Sampling parameters for generation
struct SamplingParams {
    float temperature = 1.0f;
    float top_p = 1.0f;
    int top_k = -1;
    float min_p = 0.0f;
    float repetition_penalty = 1.0f;
    float frequency_penalty = 0.0f;
    float presence_penalty = 0.0f;
    int max_tokens = 16;
    int min_tokens = 0;
    std::vector<std::string> stop_strings;
    std::vector<int> stop_token_ids;
    bool ignore_eos = false;
    int n = 1;  // Number of completions
    int best_of = 1;
    bool use_beam_search = false;
    int seed = -1;
    bool logprobs = false;
    int top_logprobs = 0;
    bool skip_special_tokens = true;
    
    // Check if should stop on EOS
    bool should_stop_on_eos() const {
        return !ignore_eos;
    }
};

// Request status
enum class RequestStatus {
    PENDING,
    RUNNING,
    PREFILLING,
    DECODING,
    FINISHED_STOPPED,
    FINISHED_LENGTH,
    FINISHED_ABORT,
    FINISHED_ERROR
};

inline std::string status_to_string(RequestStatus status) {
    switch (status) {
        case RequestStatus::PENDING: return "pending";
        case RequestStatus::RUNNING: return "running";
        case RequestStatus::PREFILLING: return "prefilling";
        case RequestStatus::DECODING: return "decoding";
        case RequestStatus::FINISHED_STOPPED: return "finished_stopped";
        case RequestStatus::FINISHED_LENGTH: return "finished_length";
        case RequestStatus::FINISHED_ABORT: return "finished_abort";
        case RequestStatus::FINISHED_ERROR: return "finished_error";
        default: return "unknown";
    }
}

inline bool is_finished(RequestStatus status) {
    return status == RequestStatus::FINISHED_STOPPED ||
           status == RequestStatus::FINISHED_LENGTH ||
           status == RequestStatus::FINISHED_ABORT ||
           status == RequestStatus::FINISHED_ERROR;
}

// Request class
class Request {
public:
    Request(
        const std::string& request_id,
        const std::vector<int>& prompt_token_ids,
        const SamplingParams& sampling_params,
        int eos_token_id = -1
    ) : request_id_(request_id),
        prompt_token_ids_(prompt_token_ids),
        sampling_params_(sampling_params),
        eos_token_id_(eos_token_id),
        status_(RequestStatus::PENDING) {
        
        arrival_time_ = std::chrono::steady_clock::now();
        num_prompt_tokens_ = static_cast<int>(prompt_token_ids.size());
    }
    
    // Accessors
    const std::string& request_id() const { return request_id_; }
    const std::vector<int>& prompt_token_ids() const { return prompt_token_ids_; }
    const std::vector<int>& output_token_ids() const { return output_token_ids_; }
    const SamplingParams& sampling_params() const { return sampling_params_; }
    RequestStatus status() const { return status_; }
    int eos_token_id() const { return eos_token_id_; }
    
    int num_prompt_tokens() const { return num_prompt_tokens_; }
    int num_output_tokens() const { return static_cast<int>(output_token_ids_.size()); }
    int num_tokens() const { return num_prompt_tokens_ + num_output_tokens(); }
    
    // State management
    void set_status(RequestStatus status) { status_ = status; }
    
    void append_token(int token_id) {
        output_token_ids_.push_back(token_id);
    }
    
    bool is_finished() const {
        return vllm::is_finished(status_);
    }
    
    // Check if should stop
    bool should_stop() const {
        if (is_finished()) return true;
        
        // Check max tokens
        if (num_output_tokens() >= sampling_params_.max_tokens) {
            return true;
        }
        
        // Check EOS token
        if (!output_token_ids_.empty() && 
            output_token_ids_.back() == eos_token_id_ &&
            sampling_params_.should_stop_on_eos()) {
            return true;
        }
        
        // Check stop token IDs
        if (!output_token_ids_.empty()) {
            int last_token = output_token_ids_.back();
            for (int stop_id : sampling_params_.stop_token_ids) {
                if (last_token == stop_id) {
                    return true;
                }
            }
        }
        
        return false;
    }
    
    // Timing
    std::chrono::steady_clock::time_point arrival_time() const { return arrival_time_; }
    
    double time_in_queue_ms() const {
        if (first_scheduled_time_.time_since_epoch().count() == 0) {
            auto now = std::chrono::steady_clock::now();
            return std::chrono::duration<double, std::milli>(now - arrival_time_).count();
        }
        return std::chrono::duration<double, std::milli>(first_scheduled_time_ - arrival_time_).count();
    }
    
    void mark_scheduled() {
        if (first_scheduled_time_.time_since_epoch().count() == 0) {
            first_scheduled_time_ = std::chrono::steady_clock::now();
        }
    }
    
    // Sequence ID for internal tracking
    int seq_id() const { return seq_id_; }
    void set_seq_id(int seq_id) { seq_id_ = seq_id; }
    
    // Number of cached tokens (for prefix caching)
    int num_cached_tokens() const { return num_cached_tokens_; }
    void set_num_cached_tokens(int n) { num_cached_tokens_ = n; }
    
private:
    std::string request_id_;
    std::vector<int> prompt_token_ids_;
    std::vector<int> output_token_ids_;
    SamplingParams sampling_params_;
    int eos_token_id_;
    RequestStatus status_;
    int num_prompt_tokens_;
    int seq_id_ = -1;
    int num_cached_tokens_ = 0;
    
    std::chrono::steady_clock::time_point arrival_time_;
    std::chrono::steady_clock::time_point first_scheduled_time_;
};

// Request output
struct RequestOutput {
    std::string request_id;
    std::vector<int> new_token_ids;
    bool finished = false;
    std::string finish_reason;
    std::string stop_reason;
    int num_cached_tokens = 0;
    
    // Logprobs (optional)
    std::vector<std::vector<std::pair<int, float>>> logprobs;
};

}  // namespace vllm
