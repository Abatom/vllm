// SPDX-License-Identifier: Apache-2.0
// vLLM C++ Implementation - LLM Engine (V1)
#pragma once

#include "vllm/common.h"
#include "vllm/core/config.h"
#include "vllm/core/kv_cache.h"
#include "vllm/engine/request.h"
#include "vllm/engine/scheduler.h"
#include "vllm/model_executor/models/qwen2.h"
#include "vllm/model_executor/models/mimo_v2_flash.h"

#include <memory>
#include <functional>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

namespace vllm {

// Model type enum
enum class ModelType {
    QWEN2,
    MIMO_V2_FLASH,
    UNKNOWN
};

// Detect model type from config or name
inline ModelType detect_model_type(const std::string& model_name) {
    std::string lower_name = model_name;
    std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(), ::tolower);
    
    if (lower_name.find("qwen2") != std::string::npos ||
        lower_name.find("qwen-2") != std::string::npos) {
        return ModelType::QWEN2;
    }
    if (lower_name.find("mimo") != std::string::npos) {
        return ModelType::MIMO_V2_FLASH;
    }
    return ModelType::UNKNOWN;
}

// Abstract model interface
class ModelInterface {
public:
    virtual ~ModelInterface() = default;
    
    virtual torch::Tensor forward(
        const torch::Tensor& input_ids,
        const torch::Tensor& positions,
        KVCacheManager* kv_cache_manager,
        const AttentionMetadata* attn_metadata
    ) = 0;
    
    virtual torch::Tensor compute_logits(const torch::Tensor& hidden_states) = 0;
};

// Qwen2 model wrapper
class Qwen2ModelWrapper : public ModelInterface {
public:
    Qwen2ModelWrapper(
        const Qwen2Config& config,
        const CacheConfig* cache_config,
        int tp_size = 1
    ) : model_(config, cache_config, tp_size) {}
    
    torch::Tensor forward(
        const torch::Tensor& input_ids,
        const torch::Tensor& positions,
        KVCacheManager* kv_cache_manager,
        const AttentionMetadata* attn_metadata
    ) override {
        return model_.forward(input_ids, positions, kv_cache_manager, attn_metadata);
    }
    
    torch::Tensor compute_logits(const torch::Tensor& hidden_states) override {
        return model_.compute_logits(hidden_states);
    }
    
    Qwen2ForCausalLM& model() { return model_; }
    
private:
    Qwen2ForCausalLM model_;
};

// MiMoV2Flash model wrapper
class MiMoV2FlashModelWrapper : public ModelInterface {
public:
    MiMoV2FlashModelWrapper(
        const MiMoV2FlashConfig& config,
        const CacheConfig* cache_config,
        int tp_size = 1
    ) : model_(config, cache_config, tp_size) {}
    
    torch::Tensor forward(
        const torch::Tensor& input_ids,
        const torch::Tensor& positions,
        KVCacheManager* kv_cache_manager,
        const AttentionMetadata* attn_metadata
    ) override {
        return model_.forward(input_ids, positions, kv_cache_manager, attn_metadata);
    }
    
    torch::Tensor compute_logits(const torch::Tensor& hidden_states) override {
        return model_.compute_logits(hidden_states);
    }
    
    MiMoV2FlashForCausalLM& model() { return model_; }
    
private:
    MiMoV2FlashForCausalLM model_;
};

// Sampler for token generation
class Sampler {
public:
    Sampler() = default;
    
    // Sample next tokens from logits
    std::vector<int> sample(
        const torch::Tensor& logits,
        const std::vector<std::shared_ptr<Request>>& requests
    ) {
        std::vector<int> next_tokens;
        next_tokens.reserve(requests.size());
        
        for (size_t i = 0; i < requests.size(); ++i) {
            const auto& params = requests[i]->sampling_params();
            auto token_logits = logits[i];
            
            // Apply temperature
            if (params.temperature > 0 && params.temperature != 1.0f) {
                token_logits = token_logits / params.temperature;
            }
            
            // Apply top-k
            if (params.top_k > 0 && params.top_k < token_logits.size(0)) {
                auto [topk_values, topk_indices] = torch::topk(token_logits, params.top_k);
                auto mask = torch::full_like(token_logits, -std::numeric_limits<float>::infinity());
                mask.scatter_(0, topk_indices, topk_values);
                token_logits = mask;
            }
            
            // Apply top-p
            if (params.top_p < 1.0f) {
                auto sorted = torch::sort(token_logits, -1, true);
                auto sorted_logits = std::get<0>(sorted);
                auto sorted_indices = std::get<1>(sorted);
                auto probs = torch::softmax(sorted_logits, -1);
                auto cumsum = torch::cumsum(probs, -1);
                
                // Find cutoff
                auto mask = cumsum > params.top_p;
                mask[0] = false;  // Always keep at least one token
                sorted_logits.masked_fill_(mask, -std::numeric_limits<float>::infinity());
                
                // Unsort
                auto inv_indices = torch::argsort(sorted_indices);
                token_logits = sorted_logits.index_select(0, inv_indices);
            }
            
            // Sample or greedy
            int next_token;
            if (params.temperature == 0 || params.temperature < 1e-5f) {
                // Greedy
                next_token = token_logits.argmax().item<int>();
            } else {
                // Multinomial sampling
                auto probs = torch::softmax(token_logits, -1);
                next_token = torch::multinomial(probs, 1).item<int>();
            }
            
            next_tokens.push_back(next_token);
        }
        
        return next_tokens;
    }
};

// Main LLM Engine class (V1)
class LLMEngine {
public:
    LLMEngine(const VllmConfig& config)
        : config_(config), is_running_(false) {
        
        initialize();
    }
    
    ~LLMEngine() {
        stop();
    }
    
    void initialize() {
        // Set device
        c10::cuda::set_device(config_.device_id);
        
        // Initialize KV cache manager
        kv_cache_manager_ = std::make_unique<KVCacheManager>(
            config_.model_config,
            config_.cache_config,
            config_.device_id
        );
        
        // Initialize scheduler
        scheduler_ = std::make_unique<Scheduler>(
            config_.scheduler_config,
            config_.cache_config,
            kv_cache_manager_.get()
        );
        
        // Initialize model
        model_type_ = detect_model_type(config_.model_config.model_name);
        
        switch (model_type_) {
            case ModelType::QWEN2: {
                auto qwen_config = Qwen2Config::from_model_config(config_.model_config);
                model_ = std::make_unique<Qwen2ModelWrapper>(
                    qwen_config, &config_.cache_config, config_.parallel_config.tensor_parallel_size
                );
                break;
            }
            case ModelType::MIMO_V2_FLASH: {
                auto mimo_config = MiMoV2FlashConfig::from_model_config(config_.model_config);
                model_ = std::make_unique<MiMoV2FlashModelWrapper>(
                    mimo_config, &config_.cache_config, config_.parallel_config.tensor_parallel_size
                );
                break;
            }
            default:
                throw std::runtime_error("Unsupported model type: " + config_.model_config.model_name);
        }
        
        // Initialize sampler
        sampler_ = std::make_unique<Sampler>();
    }
    
    // Add a request for generation
    void add_request(
        const std::string& request_id,
        const std::vector<int>& prompt_token_ids,
        const SamplingParams& sampling_params,
        int eos_token_id = -1
    ) {
        auto request = std::make_shared<Request>(
            request_id, prompt_token_ids, sampling_params, eos_token_id
        );
        
        std::lock_guard<std::mutex> lock(mutex_);
        scheduler_->add_request(request);
    }
    
    // Abort a request
    void abort_request(const std::string& request_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        scheduler_->abort_request(request_id);
    }
    
    // Run one step of generation
    std::vector<RequestOutput> step() {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Schedule
        auto scheduler_output = scheduler_->schedule();
        
        if (scheduler_output.is_empty()) {
            return {};
        }
        
        // Prepare inputs
        auto [input_ids, positions] = prepare_inputs(scheduler_output);
        
        // Forward pass
        auto hidden_states = model_->forward(
            input_ids, positions, kv_cache_manager_.get(),
            &scheduler_output.attention_metadata
        );
        
        // Compute logits (only for last token of each sequence)
        torch::Tensor logits;
        if (scheduler_output.attention_metadata.is_prefill) {
            // Get last token hidden states for each sequence
            std::vector<torch::Tensor> last_hidden_states;
            int offset = 0;
            for (const auto& request : scheduler_output.scheduled_requests) {
                int num_tokens = (request->status() == RequestStatus::PREFILLING) ?
                                request->num_prompt_tokens() : 1;
                last_hidden_states.push_back(hidden_states[offset + num_tokens - 1].unsqueeze(0));
                offset += num_tokens;
            }
            hidden_states = torch::cat(last_hidden_states, 0);
        }
        
        logits = model_->compute_logits(hidden_states);
        
        // Sample next tokens
        auto next_tokens = sampler_->sample(logits, scheduler_output.scheduled_requests);
        
        // Create outputs
        std::vector<RequestOutput> outputs;
        for (size_t i = 0; i < scheduler_output.scheduled_requests.size(); ++i) {
            const auto& request = scheduler_output.scheduled_requests[i];
            
            RequestOutput output;
            output.request_id = request->request_id();
            output.new_token_ids = {next_tokens[i]};
            output.num_cached_tokens = request->num_cached_tokens();
            
            // Check if should finish
            request->append_token(next_tokens[i]);
            if (request->should_stop()) {
                output.finished = true;
                if (request->num_output_tokens() >= request->sampling_params().max_tokens) {
                    output.finish_reason = "length";
                } else {
                    output.finish_reason = "stop";
                }
            }
            
            outputs.push_back(output);
        }
        
        // Update scheduler
        scheduler_->update_after_step(outputs);
        
        return outputs;
    }
    
    // Run generation loop
    void run(std::function<void(const std::vector<RequestOutput>&)> callback) {
        is_running_ = true;
        
        while (is_running_ && !scheduler_->is_empty()) {
            auto outputs = step();
            if (!outputs.empty() && callback) {
                callback(outputs);
            }
        }
    }
    
    // Stop the engine
    void stop() {
        is_running_ = false;
    }
    
    // Check if engine has pending work
    bool has_pending_requests() const {
        return !scheduler_->is_empty();
    }
    
    // Get statistics
    int num_waiting_requests() const { return scheduler_->num_waiting(); }
    int num_running_requests() const { return scheduler_->num_running(); }
    int num_free_blocks() const { return kv_cache_manager_->num_free_blocks(); }
    
private:
    std::tuple<torch::Tensor, torch::Tensor> prepare_inputs(
        const SchedulerOutput& scheduler_output
    ) {
        std::vector<int> all_token_ids;
        std::vector<int> all_positions;
        
        for (const auto& request : scheduler_output.scheduled_requests) {
            if (request->status() == RequestStatus::PREFILLING) {
                // Prefill: include all prompt tokens
                for (int i = 0; i < request->num_prompt_tokens(); ++i) {
                    all_token_ids.push_back(request->prompt_token_ids()[i]);
                    all_positions.push_back(i);
                }
            } else {
                // Decode: only the last token
                int last_pos = request->num_tokens() - 1;
                int last_token = request->output_token_ids().back();
                all_token_ids.push_back(last_token);
                all_positions.push_back(last_pos);
            }
        }
        
        int num_tokens = static_cast<int>(all_token_ids.size());
        
        auto input_ids = torch::from_blob(
            all_token_ids.data(), {num_tokens}, torch::kInt32
        ).to(torch::kCUDA).clone();
        
        auto positions = torch::from_blob(
            all_positions.data(), {num_tokens}, torch::kInt32
        ).to(torch::kCUDA).clone();
        
        return std::make_tuple(input_ids, positions);
    }
    
    VllmConfig config_;
    ModelType model_type_;
    
    std::unique_ptr<ModelInterface> model_;
    std::unique_ptr<KVCacheManager> kv_cache_manager_;
    std::unique_ptr<Scheduler> scheduler_;
    std::unique_ptr<Sampler> sampler_;
    
    std::atomic<bool> is_running_;
    std::mutex mutex_;
};

}  // namespace vllm
