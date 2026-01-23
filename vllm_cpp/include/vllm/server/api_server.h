// SPDX-License-Identifier: Apache-2.0
// vLLM C++ - OpenAI-Compatible API Server
#pragma once

#include "vllm/common.h"
#include "vllm/engine/engine.h"
#include "vllm/server/openai_types.h"

#define CPPHTTPLIB_OPENSSL_SUPPORT 0
#include "httplib.h"

#include <memory>
#include <mutex>
#include <thread>
#include <atomic>
#include <queue>
#include <condition_variable>

namespace vllm {

// Simple JSON parser for requests
class SimpleJsonParser {
public:
    static std::string get_string(const std::string& json, const std::string& key, const std::string& default_val = "") {
        std::string search = "\"" + key + "\"";
        size_t pos = json.find(search);
        if (pos == std::string::npos) return default_val;
        
        pos = json.find(':', pos);
        if (pos == std::string::npos) return default_val;
        
        // Skip whitespace
        pos++;
        while (pos < json.size() && std::isspace(json[pos])) pos++;
        
        if (pos >= json.size()) return default_val;
        
        if (json[pos] == '"') {
            // String value
            pos++;
            size_t end = pos;
            while (end < json.size() && json[end] != '"') {
                if (json[end] == '\\' && end + 1 < json.size()) {
                    end += 2;
                } else {
                    end++;
                }
            }
            return json.substr(pos, end - pos);
        }
        
        return default_val;
    }
    
    static int get_int(const std::string& json, const std::string& key, int default_val = 0) {
        std::string search = "\"" + key + "\"";
        size_t pos = json.find(search);
        if (pos == std::string::npos) return default_val;
        
        pos = json.find(':', pos);
        if (pos == std::string::npos) return default_val;
        
        pos++;
        while (pos < json.size() && std::isspace(json[pos])) pos++;
        
        try {
            return std::stoi(json.substr(pos));
        } catch (...) {
            return default_val;
        }
    }
    
    static float get_float(const std::string& json, const std::string& key, float default_val = 0.0f) {
        std::string search = "\"" + key + "\"";
        size_t pos = json.find(search);
        if (pos == std::string::npos) return default_val;
        
        pos = json.find(':', pos);
        if (pos == std::string::npos) return default_val;
        
        pos++;
        while (pos < json.size() && std::isspace(json[pos])) pos++;
        
        try {
            return std::stof(json.substr(pos));
        } catch (...) {
            return default_val;
        }
    }
    
    static bool get_bool(const std::string& json, const std::string& key, bool default_val = false) {
        std::string search = "\"" + key + "\"";
        size_t pos = json.find(search);
        if (pos == std::string::npos) return default_val;
        
        pos = json.find(':', pos);
        if (pos == std::string::npos) return default_val;
        
        pos++;
        while (pos < json.size() && std::isspace(json[pos])) pos++;
        
        if (json.substr(pos, 4) == "true") return true;
        if (json.substr(pos, 5) == "false") return false;
        return default_val;
    }
    
    // Extract messages array from chat completion request
    static std::vector<openai::ChatMessage> get_messages(const std::string& json) {
        std::vector<openai::ChatMessage> messages;
        
        size_t pos = json.find("\"messages\"");
        if (pos == std::string::npos) return messages;
        
        pos = json.find('[', pos);
        if (pos == std::string::npos) return messages;
        
        // Find matching ]
        int depth = 1;
        size_t start = pos + 1;
        pos++;
        
        while (pos < json.size() && depth > 0) {
            if (json[pos] == '[') depth++;
            else if (json[pos] == ']') depth--;
            else if (json[pos] == '{' && depth == 1) {
                // Found a message object
                int obj_depth = 1;
                size_t obj_start = pos;
                pos++;
                while (pos < json.size() && obj_depth > 0) {
                    if (json[pos] == '{') obj_depth++;
                    else if (json[pos] == '}') obj_depth--;
                    pos++;
                }
                
                std::string msg_json = json.substr(obj_start, pos - obj_start);
                openai::ChatMessage msg;
                msg.role = get_string(msg_json, "role");
                msg.content = get_string(msg_json, "content");
                messages.push_back(msg);
                continue;
            }
            pos++;
        }
        
        return messages;
    }
};

// API Server configuration
struct ServerConfig {
    std::string host = "0.0.0.0";
    int port = 8000;
    std::string model_path;
    std::string served_model_name;
    int max_concurrent_requests = 256;
    bool trust_remote_code = true;
    
    // Model config overrides
    int max_model_len = 0;  // 0 = use model default
    float gpu_memory_utilization = 0.9f;
    int tensor_parallel_size = 1;
};

// OpenAI-compatible API Server
class APIServer {
public:
    APIServer(const ServerConfig& config)
        : config_(config), running_(false) {
        
        initialize();
    }
    
    ~APIServer() {
        stop();
    }
    
    void initialize() {
        // Create vLLM configuration
        VllmConfig vllm_config = VllmConfig::create_default(config_.model_path);
        vllm_config.model_config.model_name = detect_model_name(config_.model_path);
        vllm_config.cache_config.gpu_memory_utilization = config_.gpu_memory_utilization;
        vllm_config.parallel_config.tensor_parallel_size = config_.tensor_parallel_size;
        vllm_config.scheduler_config.max_num_seqs = config_.max_concurrent_requests;
        
        if (config_.max_model_len > 0) {
            vllm_config.scheduler_config.max_model_len = config_.max_model_len;
        }
        
        // Set served model name
        if (config_.served_model_name.empty()) {
            served_model_name_ = vllm_config.model_config.model_name;
        } else {
            served_model_name_ = config_.served_model_name;
        }
        
        // Create engine
        engine_ = std::make_unique<LLMEngine>(vllm_config);
        
        // Setup HTTP server
        setup_routes();
    }
    
    void setup_routes() {
        // Health check
        server_.Get("/health", [](const httplib::Request&, httplib::Response& res) {
            res.set_content("{\"status\":\"ok\"}", "application/json");
        });
        
        // Version
        server_.Get("/version", [](const httplib::Request&, httplib::Response& res) {
            res.set_content("{\"version\":\"vllm-cpp-1.0.0\"}", "application/json");
        });
        
        // List models
        server_.Get("/v1/models", [this](const httplib::Request&, httplib::Response& res) {
            handle_list_models(res);
        });
        
        // Chat completions
        server_.Post("/v1/chat/completions", [this](const httplib::Request& req, httplib::Response& res) {
            handle_chat_completions(req, res);
        });
        
        // Completions (legacy)
        server_.Post("/v1/completions", [this](const httplib::Request& req, httplib::Response& res) {
            handle_completions(req, res);
        });
        
        // CORS support
        server_.Options(".*", [](const httplib::Request&, httplib::Response& res) {
            res.set_header("Access-Control-Allow-Origin", "*");
            res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
            res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");
            res.status = 204;
        });
        
        // Set default headers
        server_.set_default_headers({
            {"Access-Control-Allow-Origin", "*"},
            {"Content-Type", "application/json"}
        });
    }
    
    void handle_list_models(httplib::Response& res) {
        openai::ModelsResponse response;
        openai::ModelInfo model;
        model.id = served_model_name_;
        model.created = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()
        ).count();
        response.data.push_back(model);
        
        res.set_content(openai::to_json(response), "application/json");
    }
    
    void handle_chat_completions(const httplib::Request& req, httplib::Response& res) {
        try {
            const std::string& body = req.body;
            
            // Parse request
            std::string model = SimpleJsonParser::get_string(body, "model", served_model_name_);
            auto messages = SimpleJsonParser::get_messages(body);
            float temperature = SimpleJsonParser::get_float(body, "temperature", 1.0f);
            float top_p = SimpleJsonParser::get_float(body, "top_p", 1.0f);
            int max_tokens = SimpleJsonParser::get_int(body, "max_tokens", 256);
            bool stream = SimpleJsonParser::get_bool(body, "stream", false);
            
            if (messages.empty()) {
                send_error(res, "messages array is required", "invalid_request_error", 400);
                return;
            }
            
            // Build prompt from messages
            std::string prompt = build_prompt_from_messages(messages);
            
            // Create sampling parameters
            SamplingParams sampling_params;
            sampling_params.temperature = temperature;
            sampling_params.top_p = top_p;
            sampling_params.max_tokens = max_tokens;
            sampling_params.top_k = SimpleJsonParser::get_int(body, "top_k", -1);
            sampling_params.min_p = SimpleJsonParser::get_float(body, "min_p", 0.0f);
            sampling_params.repetition_penalty = SimpleJsonParser::get_float(body, "repetition_penalty", 1.0f);
            
            // Generate request ID
            std::string request_id = openai::generate_request_id();
            
            if (stream) {
                handle_chat_completions_stream(res, request_id, model, prompt, sampling_params);
            } else {
                handle_chat_completions_sync(res, request_id, model, prompt, sampling_params);
            }
            
        } catch (const std::exception& e) {
            send_error(res, e.what(), "internal_error", 500);
        }
    }
    
    void handle_chat_completions_sync(
        httplib::Response& res,
        const std::string& request_id,
        const std::string& model,
        const std::string& prompt,
        const SamplingParams& sampling_params
    ) {
        // TODO: Integrate with tokenizer to convert prompt to token IDs
        // For now, use placeholder tokens
        std::vector<int> prompt_tokens = tokenize(prompt);
        
        // Add request to engine
        std::string internal_id = request_id;
        engine_->add_request(internal_id, prompt_tokens, sampling_params, /*eos_token_id=*/151643);
        
        // Collect output
        std::string generated_text;
        std::string finish_reason;
        int completion_tokens = 0;
        
        while (engine_->has_pending_requests()) {
            auto outputs = engine_->step();
            for (const auto& output : outputs) {
                if (output.request_id == internal_id) {
                    // Detokenize new tokens
                    for (int token_id : output.new_token_ids) {
                        generated_text += detokenize(token_id);
                        completion_tokens++;
                    }
                    if (output.finished) {
                        finish_reason = output.finish_reason;
                    }
                }
            }
        }
        
        // Build response
        openai::ChatCompletionResponse response;
        response.id = request_id;
        response.model = model;
        
        openai::ChatCompletionChoice choice;
        choice.index = 0;
        choice.message.role = "assistant";
        choice.message.content = generated_text;
        choice.finish_reason = finish_reason;
        response.choices.push_back(choice);
        
        response.usage.prompt_tokens = static_cast<int>(prompt_tokens.size());
        response.usage.completion_tokens = completion_tokens;
        response.usage.total_tokens = response.usage.prompt_tokens + completion_tokens;
        
        res.set_content(openai::to_json(response), "application/json");
    }
    
    void handle_chat_completions_stream(
        httplib::Response& res,
        const std::string& request_id,
        const std::string& model,
        const std::string& prompt,
        const SamplingParams& sampling_params
    ) {
        // Set streaming headers
        res.set_header("Content-Type", "text/event-stream");
        res.set_header("Cache-Control", "no-cache");
        res.set_header("Connection", "keep-alive");
        
        std::vector<int> prompt_tokens = tokenize(prompt);
        
        std::string internal_id = request_id;
        engine_->add_request(internal_id, prompt_tokens, sampling_params, /*eos_token_id=*/151643);
        
        // Send initial chunk with role
        {
            openai::ChatCompletionChunk chunk;
            chunk.id = request_id;
            chunk.model = model;
            
            openai::ChatCompletionStreamChoice choice;
            choice.index = 0;
            choice.delta.role = "assistant";
            chunk.choices.push_back(choice);
            
            std::string data = "data: " + openai::to_json(chunk) + "\n\n";
            res.set_content(data, "text/event-stream");
        }
        
        // Note: True streaming requires chunked transfer encoding
        // which needs more complex handling. This is a simplified version.
        
        std::string full_content;
        while (engine_->has_pending_requests()) {
            auto outputs = engine_->step();
            for (const auto& output : outputs) {
                if (output.request_id == internal_id) {
                    for (int token_id : output.new_token_ids) {
                        std::string token_text = detokenize(token_id);
                        full_content += token_text;
                    }
                }
            }
        }
        
        // Send content chunk
        {
            openai::ChatCompletionChunk chunk;
            chunk.id = request_id;
            chunk.model = model;
            
            openai::ChatCompletionStreamChoice choice;
            choice.index = 0;
            choice.delta.content = full_content;
            chunk.choices.push_back(choice);
            
            std::string data = "data: " + openai::to_json(chunk) + "\n\n";
            data += "data: [DONE]\n\n";
            res.set_content(data, "text/event-stream");
        }
    }
    
    void handle_completions(const httplib::Request& req, httplib::Response& res) {
        try {
            const std::string& body = req.body;
            
            std::string model = SimpleJsonParser::get_string(body, "model", served_model_name_);
            std::string prompt = SimpleJsonParser::get_string(body, "prompt");
            float temperature = SimpleJsonParser::get_float(body, "temperature", 1.0f);
            float top_p = SimpleJsonParser::get_float(body, "top_p", 1.0f);
            int max_tokens = SimpleJsonParser::get_int(body, "max_tokens", 16);
            
            SamplingParams sampling_params;
            sampling_params.temperature = temperature;
            sampling_params.top_p = top_p;
            sampling_params.max_tokens = max_tokens;
            
            std::string request_id = openai::generate_request_id();
            std::vector<int> prompt_tokens = tokenize(prompt);
            
            engine_->add_request(request_id, prompt_tokens, sampling_params, /*eos_token_id=*/151643);
            
            std::string generated_text;
            std::string finish_reason;
            int completion_tokens = 0;
            
            while (engine_->has_pending_requests()) {
                auto outputs = engine_->step();
                for (const auto& output : outputs) {
                    if (output.request_id == request_id) {
                        for (int token_id : output.new_token_ids) {
                            generated_text += detokenize(token_id);
                            completion_tokens++;
                        }
                        if (output.finished) {
                            finish_reason = output.finish_reason;
                        }
                    }
                }
            }
            
            openai::CompletionResponse response;
            response.id = request_id;
            response.model = model;
            
            openai::CompletionChoice choice;
            choice.index = 0;
            choice.text = generated_text;
            choice.finish_reason = finish_reason;
            response.choices.push_back(choice);
            
            response.usage.prompt_tokens = static_cast<int>(prompt_tokens.size());
            response.usage.completion_tokens = completion_tokens;
            response.usage.total_tokens = response.usage.prompt_tokens + completion_tokens;
            
            res.set_content(openai::to_json(response), "application/json");
            
        } catch (const std::exception& e) {
            send_error(res, e.what(), "internal_error", 500);
        }
    }
    
    void send_error(httplib::Response& res, const std::string& message, 
                    const std::string& type, int status_code) {
        openai::ErrorResponse error;
        error.error.message = message;
        error.error.type = type;
        res.status = status_code;
        res.set_content(openai::to_json(error), "application/json");
    }
    
    // Start the server (blocking)
    void run() {
        running_ = true;
        std::cout << "Starting vLLM C++ API Server..." << std::endl;
        std::cout << "  Model: " << config_.model_path << std::endl;
        std::cout << "  Served as: " << served_model_name_ << std::endl;
        std::cout << "  Listening on: http://" << config_.host << ":" << config_.port << std::endl;
        std::cout << std::endl;
        std::cout << "API Endpoints:" << std::endl;
        std::cout << "  GET  /health              - Health check" << std::endl;
        std::cout << "  GET  /v1/models           - List models" << std::endl;
        std::cout << "  POST /v1/chat/completions - Chat completions" << std::endl;
        std::cout << "  POST /v1/completions      - Text completions" << std::endl;
        std::cout << std::endl;
        
        server_.listen(config_.host, config_.port);
    }
    
    // Start the server in background thread
    void start() {
        if (!running_) {
            server_thread_ = std::thread([this]() {
                run();
            });
        }
    }
    
    // Stop the server
    void stop() {
        if (running_) {
            running_ = false;
            server_.stop();
            if (server_thread_.joinable()) {
                server_thread_.join();
            }
        }
    }
    
    bool is_running() const { return running_; }
    
private:
    std::string detect_model_name(const std::string& model_path) {
        std::string lower_path = model_path;
        std::transform(lower_path.begin(), lower_path.end(), lower_path.begin(), ::tolower);
        
        if (lower_path.find("qwen2") != std::string::npos ||
            lower_path.find("qwen-2") != std::string::npos) {
            return "qwen2";
        }
        if (lower_path.find("mimo") != std::string::npos) {
            return "mimo_v2_flash";
        }
        return "unknown";
    }
    
    std::string build_prompt_from_messages(const std::vector<openai::ChatMessage>& messages) {
        // Build chat prompt in ChatML format
        std::string prompt;
        for (const auto& msg : messages) {
            if (msg.role == "system") {
                prompt += "<|im_start|>system\n" + msg.content + "<|im_end|>\n";
            } else if (msg.role == "user") {
                prompt += "<|im_start|>user\n" + msg.content + "<|im_end|>\n";
            } else if (msg.role == "assistant") {
                prompt += "<|im_start|>assistant\n" + msg.content + "<|im_end|>\n";
            }
        }
        prompt += "<|im_start|>assistant\n";
        return prompt;
    }
    
    // Placeholder tokenizer - needs real implementation
    std::vector<int> tokenize(const std::string& text) {
        // TODO: Integrate with sentencepiece or HuggingFace tokenizers
        // For now, return placeholder tokens based on text length
        std::vector<int> tokens;
        // Simulate ~4 chars per token
        int num_tokens = static_cast<int>(text.size() / 4) + 1;
        for (int i = 0; i < num_tokens; ++i) {
            tokens.push_back(1000 + (i % 1000));  // Placeholder token IDs
        }
        return tokens;
    }
    
    // Placeholder detokenizer - needs real implementation
    std::string detokenize(int token_id) {
        // TODO: Integrate with sentencepiece or HuggingFace tokenizers
        // For now, return placeholder text
        return " [token:" + std::to_string(token_id) + "]";
    }
    
    ServerConfig config_;
    std::string served_model_name_;
    std::unique_ptr<LLMEngine> engine_;
    httplib::Server server_;
    std::thread server_thread_;
    std::atomic<bool> running_;
};

}  // namespace vllm
