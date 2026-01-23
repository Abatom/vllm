// SPDX-License-Identifier: Apache-2.0
// vLLM C++ - OpenAI API Types
#pragma once

#include <string>
#include <vector>
#include <optional>
#include <chrono>

// Use nlohmann/json if available, otherwise provide minimal JSON support
#if __has_include(<nlohmann/json.hpp>)
#include <nlohmann/json.hpp>
#define VLLM_HAS_NLOHMANN_JSON 1
#else
#define VLLM_HAS_NLOHMANN_JSON 0
#endif

namespace vllm {
namespace openai {

// ============================================================================
// Request Types
// ============================================================================

// Chat message
struct ChatMessage {
    std::string role;      // "system", "user", "assistant"
    std::string content;
    std::optional<std::string> name;
};

// Chat completion request
struct ChatCompletionRequest {
    std::string model;
    std::vector<ChatMessage> messages;
    float temperature = 1.0f;
    float top_p = 1.0f;
    int n = 1;
    bool stream = false;
    std::optional<std::vector<std::string>> stop;
    int max_tokens = 16;
    float presence_penalty = 0.0f;
    float frequency_penalty = 0.0f;
    std::optional<std::string> user;
    
    // vLLM extensions
    int top_k = -1;
    float min_p = 0.0f;
    float repetition_penalty = 1.0f;
    bool skip_special_tokens = true;
};

// Completion request (legacy)
struct CompletionRequest {
    std::string model;
    std::string prompt;
    float temperature = 1.0f;
    float top_p = 1.0f;
    int n = 1;
    bool stream = false;
    std::optional<std::vector<std::string>> stop;
    int max_tokens = 16;
    float presence_penalty = 0.0f;
    float frequency_penalty = 0.0f;
    std::optional<std::string> user;
    bool echo = false;
    
    // vLLM extensions
    int top_k = -1;
    float min_p = 0.0f;
    float repetition_penalty = 1.0f;
};

// ============================================================================
// Response Types
// ============================================================================

// Usage statistics
struct UsageInfo {
    int prompt_tokens = 0;
    int completion_tokens = 0;
    int total_tokens = 0;
};

// Choice for completion
struct CompletionChoice {
    int index = 0;
    std::string text;
    std::optional<std::string> finish_reason;
};

// Completion response
struct CompletionResponse {
    std::string id;
    std::string object = "text_completion";
    int64_t created = 0;
    std::string model;
    std::vector<CompletionChoice> choices;
    UsageInfo usage;
    
    CompletionResponse() {
        created = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()
        ).count();
    }
};

// Choice for chat completion
struct ChatCompletionChoice {
    int index = 0;
    ChatMessage message;
    std::optional<std::string> finish_reason;
};

// Chat completion response
struct ChatCompletionResponse {
    std::string id;
    std::string object = "chat.completion";
    int64_t created = 0;
    std::string model;
    std::vector<ChatCompletionChoice> choices;
    UsageInfo usage;
    
    ChatCompletionResponse() {
        created = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()
        ).count();
    }
};

// Streaming delta
struct ChatCompletionDelta {
    std::optional<std::string> role;
    std::optional<std::string> content;
};

// Streaming choice
struct ChatCompletionStreamChoice {
    int index = 0;
    ChatCompletionDelta delta;
    std::optional<std::string> finish_reason;
};

// Streaming response chunk
struct ChatCompletionChunk {
    std::string id;
    std::string object = "chat.completion.chunk";
    int64_t created = 0;
    std::string model;
    std::vector<ChatCompletionStreamChoice> choices;
    
    ChatCompletionChunk() {
        created = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()
        ).count();
    }
};

// Model info
struct ModelInfo {
    std::string id;
    std::string object = "model";
    int64_t created = 0;
    std::string owned_by = "vllm";
};

// Models list response
struct ModelsResponse {
    std::string object = "list";
    std::vector<ModelInfo> data;
};

// Error response
struct ErrorDetail {
    std::string message;
    std::string type;
    std::optional<std::string> param;
    std::optional<std::string> code;
};

struct ErrorResponse {
    ErrorDetail error;
};

// ============================================================================
// JSON Serialization (using simple string building if nlohmann not available)
// ============================================================================

inline std::string escape_json_string(const std::string& s) {
    std::string result;
    result.reserve(s.size() + 16);
    for (char c : s) {
        switch (c) {
            case '"': result += "\\\""; break;
            case '\\': result += "\\\\"; break;
            case '\b': result += "\\b"; break;
            case '\f': result += "\\f"; break;
            case '\n': result += "\\n"; break;
            case '\r': result += "\\r"; break;
            case '\t': result += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned char>(c));
                    result += buf;
                } else {
                    result += c;
                }
        }
    }
    return result;
}

inline std::string to_json(const UsageInfo& u) {
    return "{\"prompt_tokens\":" + std::to_string(u.prompt_tokens) +
           ",\"completion_tokens\":" + std::to_string(u.completion_tokens) +
           ",\"total_tokens\":" + std::to_string(u.total_tokens) + "}";
}

inline std::string to_json(const CompletionChoice& c) {
    std::string json = "{\"index\":" + std::to_string(c.index) +
                       ",\"text\":\"" + escape_json_string(c.text) + "\"";
    if (c.finish_reason) {
        json += ",\"finish_reason\":\"" + *c.finish_reason + "\"";
    } else {
        json += ",\"finish_reason\":null";
    }
    json += "}";
    return json;
}

inline std::string to_json(const CompletionResponse& r) {
    std::string json = "{\"id\":\"" + r.id + "\"" +
                       ",\"object\":\"" + r.object + "\"" +
                       ",\"created\":" + std::to_string(r.created) +
                       ",\"model\":\"" + r.model + "\"" +
                       ",\"choices\":[";
    for (size_t i = 0; i < r.choices.size(); ++i) {
        if (i > 0) json += ",";
        json += to_json(r.choices[i]);
    }
    json += "],\"usage\":" + to_json(r.usage) + "}";
    return json;
}

inline std::string to_json(const ChatMessage& m) {
    return "{\"role\":\"" + m.role + "\",\"content\":\"" + escape_json_string(m.content) + "\"}";
}

inline std::string to_json(const ChatCompletionChoice& c) {
    std::string json = "{\"index\":" + std::to_string(c.index) +
                       ",\"message\":" + to_json(c.message);
    if (c.finish_reason) {
        json += ",\"finish_reason\":\"" + *c.finish_reason + "\"";
    } else {
        json += ",\"finish_reason\":null";
    }
    json += "}";
    return json;
}

inline std::string to_json(const ChatCompletionResponse& r) {
    std::string json = "{\"id\":\"" + r.id + "\"" +
                       ",\"object\":\"" + r.object + "\"" +
                       ",\"created\":" + std::to_string(r.created) +
                       ",\"model\":\"" + r.model + "\"" +
                       ",\"choices\":[";
    for (size_t i = 0; i < r.choices.size(); ++i) {
        if (i > 0) json += ",";
        json += to_json(r.choices[i]);
    }
    json += "],\"usage\":" + to_json(r.usage) + "}";
    return json;
}

inline std::string to_json(const ChatCompletionDelta& d) {
    std::string json = "{";
    bool first = true;
    if (d.role) {
        json += "\"role\":\"" + *d.role + "\"";
        first = false;
    }
    if (d.content) {
        if (!first) json += ",";
        json += "\"content\":\"" + escape_json_string(*d.content) + "\"";
    }
    json += "}";
    return json;
}

inline std::string to_json(const ChatCompletionStreamChoice& c) {
    std::string json = "{\"index\":" + std::to_string(c.index) +
                       ",\"delta\":" + to_json(c.delta);
    if (c.finish_reason) {
        json += ",\"finish_reason\":\"" + *c.finish_reason + "\"";
    } else {
        json += ",\"finish_reason\":null";
    }
    json += "}";
    return json;
}

inline std::string to_json(const ChatCompletionChunk& r) {
    std::string json = "{\"id\":\"" + r.id + "\"" +
                       ",\"object\":\"" + r.object + "\"" +
                       ",\"created\":" + std::to_string(r.created) +
                       ",\"model\":\"" + r.model + "\"" +
                       ",\"choices\":[";
    for (size_t i = 0; i < r.choices.size(); ++i) {
        if (i > 0) json += ",";
        json += to_json(r.choices[i]);
    }
    json += "]}";
    return json;
}

inline std::string to_json(const ModelInfo& m) {
    return "{\"id\":\"" + m.id + "\"" +
           ",\"object\":\"" + m.object + "\"" +
           ",\"created\":" + std::to_string(m.created) +
           ",\"owned_by\":\"" + m.owned_by + "\"}";
}

inline std::string to_json(const ModelsResponse& r) {
    std::string json = "{\"object\":\"" + r.object + "\",\"data\":[";
    for (size_t i = 0; i < r.data.size(); ++i) {
        if (i > 0) json += ",";
        json += to_json(r.data[i]);
    }
    json += "]}";
    return json;
}

inline std::string to_json(const ErrorResponse& e) {
    std::string json = "{\"error\":{\"message\":\"" + escape_json_string(e.error.message) + "\"" +
                       ",\"type\":\"" + e.error.type + "\"";
    if (e.error.param) {
        json += ",\"param\":\"" + *e.error.param + "\"";
    } else {
        json += ",\"param\":null";
    }
    if (e.error.code) {
        json += ",\"code\":\"" + *e.error.code + "\"";
    } else {
        json += ",\"code\":null";
    }
    json += "}}";
    return json;
}

// Generate unique request ID
inline std::string generate_request_id() {
    static std::atomic<uint64_t> counter{0};
    auto now = std::chrono::system_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()
    ).count();
    return "chatcmpl-" + std::to_string(ms) + "-" + std::to_string(counter++);
}

}  // namespace openai
}  // namespace vllm
