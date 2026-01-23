// SPDX-License-Identifier: Apache-2.0
// vLLM C++ Implementation - Main Header
#pragma once

// Common utilities
#include "vllm/common.h"

// Core components
#include "vllm/core/config.h"
#include "vllm/core/tensor.h"
#include "vllm/core/kv_cache.h"

// Model executor layers
#include "vllm/model_executor/layers/linear.h"
#include "vllm/model_executor/layers/layernorm.h"
#include "vllm/model_executor/layers/activation.h"
#include "vllm/model_executor/layers/rotary_embedding.h"
#include "vllm/model_executor/layers/attention.h"
#include "vllm/model_executor/layers/mlp.h"

// Models
#include "vllm/model_executor/models/qwen2.h"
#include "vllm/model_executor/models/mimo_v2_flash.h"

// Engine components
#include "vllm/engine/request.h"
#include "vllm/engine/scheduler.h"
#include "vllm/engine/engine.h"

namespace vllm {

// Version information
inline std::string version() {
    return std::to_string(VERSION_MAJOR) + "." + 
           std::to_string(VERSION_MINOR) + "." + 
           std::to_string(VERSION_PATCH);
}

// Initialize vLLM (call once at startup)
inline void initialize(int device_id = 0) {
    torch::cuda::set_device(device_id);
}

}  // namespace vllm
