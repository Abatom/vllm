// SPDX-License-Identifier: Apache-2.0
// vLLM C++ Implementation - Common Definitions
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <torch/torch.h>
#include <ATen/cuda/CUDAContext.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>
#include <unordered_map>
#include <functional>
#include <stdexcept>

namespace vllm {

// Version info
constexpr int VERSION_MAJOR = 1;
constexpr int VERSION_MINOR = 0;
constexpr int VERSION_PATCH = 0;

// Data types supported
enum class DataType {
    kFloat32,
    kFloat16,
    kBFloat16,
    kInt8,
    kInt32,
    kInt64,
    kFP8E4M3,
    kFP8E5M2,
};

// Convert DataType to torch::ScalarType
inline torch::ScalarType to_torch_dtype(DataType dtype) {
    switch (dtype) {
        case DataType::kFloat32: return torch::kFloat32;
        case DataType::kFloat16: return torch::kFloat16;
        case DataType::kBFloat16: return torch::kBFloat16;
        case DataType::kInt8: return torch::kInt8;
        case DataType::kInt32: return torch::kInt32;
        case DataType::kInt64: return torch::kInt64;
        default:
            throw std::runtime_error("Unsupported data type for torch conversion");
    }
}

// Convert torch::ScalarType to DataType
inline DataType from_torch_dtype(torch::ScalarType dtype) {
    switch (dtype) {
        case torch::kFloat32: return DataType::kFloat32;
        case torch::kFloat16: return DataType::kFloat16;
        case torch::kBFloat16: return DataType::kBFloat16;
        case torch::kInt8: return DataType::kInt8;
        case torch::kInt32: return DataType::kInt32;
        case torch::kInt64: return DataType::kInt64;
        default:
            throw std::runtime_error("Unsupported torch dtype for conversion");
    }
}

// Get size of data type in bytes
inline size_t dtype_size(DataType dtype) {
    switch (dtype) {
        case DataType::kFloat32: return 4;
        case DataType::kFloat16: return 2;
        case DataType::kBFloat16: return 2;
        case DataType::kInt8: return 1;
        case DataType::kInt32: return 4;
        case DataType::kInt64: return 8;
        case DataType::kFP8E4M3: return 1;
        case DataType::kFP8E5M2: return 1;
        default: return 0;
    }
}

// CUDA error checking macro
#define VLLM_CUDA_CHECK(call)                                                    \
    do {                                                                         \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess) {                                                \
            throw std::runtime_error(                                            \
                std::string("CUDA error: ") + cudaGetErrorString(err) +          \
                " at " + __FILE__ + ":" + std::to_string(__LINE__));             \
        }                                                                        \
    } while (0)

// Assertion macro
#define VLLM_ASSERT(cond, msg)                                                   \
    do {                                                                         \
        if (!(cond)) {                                                           \
            throw std::runtime_error(                                            \
                std::string("Assertion failed: ") + msg +                        \
                " at " + __FILE__ + ":" + std::to_string(__LINE__));             \
        }                                                                        \
    } while (0)

// Get current CUDA stream
inline cudaStream_t get_cuda_stream() {
    return at::cuda::getCurrentCUDAStream().stream();
}

// Tensor shape utilities
inline std::vector<int64_t> get_shape(const torch::Tensor& tensor) {
    std::vector<int64_t> shape;
    for (int i = 0; i < tensor.dim(); ++i) {
        shape.push_back(tensor.size(i));
    }
    return shape;
}

// Device information
struct DeviceInfo {
    int device_id;
    std::string name;
    int compute_capability_major;
    int compute_capability_minor;
    size_t total_memory;
    int sm_count;
    int max_threads_per_sm;
    
    static DeviceInfo get(int device_id = 0) {
        DeviceInfo info;
        info.device_id = device_id;
        
        cudaDeviceProp prop;
        VLLM_CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
        
        info.name = prop.name;
        info.compute_capability_major = prop.major;
        info.compute_capability_minor = prop.minor;
        info.total_memory = prop.totalGlobalMem;
        info.sm_count = prop.multiProcessorCount;
        info.max_threads_per_sm = prop.maxThreadsPerMultiProcessor;
        
        return info;
    }
    
    int compute_capability() const {
        return compute_capability_major * 10 + compute_capability_minor;
    }
};

}  // namespace vllm
