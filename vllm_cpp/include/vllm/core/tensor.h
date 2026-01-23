// SPDX-License-Identifier: Apache-2.0
// vLLM C++ Implementation - Tensor Utilities
#pragma once

#include "vllm/common.h"

namespace vllm {

// Tensor wrapper with additional utilities
class TensorWrapper {
public:
    TensorWrapper() = default;
    explicit TensorWrapper(torch::Tensor tensor) : tensor_(std::move(tensor)) {}
    
    // Factory methods
    static TensorWrapper empty(
        const std::vector<int64_t>& shape,
        DataType dtype,
        int device_id = 0
    ) {
        auto options = torch::TensorOptions()
            .dtype(to_torch_dtype(dtype))
            .device(torch::kCUDA, device_id);
        return TensorWrapper(torch::empty(shape, options));
    }
    
    static TensorWrapper zeros(
        const std::vector<int64_t>& shape,
        DataType dtype,
        int device_id = 0
    ) {
        auto options = torch::TensorOptions()
            .dtype(to_torch_dtype(dtype))
            .device(torch::kCUDA, device_id);
        return TensorWrapper(torch::zeros(shape, options));
    }
    
    static TensorWrapper ones(
        const std::vector<int64_t>& shape,
        DataType dtype,
        int device_id = 0
    ) {
        auto options = torch::TensorOptions()
            .dtype(to_torch_dtype(dtype))
            .device(torch::kCUDA, device_id);
        return TensorWrapper(torch::ones(shape, options));
    }
    
    // Accessors
    torch::Tensor& tensor() { return tensor_; }
    const torch::Tensor& tensor() const { return tensor_; }
    
    torch::Tensor* operator->() { return &tensor_; }
    const torch::Tensor* operator->() const { return &tensor_; }
    
    // Properties
    int64_t size(int dim) const { return tensor_.size(dim); }
    int64_t numel() const { return tensor_.numel(); }
    int64_t dim() const { return tensor_.dim(); }
    std::vector<int64_t> shape() const { return get_shape(tensor_); }
    DataType dtype() const { return from_torch_dtype(tensor_.scalar_type()); }
    int device_id() const { return tensor_.device().index(); }
    bool is_cuda() const { return tensor_.is_cuda(); }
    bool is_contiguous() const { return tensor_.is_contiguous(); }
    
    // Data pointers
    template<typename T>
    T* data() { return tensor_.data_ptr<T>(); }
    
    template<typename T>
    const T* data() const { return tensor_.data_ptr<T>(); }
    
    void* data_ptr() { return tensor_.data_ptr(); }
    const void* data_ptr() const { return tensor_.data_ptr(); }
    
    // Operations
    TensorWrapper contiguous() const {
        return TensorWrapper(tensor_.contiguous());
    }
    
    TensorWrapper to(DataType dtype) const {
        return TensorWrapper(tensor_.to(to_torch_dtype(dtype)));
    }
    
    TensorWrapper to(int device_id) const {
        return TensorWrapper(tensor_.to(torch::Device(torch::kCUDA, device_id)));
    }
    
    TensorWrapper view(const std::vector<int64_t>& shape) const {
        return TensorWrapper(tensor_.view(shape));
    }
    
    TensorWrapper reshape(const std::vector<int64_t>& shape) const {
        return TensorWrapper(tensor_.reshape(shape));
    }
    
    TensorWrapper transpose(int dim0, int dim1) const {
        return TensorWrapper(tensor_.transpose(dim0, dim1));
    }
    
    TensorWrapper clone() const {
        return TensorWrapper(tensor_.clone());
    }
    
    // Slicing
    TensorWrapper slice(int dim, int64_t start, int64_t end) const {
        return TensorWrapper(tensor_.slice(dim, start, end));
    }
    
    TensorWrapper index(const std::vector<torch::Tensor>& indices) const {
        return TensorWrapper(tensor_.index(
            std::vector<at::indexing::TensorIndex>(indices.begin(), indices.end())
        ));
    }
    
    // Check validity
    bool defined() const { return tensor_.defined(); }
    operator bool() const { return defined(); }
    
private:
    torch::Tensor tensor_;
};

// Intermediate tensors for pipeline parallelism
class IntermediateTensors {
public:
    void set(const std::string& key, TensorWrapper tensor) {
        tensors_[key] = std::move(tensor);
    }
    
    TensorWrapper& get(const std::string& key) {
        return tensors_.at(key);
    }
    
    const TensorWrapper& get(const std::string& key) const {
        return tensors_.at(key);
    }
    
    bool has(const std::string& key) const {
        return tensors_.find(key) != tensors_.end();
    }
    
    void clear() {
        tensors_.clear();
    }
    
private:
    std::unordered_map<std::string, TensorWrapper> tensors_;
};

}  // namespace vllm
