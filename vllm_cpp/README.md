# vLLM C++ Implementation

A C++ implementation of vLLM for NVIDIA GPUs, focusing on the V1 engine architecture.

## Features

- **V1 Engine Only**: Clean implementation based on vLLM's V1 architecture
- **NVIDIA GPU Support**: Optimized CUDA kernels for NVIDIA GPUs
- **OpenAI-Compatible API Server**: HTTP server with /v1/chat/completions, /v1/completions endpoints
- **Supported Models**:
  - Qwen2 (Qwen2-7B, Qwen2-72B, etc.)
  - MiMoV2Flash (Mixture of Experts with Flash Attention)
- **Core Components**:
  - Paged Attention with KV Cache management
  - Efficient scheduler with continuous batching
  - Tensor parallelism support
  - Flash Attention integration
  - FP16/BF16 support

## Requirements

- CMake >= 3.26
- CUDA Toolkit >= 11.8
- PyTorch >= 2.0 (for libtorch)
- C++17 compatible compiler
- NVIDIA GPU with compute capability >= 8.0 (Ampere or newer recommended)

## Project Structure

```
vllm_cpp/
├── CMakeLists.txt           # Main CMake configuration
├── README.md                # This file
├── include/
│   └── vllm/
│       ├── common.h         # Common definitions and utilities
│       ├── core/
│       │   ├── config.h     # Configuration classes
│       │   ├── tensor.h     # Tensor utilities
│       │   └── kv_cache.h   # KV cache management
│       ├── engine/
│       │   ├── request.h    # Request definition
│       │   ├── scheduler.h  # V1 Scheduler
│       │   └── engine.h     # LLM Engine
│       └── model_executor/
│           ├── layers/
│           │   ├── linear.h           # Linear layers
│           │   ├── layernorm.h        # RMSNorm
│           │   ├── activation.h       # SiLU, GELU, etc.
│           │   ├── rotary_embedding.h # RoPE
│           │   ├── attention.h        # Attention layer
│           │   └── mlp.h              # MLP and MoE
│           └── models/
│               ├── qwen2.h            # Qwen2 model
│               └── mimo_v2_flash.h    # MiMoV2Flash model
├── src/
│   └── kernels/
│       ├── activation/      # Activation CUDA kernels
│       ├── layernorm/       # LayerNorm CUDA kernels
│       ├── attention/       # Attention and RoPE kernels
│       └── cache/           # KV cache kernels
├── examples/
│   ├── basic_inference.cpp  # Basic inference example
│   ├── qwen2_example.cpp    # Qwen2 model example
│   └── mimo_v2_flash_example.cpp  # MiMoV2Flash example
└── tests/
    ├── test_layers_simple.cpp   # Layer unit tests
    └── test_models_simple.cpp   # Model integration tests
```

## Building

### Prerequisites

1. Install CUDA Toolkit:
```bash
# Ubuntu
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda
```

2. Install PyTorch (for libtorch):
```bash
pip install torch
```

### Build Steps

```bash
cd vllm_cpp

# Create build directory
mkdir build && cd build

# Configure
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
make -j$(nproc)
```

### Build Options

- `VLLM_BUILD_TESTS`: Build test programs (default: ON)
- `VLLM_BUILD_EXAMPLES`: Build example programs (default: ON)
- `VLLM_USE_FLASH_ATTENTION`: Enable FlashAttention (default: ON)
- `VLLM_USE_CUTLASS`: Enable CUTLASS optimizations (default: ON)

Example with options:
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DVLLM_BUILD_TESTS=ON \
         -DVLLM_USE_FLASH_ATTENTION=ON
```

## Usage

### Starting the API Server

The easiest way to use vLLM C++ is through the OpenAI-compatible API server:

```bash
# Start the server with a MiMoV2Flash model
./vllm_server --model /path/to/mimo-v2-flash --port 8000

# Or with a Qwen2 model
./vllm_server --model /path/to/Qwen2-7B --port 8000 --served-model-name qwen2-7b
```

**Server Options:**
```
--model <path>              Path to the model directory (required)
--served-model-name <name>  Name to serve the model as
--host <host>               Host to bind to (default: 0.0.0.0)
--port <port>               Port to listen on (default: 8000)
--max-model-len <len>       Maximum model context length
--gpu-memory-utilization <f> GPU memory utilization (default: 0.9)
--tensor-parallel-size <n>  Tensor parallel size (default: 1)
```

### Making API Requests

**Chat Completions:**
```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "mimo-v2-flash",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ],
        "max_tokens": 100,
        "temperature": 0.7
    }'
```

**Text Completions:**
```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "mimo-v2-flash",
        "prompt": "The capital of France is",
        "max_tokens": 50
    }'
```

**List Models:**
```bash
curl http://localhost:8000/v1/models
```

### Basic Inference

```cpp
#include "vllm/engine/engine.h"

using namespace vllm;

int main() {
    // Create configuration
    VllmConfig config = VllmConfig::create_default("/path/to/model");
    config.model_config.model_name = "qwen2";
    
    // Create engine
    LLMEngine engine(config);
    
    // Create sampling parameters
    SamplingParams sampling_params;
    sampling_params.max_tokens = 100;
    sampling_params.temperature = 0.7f;
    
    // Add request
    std::vector<int> prompt_tokens = {1, 2, 3, 4};  // Your tokenized prompt
    engine.add_request("request-1", prompt_tokens, sampling_params);
    
    // Run generation
    engine.run([](const std::vector<RequestOutput>& outputs) {
        for (const auto& output : outputs) {
            for (int token_id : output.new_token_ids) {
                std::cout << token_id << " ";
            }
        }
    });
    
    return 0;
}
```

### Qwen2 Model Direct Usage

```cpp
#include "vllm/model_executor/models/qwen2.h"

using namespace vllm;

int main() {
    // Create config
    Qwen2Config config;
    config.hidden_size = 4096;
    config.num_hidden_layers = 32;
    // ... set other config options
    config.compute_derived();
    
    // Create model
    Qwen2ForCausalLM model(config, nullptr, 1);
    
    // Forward pass
    auto input_ids = torch::randint(0, config.vocab_size, {4});
    auto positions = torch::arange(0, 4);
    
    auto hidden = model.forward(input_ids, positions);
    auto logits = model.compute_logits(hidden);
    
    return 0;
}
```

## Supported Models

### Qwen2

The Qwen2 model family including:
- Qwen2-0.5B
- Qwen2-1.5B
- Qwen2-7B
- Qwen2-14B
- Qwen2-72B

Features:
- Group Query Attention (GQA)
- RoPE position encoding
- SiLU activation
- RMSNorm

### MiMoV2Flash

Mixture of Experts model with:
- Dense and MoE layers
- Sliding Window Attention (SWA) for some layers
- Grouped top-k expert routing
- E-score correction bias
- Different head dimensions for K and V

## Performance

The C++ implementation provides:
- Native CUDA kernel execution without Python overhead
- Efficient memory management with paged attention
- Optimized tensor operations via libtorch
- Continuous batching for high throughput

## Differences from Python vLLM

1. **V1 Only**: Only implements the V1 engine architecture
2. **No LoRA**: LoRA functionality is not included
3. **NVIDIA Only**: Only supports NVIDIA GPUs (no ROCm, CPU, TPU)
4. **Limited Models**: Only Qwen2 and MiMoV2Flash models supported
5. **No Python Bindings**: Pure C++ implementation

## License

Apache-2.0 License

## Acknowledgments

This project is based on [vLLM](https://github.com/vllm-project/vllm), a high-throughput and memory-efficient inference and serving engine for LLMs.
