// SPDX-License-Identifier: Apache-2.0
// vLLM C++ - Basic Inference Example

#include <iostream>
#include <vector>
#include <string>

#include "vllm/common.h"
#include "vllm/core/config.h"
#include "vllm/engine/engine.h"

using namespace vllm;

int main(int argc, char** argv) {
    std::cout << "vLLM C++ Basic Inference Example" << std::endl;
    std::cout << "=================================" << std::endl;
    
    // Check for model path argument
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <model_path> [prompt]" << std::endl;
        std::cout << "Example: " << argv[0] << " /path/to/Qwen2-7B \"Hello, how are you?\"" << std::endl;
        return 1;
    }
    
    std::string model_path = argv[1];
    std::string prompt = (argc > 2) ? argv[2] : "The capital of France is";
    
    try {
        // Print device info
        auto device_info = DeviceInfo::get(0);
        std::cout << "Device: " << device_info.name << std::endl;
        std::cout << "Compute Capability: " << device_info.compute_capability_major 
                  << "." << device_info.compute_capability_minor << std::endl;
        std::cout << "Memory: " << (device_info.total_memory / 1024 / 1024 / 1024) << " GB" << std::endl;
        std::cout << std::endl;
        
        // Create configuration
        VllmConfig config = VllmConfig::create_default(model_path);
        config.model_config.model_name = "qwen2";  // Set model type
        config.scheduler_config.max_num_seqs = 4;
        config.scheduler_config.max_num_batched_tokens = 2048;
        
        std::cout << "Loading model from: " << model_path << std::endl;
        
        // Create engine
        LLMEngine engine(config);
        
        std::cout << "Model loaded successfully!" << std::endl;
        std::cout << "Free KV cache blocks: " << engine.num_free_blocks() << std::endl;
        std::cout << std::endl;
        
        // Create sampling parameters
        SamplingParams sampling_params;
        sampling_params.max_tokens = 100;
        sampling_params.temperature = 0.7f;
        sampling_params.top_p = 0.9f;
        
        // Tokenize prompt (simplified - in real usage, use a tokenizer)
        // This is a placeholder - actual tokenization would use the model's tokenizer
        std::vector<int> prompt_tokens;
        std::cout << "Note: This example uses placeholder tokenization." << std::endl;
        std::cout << "In production, integrate with the actual tokenizer." << std::endl;
        
        // Add request
        std::string request_id = "request-1";
        engine.add_request(request_id, prompt_tokens, sampling_params, /*eos_token_id=*/151643);
        
        std::cout << "Request added: " << request_id << std::endl;
        std::cout << "Prompt: " << prompt << std::endl;
        std::cout << std::endl;
        
        // Run generation
        std::cout << "Generated tokens: ";
        
        engine.run([](const std::vector<RequestOutput>& outputs) {
            for (const auto& output : outputs) {
                for (int token_id : output.new_token_ids) {
                    std::cout << token_id << " ";
                }
                std::cout.flush();
                
                if (output.finished) {
                    std::cout << std::endl;
                    std::cout << "Finish reason: " << output.finish_reason << std::endl;
                }
            }
        });
        
        std::cout << std::endl;
        std::cout << "Generation complete!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
