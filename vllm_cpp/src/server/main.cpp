// SPDX-License-Identifier: Apache-2.0
// vLLM C++ - API Server Main Entry Point

#include <iostream>
#include <string>
#include <csignal>
#include <cstdlib>

#include "vllm/server/api_server.h"

using namespace vllm;

// Global server pointer for signal handling
static APIServer* g_server = nullptr;

void signal_handler(int signum) {
    std::cout << "\nReceived signal " << signum << ", shutting down..." << std::endl;
    if (g_server) {
        g_server->stop();
    }
    std::exit(0);
}

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]" << std::endl;
    std::cout << std::endl;
    std::cout << "vLLM C++ OpenAI-Compatible API Server" << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --model <path>              Path to the model directory (required)" << std::endl;
    std::cout << "  --served-model-name <name>  Name to serve the model as (default: auto-detect)" << std::endl;
    std::cout << "  --host <host>               Host to bind to (default: 0.0.0.0)" << std::endl;
    std::cout << "  --port <port>               Port to listen on (default: 8000)" << std::endl;
    std::cout << "  --max-model-len <len>       Maximum model context length (default: model default)" << std::endl;
    std::cout << "  --gpu-memory-utilization <f> GPU memory utilization (default: 0.9)" << std::endl;
    std::cout << "  --tensor-parallel-size <n>  Tensor parallel size (default: 1)" << std::endl;
    std::cout << "  --max-concurrent <n>        Maximum concurrent requests (default: 256)" << std::endl;
    std::cout << "  --help                      Show this help message" << std::endl;
    std::cout << std::endl;
    std::cout << "Example:" << std::endl;
    std::cout << "  " << program_name << " --model /path/to/mimo-v2-flash --port 8000" << std::endl;
    std::cout << std::endl;
    std::cout << "API Endpoints:" << std::endl;
    std::cout << "  GET  /health              - Health check" << std::endl;
    std::cout << "  GET  /v1/models           - List available models" << std::endl;
    std::cout << "  POST /v1/chat/completions - Chat completions (OpenAI compatible)" << std::endl;
    std::cout << "  POST /v1/completions      - Text completions (OpenAI compatible)" << std::endl;
}

int main(int argc, char* argv[]) {
    ServerConfig config;
    bool show_help = false;
    bool model_specified = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help" || arg == "-h") {
            show_help = true;
        } else if (arg == "--model" && i + 1 < argc) {
            config.model_path = argv[++i];
            model_specified = true;
        } else if (arg == "--served-model-name" && i + 1 < argc) {
            config.served_model_name = argv[++i];
        } else if (arg == "--host" && i + 1 < argc) {
            config.host = argv[++i];
        } else if (arg == "--port" && i + 1 < argc) {
            config.port = std::stoi(argv[++i]);
        } else if (arg == "--max-model-len" && i + 1 < argc) {
            config.max_model_len = std::stoi(argv[++i]);
        } else if (arg == "--gpu-memory-utilization" && i + 1 < argc) {
            config.gpu_memory_utilization = std::stof(argv[++i]);
        } else if (arg == "--tensor-parallel-size" && i + 1 < argc) {
            config.tensor_parallel_size = std::stoi(argv[++i]);
        } else if (arg == "--max-concurrent" && i + 1 < argc) {
            config.max_concurrent_requests = std::stoi(argv[++i]);
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }
    
    if (show_help) {
        print_usage(argv[0]);
        return 0;
    }
    
    if (!model_specified) {
        std::cerr << "Error: --model is required" << std::endl;
        std::cerr << std::endl;
        print_usage(argv[0]);
        return 1;
    }
    
    // Set up signal handlers
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);
    
    try {
        std::cout << "============================================" << std::endl;
        std::cout << "     vLLM C++ OpenAI-Compatible Server     " << std::endl;
        std::cout << "============================================" << std::endl;
        std::cout << std::endl;
        
        // Create and run server
        APIServer server(config);
        g_server = &server;
        
        server.run();
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
