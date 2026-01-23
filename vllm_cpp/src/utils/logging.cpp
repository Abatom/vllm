// SPDX-License-Identifier: Apache-2.0
// vLLM C++ Implementation - Logging Utilities

#include <iostream>
#include <iomanip>
#include <chrono>
#include <sstream>
#include <mutex>

namespace vllm {
namespace logging {

enum class LogLevel {
    DEBUG = 0,
    INFO = 1,
    WARNING = 2,
    ERROR = 3,
    FATAL = 4
};

// Global log level
static LogLevel g_log_level = LogLevel::INFO;
static std::mutex g_log_mutex;

// Set global log level
void set_log_level(LogLevel level) {
    g_log_level = level;
}

// Get log level string
std::string level_to_string(LogLevel level) {
    switch (level) {
        case LogLevel::DEBUG: return "DEBUG";
        case LogLevel::INFO: return "INFO";
        case LogLevel::WARNING: return "WARNING";
        case LogLevel::ERROR: return "ERROR";
        case LogLevel::FATAL: return "FATAL";
        default: return "UNKNOWN";
    }
}

// Get current timestamp
std::string get_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S");
    ss << '.' << std::setfill('0') << std::setw(3) << ms.count();
    return ss.str();
}

// Log a message
void log(LogLevel level, const std::string& message, const char* file, int line) {
    if (level < g_log_level) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(g_log_mutex);
    
    std::ostream& out = (level >= LogLevel::WARNING) ? std::cerr : std::cout;
    
    out << "[" << get_timestamp() << "] "
        << "[" << level_to_string(level) << "] ";
    
    if (level >= LogLevel::WARNING) {
        out << "[" << file << ":" << line << "] ";
    }
    
    out << message << std::endl;
    
    if (level == LogLevel::FATAL) {
        std::abort();
    }
}

}  // namespace logging
}  // namespace vllm

// Macros for logging
#define VLLM_LOG_DEBUG(msg) vllm::logging::log(vllm::logging::LogLevel::DEBUG, msg, __FILE__, __LINE__)
#define VLLM_LOG_INFO(msg) vllm::logging::log(vllm::logging::LogLevel::INFO, msg, __FILE__, __LINE__)
#define VLLM_LOG_WARNING(msg) vllm::logging::log(vllm::logging::LogLevel::WARNING, msg, __FILE__, __LINE__)
#define VLLM_LOG_ERROR(msg) vllm::logging::log(vllm::logging::LogLevel::ERROR, msg, __FILE__, __LINE__)
#define VLLM_LOG_FATAL(msg) vllm::logging::log(vllm::logging::LogLevel::FATAL, msg, __FILE__, __LINE__)
