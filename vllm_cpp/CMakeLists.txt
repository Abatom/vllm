# SPDX-License-Identifier: Apache-2.0
# vLLM C++ Implementation - CMake Build Configuration
# Only supports NVIDIA GPUs (CUDA), V1 engine only
# Supported models: Qwen2, MiMoV2Flash

cmake_minimum_required(VERSION 3.26)

# Set CUDA architectures before project()
# Supports Ampere (80, 86), Ada Lovelace (89), and Hopper (90)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "OFF")
    set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90" CACHE STRING "CUDA architectures" FORCE)
endif()

project(vllm_cpp 
    VERSION 1.0.0
    LANGUAGES CXX CUDA
    DESCRIPTION "vLLM C++ Implementation for NVIDIA GPUs"
)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# CUDA Standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build options
option(VLLM_BUILD_TESTS "Build test programs" ON)
option(VLLM_BUILD_EXAMPLES "Build example programs" ON)
option(VLLM_BUILD_SERVER "Build API server" ON)
option(VLLM_USE_FLASH_ATTENTION "Use FlashAttention" ON)
option(VLLM_USE_CUTLASS "Use CUTLASS for optimized kernels" ON)

# Find required packages
find_package(CUDAToolkit REQUIRED)

# Find Torch
find_package(Torch QUIET)
if(NOT Torch_FOUND)
    message(STATUS "PyTorch not found via find_package, trying to locate via Python")
    execute_process(
        COMMAND python3 -c "import torch; print(torch.utils.cmake_prefix_path)"
        OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(TORCH_CMAKE_PREFIX)
        list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX})
        find_package(Torch REQUIRED)
    else()
        message(FATAL_ERROR "PyTorch is required but not found. Please install PyTorch or set CMAKE_PREFIX_PATH.")
    endif()
endif()

# Find optional packages
find_package(OpenMP QUIET)

# CUDA flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g")
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
    ${TORCH_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
)

# CUTLASS (optional)
if(VLLM_USE_CUTLASS)
    set(CUTLASS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass" CACHE PATH "CUTLASS directory")
    if(EXISTS ${CUTLASS_DIR})
        include_directories(${CUTLASS_DIR}/include)
        add_definitions(-DVLLM_USE_CUTLASS)
    else()
        message(WARNING "CUTLASS not found at ${CUTLASS_DIR}. Some optimizations will be disabled.")
        set(VLLM_USE_CUTLASS OFF)
    endif()
endif()

# FlashAttention (optional)
if(VLLM_USE_FLASH_ATTENTION)
    add_definitions(-DVLLM_USE_FLASH_ATTENTION)
endif()

# Collect source files
file(GLOB_RECURSE CORE_SOURCES 
    "${CMAKE_CURRENT_SOURCE_DIR}/src/core/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/core/*.cu"
)

file(GLOB_RECURSE ENGINE_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/engine/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/engine/*.cu"
)

file(GLOB_RECURSE MODEL_EXECUTOR_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/model_executor/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/model_executor/*.cu"
)

file(GLOB_RECURSE DISTRIBUTED_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/distributed/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/distributed/*.cu"
)

file(GLOB_RECURSE ATTENTION_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/attention/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/attention/*.cu"
)

file(GLOB_RECURSE UTILS_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/utils/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/utils/*.cu"
)

file(GLOB_RECURSE KERNEL_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/*.cu"
)

# Main library
add_library(vllm_cpp STATIC
    ${CORE_SOURCES}
    ${ENGINE_SOURCES}
    ${MODEL_EXECUTOR_SOURCES}
    ${DISTRIBUTED_SOURCES}
    ${ATTENTION_SOURCES}
    ${UTILS_SOURCES}
    ${KERNEL_SOURCES}
)

target_link_libraries(vllm_cpp
    ${TORCH_LIBRARIES}
    CUDA::cudart
    CUDA::cublas
    CUDA::curand
)

if(OpenMP_FOUND)
    target_link_libraries(vllm_cpp OpenMP::OpenMP_CXX)
endif()

# Set target properties
set_target_properties(vllm_cpp PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Install rules
install(TARGETS vllm_cpp
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp" PATTERN "*.cuh"
)

# Tests
if(VLLM_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Examples
if(VLLM_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# API Server
if(VLLM_BUILD_SERVER)
    # Find threading library
    find_package(Threads REQUIRED)
    
    add_executable(vllm_server
        ${CMAKE_CURRENT_SOURCE_DIR}/src/server/main.cpp
    )
    
    target_link_libraries(vllm_server
        vllm_cpp
        Threads::Threads
    )
    
    # Install server
    install(TARGETS vllm_server
        RUNTIME DESTINATION bin
    )
endif()

# Print configuration summary
message(STATUS "")
message(STATUS "vLLM C++ Configuration:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  FlashAttention: ${VLLM_USE_FLASH_ATTENTION}")
message(STATUS "  CUTLASS: ${VLLM_USE_CUTLASS}")
message(STATUS "  Build Tests: ${VLLM_BUILD_TESTS}")
message(STATUS "  Build Examples: ${VLLM_BUILD_EXAMPLES}")
message(STATUS "  Build Server: ${VLLM_BUILD_SERVER}")
message(STATUS "")
