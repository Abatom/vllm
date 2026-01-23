# SPDX-License-Identifier: Apache-2.0
# Examples CMakeLists.txt

# Basic inference example
add_executable(basic_inference basic_inference.cpp)
target_link_libraries(basic_inference vllm_cpp)

# Qwen2 example
add_executable(qwen2_example qwen2_example.cpp)
target_link_libraries(qwen2_example vllm_cpp)

# MiMoV2Flash example
add_executable(mimo_v2_flash_example mimo_v2_flash_example.cpp)
target_link_libraries(mimo_v2_flash_example vllm_cpp)
