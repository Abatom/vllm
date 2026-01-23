# SPDX-License-Identifier: Apache-2.0
# Tests CMakeLists.txt

# Simple tests (always build these)
add_executable(test_layers_simple test_layers_simple.cpp)
target_link_libraries(test_layers_simple vllm_cpp)
add_test(NAME LayerTests COMMAND test_layers_simple)

add_executable(test_models_simple test_models_simple.cpp)
target_link_libraries(test_models_simple vllm_cpp)
add_test(NAME ModelTests COMMAND test_models_simple)
