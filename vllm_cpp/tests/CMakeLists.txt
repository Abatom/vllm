# SPDX-License-Identifier: Apache-2.0
# Tests CMakeLists.txt

# Find GTest (optional)
find_package(GTest QUIET)

if(GTest_FOUND)
    # Test with GTest
    add_executable(test_layers test_layers.cpp)
    target_link_libraries(test_layers vllm_cpp GTest::gtest GTest::gtest_main)
    add_test(NAME LayerTests COMMAND test_layers)
    
    add_executable(test_models test_models.cpp)
    target_link_libraries(test_models vllm_cpp GTest::gtest GTest::gtest_main)
    add_test(NAME ModelTests COMMAND test_models)
    
    add_executable(test_kernels test_kernels.cpp)
    target_link_libraries(test_kernels vllm_cpp GTest::gtest GTest::gtest_main)
    add_test(NAME KernelTests COMMAND test_kernels)
else()
    message(STATUS "GTest not found. Building simple tests without GTest framework.")
    
    # Simple tests without GTest
    add_executable(test_layers_simple test_layers_simple.cpp)
    target_link_libraries(test_layers_simple vllm_cpp)
    
    add_executable(test_models_simple test_models_simple.cpp)
    target_link_libraries(test_models_simple vllm_cpp)
endif()
