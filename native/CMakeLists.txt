cmake_minimum_required(VERSION 3.21)
project(iris_native LANGUAGES C CXX)
set(INFERBRIDGE_NATIVE_SUPPORT_DIR "" CACHE PATH
    "InferBridge native harness SDK directory")
if(NOT EXISTS "${INFERBRIDGE_NATIVE_SUPPORT_DIR}/InferBridgeNativeHarnessSupport.cmake")
    message(FATAL_ERROR
        "INFERBRIDGE_NATIVE_SUPPORT_DIR must identify the InferBridge native harness SDK")
endif()
include("${INFERBRIDGE_NATIVE_SUPPORT_DIR}/InferBridgeNativeHarnessSupport.cmake")
link_libraries(InferBridge::NativeHarnessSupport)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    set(
        CMAKE_MSVC_RUNTIME_LIBRARY
        "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

option(IRIS_WITH_VULKAN
    "Build the dependency-free Vulkan executor" ON)
option(IRIS_WITH_METAL
    "Build the native Metal executor" ${APPLE})

if(IRIS_WITH_METAL)
    if(NOT APPLE)
        message(FATAL_ERROR "IRIS_WITH_METAL requires Apple platforms")
    endif()
    enable_language(OBJCXX)
endif()

if(IRIS_WITH_VULKAN)
    find_package(Vulkan REQUIRED COMPONENTS glslc)
    set(IRIS_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
    file(MAKE_DIRECTORY "${IRIS_GENERATED_DIR}")
    function(iris_compile_shader NAME)
        set(SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/${NAME}.glsl")
        set(SPIRV "${IRIS_GENERATED_DIR}/${NAME}.spv")
        set(HEADER "${IRIS_GENERATED_DIR}/${NAME}_spv.h")
        add_custom_command(
            OUTPUT "${SPIRV}"
            COMMAND Vulkan::glslc -O -fshader-stage=compute
                --target-env=vulkan1.3 --target-spv=spv1.6 -Werror
                "-I${CMAKE_CURRENT_SOURCE_DIR}/src/shaders"
                -o "${SPIRV}" "${SOURCE}"
            DEPENDS "${SOURCE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/indexing.h"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/conv2d_common.glsl"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/linear_common.glsl"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/project_tokens_common.glsl"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/conv_transpose_nonoverlap_common.glsl"
            VERBATIM)
        add_custom_command(
            OUTPUT "${HEADER}"
            COMMAND "${CMAKE_COMMAND}" "-DINPUT=${SPIRV}"
                "-DOUTPUT=${HEADER}" "-DVARIABLE=iris_${NAME}_spv"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_binary.cmake"
            DEPENDS "${SPIRV}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_binary.cmake"
            VERBATIM)
        set(IRIS_SHADER_HEADERS
            ${IRIS_SHADER_HEADERS} "${HEADER}" PARENT_SCOPE)
    endfunction()
    foreach(SHADER
            add_scaled add add_position bilinear_align_true
            bilinear_align_true_image bmm bmm_score_half bmm_value_half
            conv2d conv2d_pointwise_gemm conv2d8 conv2d_half conv2d8_half conv2d_tiled conv2d8_tiled
            conv2d8_stride2_tiled conv2d8_stride2_tiled_half
            conv2d8_tiled16x8
            conv2d8_tiled16x8_half
            conv2d_winograd
            conv_transpose_nonoverlap conv_transpose_nonoverlap_half
            elementwise gelu layer_norm linear linear16 linear_half
            linear16_half linear_vec8 linear_vec8_half prepare_tokens position_bicubic project_tokens
            project_tokens_half relu softmax_lastdim softmax_lastdim_half
            group_norm group_norm_silu silu nearest concatenate add_channel nchw_tokens
            geglu attention_scores attention_values preprocess_rgb
            preprocess_texture seeded_noise posterior_sample scale_values
            depth_output depth_range normalize_depth depth_to_image)
        iris_compile_shader(${SHADER})
    endforeach()
    foreach(SHARED_SHADER quantize_rows_int8 linear_int8_tiled)
        inferbridge_compile_native_support_shader(
            NAME "${SHARED_SHADER}" PREFIX iris
            GENERATED_DIR "${IRIS_GENERATED_DIR}"
            EMBED_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_binary.cmake"
            OUT_VAR IRIS_SHADER_HEADERS)
    endforeach()
endif()

add_library(iris_model STATIC
    src/model_bundle.cpp
    src/prompt_cache.cpp
    src/safetensors.cpp
    src/tensor_cpu.cpp
    src/transformer_cpu.cpp
    src/unet_cpu.cpp
    src/vae_cpu.cpp
)
if(IRIS_WITH_VULKAN)
    target_sources(iris_model PRIVATE
        src/vulkan.cpp src/operators.cpp src/gpu_model.cpp
        src/iris_gpu.cpp
        ${IRIS_SHADER_HEADERS})
    target_include_directories(
        iris_model PRIVATE "${IRIS_GENERATED_DIR}")
    target_link_libraries(iris_model PUBLIC Vulkan::Vulkan)
    target_compile_definitions(
        iris_model PUBLIC IRIS_WITH_VULKAN=1)
endif()
target_include_directories(iris_model PUBLIC src)
set_target_properties(iris_model PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(iris_native SHARED
    src/iris.cpp
    src/inferbridge_harness.cpp)
if(IRIS_WITH_VULKAN)
    target_sources(iris_native PRIVATE src/external_gpu.cpp)
else()
    target_sources(iris_native PRIVATE src/external_gpu_stub.cpp)
endif()
if(IRIS_WITH_METAL)
    target_sources(iris_native PRIVATE src/metal_executor.mm)
    target_link_libraries(iris_native PRIVATE
        "-framework Foundation"
        "-framework Metal"
        "-framework MetalPerformanceShadersGraph")
    target_compile_definitions(iris_native PRIVATE IRIS_WITH_METAL=1)
    set_source_files_properties(src/metal_executor.mm PROPERTIES
        COMPILE_OPTIONS "-fobjc-arc")
endif()
target_compile_definitions(
    iris_native PRIVATE
        IRIS_NATIVE_BUILD
        INFERBRIDGE_HARNESS_BUILDING_LIBRARY)
target_include_directories(iris_native PUBLIC include)
target_link_libraries(iris_native PRIVATE iris_model)
if(WIN32 AND IRIS_WITH_VULKAN)
    target_link_libraries(iris_native PRIVATE d3d12 dxgi)
endif()
set_target_properties(
    iris_native PROPERTIES
    OUTPUT_NAME "iris_native"
    WINDOWS_EXPORT_ALL_SYMBOLS OFF)

add_executable(iris_model_probe tools/model_probe.cpp)
target_link_libraries(iris_model_probe PRIVATE iris_model)

add_executable(iris_vae_probe tools/vae_probe.cpp)
target_link_libraries(iris_vae_probe PRIVATE iris_model)

add_executable(iris_unet_probe tools/unet_probe.cpp)
target_link_libraries(iris_unet_probe PRIVATE iris_model)

add_executable(iris_full_graph_probe tools/full_graph_probe.cpp)
target_include_directories(iris_full_graph_probe PRIVATE include)
target_link_libraries(iris_full_graph_probe PRIVATE iris_native)

add_executable(iris_size_probe tools/size_probe.cpp)
target_include_directories(iris_size_probe PRIVATE include)
target_link_libraries(iris_size_probe PRIVATE iris_native)
if(IRIS_WITH_VULKAN)
    add_executable(
        iris_gpu_component_probe tools/gpu_component_probe.cpp)
    target_link_libraries(
        iris_gpu_component_probe PRIVATE iris_model)
    target_include_directories(
        iris_gpu_component_probe PRIVATE src)
endif()

add_executable(iris_c_abi_smoke tests/c_abi_smoke.c)
target_include_directories(iris_c_abi_smoke PRIVATE include)
target_link_libraries(iris_c_abi_smoke PRIVATE iris_native)
add_executable(iris_harness_abi_smoke tests/harness_abi_smoke.c)
target_include_directories(iris_harness_abi_smoke PRIVATE include)
target_link_libraries(iris_harness_abi_smoke PRIVATE iris_native)
add_executable(iris_harness_full_graph tests/harness_full_graph.cpp)
target_include_directories(iris_harness_full_graph PRIVATE include)
target_link_libraries(iris_harness_full_graph PRIVATE iris_native)
if(WIN32 AND IRIS_WITH_VULKAN)
    add_executable(
        iris_d3d12_abi2_full_graph tests/d3d12_abi2_full_graph.cpp)
    target_include_directories(
        iris_d3d12_abi2_full_graph PRIVATE include)
    target_link_libraries(
        iris_d3d12_abi2_full_graph PRIVATE iris_native d3d12 dxgi)
endif()

enable_testing()
add_test(NAME iris_model_probe_usage COMMAND iris_model_probe)
set_tests_properties(iris_model_probe_usage PROPERTIES WILL_FAIL TRUE)
add_test(NAME iris_c_abi_smoke COMMAND iris_c_abi_smoke)
add_test(NAME iris_harness_abi_smoke COMMAND iris_harness_abi_smoke)
add_test(NAME iris_harness_full_graph COMMAND iris_harness_full_graph)
set_tests_properties(
    iris_harness_full_graph PROPERTIES SKIP_RETURN_CODE 77)
