#ifndef IRIS_NATIVE_H
#define IRIS_NATIVE_H

#include <stdint.h>

#if defined(_WIN32)
#  if defined(IRIS_NATIVE_BUILD)
#    define IRIS_API __declspec(dllexport)
#  else
#    define IRIS_API __declspec(dllimport)
#  endif
#else
#  define IRIS_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define IRIS_NATIVE_ABI_VERSION 3u

typedef struct iris_context iris_context;

enum {
    IRIS_OK = 0,
    IRIS_INVALID_ARGUMENT = 1,
    IRIS_MODEL_ERROR = 2,
    IRIS_RUNTIME_ERROR = 3
};

IRIS_API uint32_t iris_native_abi_version(void);
IRIS_API const char* iris_last_error(void);
IRIS_API int iris_get_transfer_counters(
    uint64_t* upload_bytes, uint64_t* download_bytes);

IRIS_API int iris_create(
    const char* snapshot_root_utf8,
    const char* empty_prompt_cache_utf8,
    iris_context** output);
IRIS_API int iris_create_vulkan(
    const char* snapshot_root_utf8,
    const char* empty_prompt_cache_utf8,
    uint32_t device_index,
    iris_context** output);
IRIS_API void iris_destroy(iris_context* context);

/*
 * RGB is interleaved float32 in [0,1]. Depth contains width*height float32
 * values in [0,1]. Noise arrays are NCHW and contain
 * 4*floor(width/8)*floor(height/8) values each.
 */
IRIS_API int iris_infer_rgb_f32_with_noise(
    iris_context* context,
    const float* rgb,
    uint32_t width,
    uint32_t height,
    const float* initial_noise,
    const float* posterior_noise,
    float* depth);

IRIS_API int iris_infer_rgb_f32(
    iris_context* context,
    const float* rgb,
    uint32_t width,
    uint32_t height,
    uint64_t seed,
    float* depth);

/*
 * InferBridge image contract. The source is tightly packed BGRA8. The first
 * three bytes are intentionally retained in BGR order, matching the Python
 * harness. Iris processes a nearest-neighbour image whose longest edge is
 * 384 pixels by default, then nearest-resizes and min/max-normalizes the
 * depth back to source dimensions. INFERBRIDGE_DIFFUSION_LONG_EDGE can select
 * a value from 256 through 1024.
 */
IRIS_API int iris_inferbridge_image_shape(
    uint32_t source_width,
    uint32_t source_height,
    uint32_t* processing_width,
    uint32_t* processing_height);

IRIS_API int iris_infer_bgra8_f32_with_noise(
    iris_context* context,
    const uint8_t* bgra,
    uint32_t width,
    uint32_t height,
    uint32_t row_stride_bytes,
    const float* initial_noise,
    const float* posterior_noise,
    float* depth);

IRIS_API int iris_infer_bgra8_f32(
    iris_context* context,
    const uint8_t* bgra,
    uint32_t width,
    uint32_t height,
    uint32_t row_stride_bytes,
    uint64_t seed,
    float* depth);

#ifdef __cplusplus
}
#endif

#endif
