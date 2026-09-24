#pragma once
#if defined(__linux__) && !defined(__ANDROID__)
#include <inferbridge/linux_capture_vulkan.h>
struct iris_context;
ibr_linux_capture_capabilities
iris_linux_capture_capabilities(iris_context *);
void iris_infer_linux_capture(
    iris_context *, const inferbridge::linux_capture::LinuxDmaBufImage &,
    uint64_t, float *);
#endif
