#pragma once

#include "external_gpu.h"
#include "iris_native.h"

#include <memory>
#include <string>

namespace iris_native {
std::shared_ptr<ExternalGpu> create_metal_external_gpu(
    iris_context* context, const std::string& cache_path);
}
