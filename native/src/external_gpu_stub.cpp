#include "external_gpu.h"

#include <stdexcept>

namespace iris_native {

std::shared_ptr<ExternalGpu> create_external_gpu(
    const std::string&, const std::string&, std::uint32_t) {
    throw std::runtime_error(
        "Iris external D3D12 interop is unavailable in this build");
}

ExternalGpuCapabilities probe_external_gpu(std::uint32_t) {
    return {};
}

}  // namespace iris_native
