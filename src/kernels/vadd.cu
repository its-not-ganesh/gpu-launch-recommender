#include "gpu_launch/benchmark.hpp"

namespace gpu_launch {
namespace {

__global__ void vadd(std::uint64_t n, const float* x, const float* y, float* z) {
    const auto stride = std::uint64_t(blockDim.x) * gridDim.x;
    const auto start = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    for (auto i = start; i < n; i += stride) z[i] = x[i] + y[i];
}

}  // namespace

KernelFunction vadd_function() {
    return query_function(reinterpret_cast<const void*>(vadd));
}

void launch_vadd(const Launch& launch, cudaStream_t stream, std::uint64_t n,
                 const float* x, const float* y, float* z) {
    vadd<<<dimensions(launch.grid), dimensions(launch.block), launch.shared, stream>>>(n, x, y, z);
    cuda_check(cudaGetLastError(), "vadd launch");
}

}  // namespace gpu_launch
