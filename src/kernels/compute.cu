#include "gpu_launch/benchmark.hpp"

namespace gpu_launch {
namespace {

__global__ void compute(std::uint64_t n, const float* x, const float* a, const float* b, float* output) {
    const auto stride = std::uint64_t(blockDim.x) * gridDim.x;
    const auto start = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    for (auto i = start; i < n; i += stride) {
        float result = x[i];
        const float ai = a[i];
        const float bi = b[i];
        for (int step = 0; step < 128; ++step) result = fmaf(result, ai, bi);
        output[i] = result;
    }
}

}  // namespace

KernelFunction compute_function() {
    return query_function(reinterpret_cast<const void*>(compute));
}

void launch_compute(const Launch& launch, cudaStream_t stream, std::uint64_t n, const float* x,
                    const float* a, const float* b, float* output) {
    compute<<<dimensions(launch.grid), dimensions(launch.block), launch.shared, stream>>>(n, x, a, b, output);
    cuda_check(cudaGetLastError(), "compute launch");
}

}  // namespace gpu_launch
