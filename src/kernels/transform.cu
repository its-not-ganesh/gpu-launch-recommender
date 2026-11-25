#include "gpu_launch/benchmark.hpp"

namespace gpu_launch {
namespace {

__global__ void transform(std::uint64_t width, std::uint64_t height, const float* input, float* output) {
    const auto stride_x = std::uint64_t(blockDim.x) * gridDim.x;
    const auto stride_y = std::uint64_t(blockDim.y) * gridDim.y;
    const auto start_x = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const auto start_y = std::uint64_t(blockIdx.y) * blockDim.y + threadIdx.y;
    for (auto y = start_y; y < height; y += stride_y) {
        const auto row = y * width;
        for (auto x = start_x; x < width; x += stride_x) {
            output[row + x] = 0.5f * input[row + x] + 0.25f * input[row + width - 1 - x];
        }
    }
}

}  // namespace

KernelFunction transform_function() {
    return query_function(reinterpret_cast<const void*>(transform));
}

void launch_transform(const Launch& launch, cudaStream_t stream, std::uint64_t width,
                      std::uint64_t height, const float* input, float* output) {
    transform<<<dimensions(launch.grid), dimensions(launch.block), launch.shared, stream>>>(
        width, height, input, output);
    cuda_check(cudaGetLastError(), "transform launch");
}

}  // namespace gpu_launch
