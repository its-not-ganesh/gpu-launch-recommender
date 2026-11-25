#include "gpu_launch/benchmark.hpp"

namespace gpu_launch {
namespace {

__global__ void reduce_stage(std::uint64_t n, const float* input, float* output) {
    extern __shared__ float partial[];
    const auto stride = std::uint64_t(blockDim.x) * gridDim.x;
    const auto start = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    float sum = 0;
    for (auto i = start; i < n; i += stride) sum += input[i];

    partial[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset) partial[threadIdx.x] += partial[threadIdx.x + offset];
        __syncthreads();
    }
    if (threadIdx.x == 0) output[blockIdx.x] = partial[0];
}

}  // namespace

KernelFunctions reduce_functions() {
    const auto stage = query_function(reinterpret_cast<const void*>(reduce_stage));
    return {stage, stage};
}

void launch_reduce(const Launch& launch, cudaStream_t stream, std::uint64_t n,
                   const float* input, float* partials, float* output) {
    reduce_stage<<<dimensions(launch.grid), dimensions(launch.block), launch.shared, stream>>>(n, input, partials);
    cuda_check(cudaGetLastError(), "reduction first-stage launch");
    reduce_stage<<<dimensions(launch.final_grid), dimensions(launch.final_block), launch.final_shared, stream>>>(
        launch.grid.x, partials, output);
    cuda_check(cudaGetLastError(), "reduction final-stage launch");
}

}  // namespace gpu_launch
