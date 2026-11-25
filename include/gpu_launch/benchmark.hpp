#pragma once

#include "gpu_launch/launch.hpp"

#include <array>
#include <memory>
#include <type_traits>

namespace gpu_launch {

enum class ReductionFixture { seeded, all_ones, tail_impulse };

class Benchmark {
public:
    Benchmark(const DeviceInfo& device, const KernelSpec& spec, const Workload& workload);
    void set_reduction_fixture(ReductionFixture fixture);
    void validate(const Launch& launch);
    void run(const Launch& launch);
    cudaStream_t stream() const { return stream_.get(); }

private:
    struct FreeDeviceBuffer {
        void operator()(float* ptr) const noexcept;
    };
    struct DestroyStream {
        void operator()(std::remove_pointer_t<cudaStream_t>* ptr) const noexcept;
    };
    using DeviceBuffer = std::unique_ptr<float, FreeDeviceBuffer>;
    using Stream = std::unique_ptr<std::remove_pointer_t<cudaStream_t>, DestroyStream>;
    static DeviceBuffer allocate(std::uint64_t floats);

    const KernelSpec& spec_;
    Workload workload_;
    Stream stream_;
    std::array<DeviceBuffer, 3> input_;
    DeviceBuffer output_;
    DeviceBuffer partials_;
    std::vector<float> reduction_input_;
    std::vector<float> reference_;
    std::vector<float> actual_;
    std::vector<float> partial_values_;
};

KernelFunctions benchmark_functions(KernelId kernel);
void check_benchmark(const DeviceInfo& device, const KernelSpec& spec, const Workload& workload,
                     const std::vector<Candidate>& candidates);

KernelFunction vadd_function();
KernelFunction compute_function();
KernelFunctions reduce_functions();
KernelFunction transform_function();
void launch_vadd(const Launch&, cudaStream_t, std::uint64_t, const float*, const float*, float*);
void launch_compute(const Launch&, cudaStream_t, std::uint64_t, const float*, const float*, const float*, float*);
void launch_reduce(const Launch&, cudaStream_t, std::uint64_t, const float*, float*, float*);
void launch_transform(const Launch&, cudaStream_t, std::uint64_t, std::uint64_t, const float*, float*);

inline dim3 dimensions(Shape shape) {
    return dim3(shape.x, shape.y, shape.z);
}

}  // namespace gpu_launch
