#include "gpu_launch/benchmark.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>

namespace gpu_launch {
namespace {

float dyadic(std::mt19937& random) {
    return (static_cast<int>(random() % 513) - 256) / 256.0f;
}

std::size_t bytes(std::uint64_t count) {
    const auto size = checked_product(count, sizeof(float));
    if (size > std::numeric_limits<std::size_t>::max()) throw std::overflow_error("allocation size exceeds size_t");
    return static_cast<std::size_t>(size);
}

std::string context(const KernelSpec& spec, const Workload& workload, const Launch& launch) {
    const auto size = spec.two_dimensional ? std::to_string(workload.width) + "x" + std::to_string(workload.height)
                                           : std::to_string(workload.n);
    return std::string(spec.name) + " size=" + size + " block=" + describe(launch.block) +
           " grid=" + describe(launch.grid);
}

}  // namespace

void Benchmark::FreeDeviceBuffer::operator()(float* ptr) const noexcept {
    cudaFree(ptr);
}

void Benchmark::DestroyStream::operator()(std::remove_pointer_t<cudaStream_t>* ptr) const noexcept {
    cudaStreamDestroy(ptr);
}

Benchmark::DeviceBuffer Benchmark::allocate(std::uint64_t floats) {
    float* data = nullptr;
    cuda_check(cudaMalloc(&data, bytes(floats)), "cudaMalloc");
    return DeviceBuffer(data);
}

Benchmark::Benchmark(const DeviceInfo& device, const KernelSpec& spec, const Workload& workload)
    : spec_(spec), workload_(workload) {
    const auto n = workload.elements();
    const auto inputs = spec.id == KernelId::compute ? 3u : spec.id == KernelId::vadd ? 2u : 1u;
    const auto outputs = spec.reduction() ? 1 : n;
    const auto max_partials = spec.reduction() ? 4ull * device.properties.multiProcessorCount : 0;
    const auto required = bytes(checked_product(n, inputs) + outputs + max_partials);
    std::size_t available = 0;
    std::size_t total = 0;
    cuda_check(cudaMemGetInfo(&available, &total), "cudaMemGetInfo");
    if (required > available) throw std::runtime_error("workload exceeds available GPU memory");

    cudaStream_t created_stream = nullptr;
    cuda_check(cudaStreamCreateWithFlags(&created_stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
    stream_.reset(created_stream);
    for (unsigned i = 0; i < inputs; ++i) input_[i] = allocate(n);
    output_ = allocate(outputs);
    reference_.resize(outputs);
    actual_.resize(outputs);
    if (spec.reduction()) {
        partials_ = allocate(max_partials);
        partial_values_.resize(max_partials);
        reduction_input_.resize(n);
        set_reduction_fixture(ReductionFixture::seeded);
        return;
    }

    std::mt19937 random(workload.seed);
    std::array<std::vector<float>, 3> host;
    for (unsigned i = 0; i < inputs; ++i) host[i].resize(n);
    for (std::uint64_t i = 0; i < n; ++i) {
        host[0][i] = dyadic(random);
        if (spec.id == KernelId::vadd) {
            host[1][i] = dyadic(random);
            reference_[i] = host[0][i] + host[1][i];
        } else if (spec.id == KernelId::compute) {
            host[1][i] = 0.875f + (random() % 512) / 8192.0f;
            host[2][i] = dyadic(random) / 16.0f;
            float result = host[0][i];
            for (int step = 0; step < 128; ++step) result = std::fma(result, host[1][i], host[2][i]);
            reference_[i] = result;
        }
    }
    if (spec.id == KernelId::transform) {
        for (std::uint64_t row = 0; row < n; row += workload.width) {
            for (std::uint64_t x = 0; x < workload.width; ++x) {
                reference_[row + x] = 0.5f * host[0][row + x] + 0.25f * host[0][row + workload.width - 1 - x];
            }
        }
    }
    for (unsigned i = 0; i < inputs; ++i) {
        cuda_check(cudaMemcpyAsync(input_[i].get(), host[i].data(), bytes(n), cudaMemcpyHostToDevice, stream()),
                   "upload input");
    }
    cuda_check(cudaStreamSynchronize(stream()), "synchronize input upload");
}

void Benchmark::set_reduction_fixture(ReductionFixture fixture) {
    if (!spec_.reduction()) throw std::logic_error("reduction fixture on a non-reduction workload");
    std::mt19937 random(workload_.seed);
    std::uint64_t exact_sum = 0;
    for (auto& value : reduction_input_) {
        const bool one = fixture == ReductionFixture::all_ones ||
                         (fixture == ReductionFixture::seeded && random() % 2 == 1);
        value = one ? 1.0f : 0.0f;
        exact_sum += one;
    }
    if (reduction_input_.back() == 0.0f) ++exact_sum;
    reduction_input_.back() = 1.0f;
    reference_[0] = static_cast<float>(exact_sum);
    cuda_check(cudaMemcpyAsync(input_[0].get(), reduction_input_.data(), bytes(workload_.n),
                              cudaMemcpyHostToDevice, stream()),
               "upload reduction input");
    cuda_check(cudaStreamSynchronize(stream()), "synchronize reduction upload");
}

void Benchmark::run(const Launch& launch) {
    switch (spec_.id) {
        case KernelId::vadd:
            launch_vadd(launch, stream(), workload_.n, input_[0].get(), input_[1].get(), output_.get());
            break;
        case KernelId::compute:
            launch_compute(launch, stream(), workload_.n, input_[0].get(), input_[1].get(), input_[2].get(), output_.get());
            break;
        case KernelId::reduce:
            launch_reduce(launch, stream(), workload_.n, input_[0].get(), partials_.get(), output_.get());
            break;
        case KernelId::transform:
            launch_transform(launch, stream(), workload_.width, workload_.height, input_[0].get(), output_.get());
            break;
    }
}

void Benchmark::validate(const Launch& launch) {
    try {
        const auto output_bytes = bytes(actual_.size());
        cuda_check(cudaMemsetAsync(output_.get(), 0xff, output_bytes, stream()), "set output sentinel");
        if (spec_.reduction()) {
            cuda_check(cudaMemsetAsync(partials_.get(), 0xff, bytes(launch.grid.x), stream()), "set partial sentinels");
        }
        run(launch);
        cuda_check(cudaMemcpyAsync(actual_.data(), output_.get(), output_bytes, cudaMemcpyDeviceToHost, stream()),
                   "copy output for correctness");
        if (spec_.reduction()) {
            cuda_check(cudaMemcpyAsync(partial_values_.data(), partials_.get(), bytes(launch.grid.x),
                                       cudaMemcpyDeviceToHost, stream()), "copy reduction partials");
        }
        cuda_check(cudaStreamSynchronize(stream()), "synchronize correctness check");

        for (std::size_t i = 0; i < actual_.size(); ++i) {
            const auto expected = reference_[i];
            const auto tolerance = spec_.id == KernelId::compute ? 1e-6f + 1e-5f * std::abs(expected) : 0.0f;
            if (!std::isfinite(actual_[i]) || std::abs(actual_[i] - expected) > tolerance) {
                throw std::runtime_error("output mismatch at " + std::to_string(i) + ", expected " +
                                         std::to_string(expected) + ", got " + std::to_string(actual_[i]));
            }
        }
        if (!spec_.reduction()) return;

        const std::uint64_t threads = launch.block.x;
        const auto stride = threads * launch.grid.x;
        for (unsigned block = 0; block < launch.grid.x; ++block) {
            std::uint64_t expected = 0;
            for (auto start = block * threads; start < workload_.n; start += stride) {
                for (auto i = start; i < std::min(start + threads, workload_.n); ++i) {
                    expected += reduction_input_[i] == 1.0f;
                }
            }
            if (partial_values_[block] != static_cast<float>(expected)) {
                throw std::runtime_error("partial reduction mismatch at block " + std::to_string(block));
            }
        }
    } catch (const std::exception& error) {
        cudaStreamSynchronize(stream());
        throw std::runtime_error(context(spec_, workload_, launch) + ": " + error.what());
    }
}

KernelFunctions benchmark_functions(KernelId kernel) {
    switch (kernel) {
        case KernelId::vadd: return {vadd_function(), {}};
        case KernelId::compute: return {compute_function(), {}};
        case KernelId::reduce: return reduce_functions();
        case KernelId::transform: return {transform_function(), {}};
    }
    throw std::logic_error("unknown benchmark entry point");
}

void check_benchmark(const DeviceInfo& device, const KernelSpec& spec, const Workload& workload,
                     const std::vector<Candidate>& candidates) {
    Benchmark benchmark(device, spec, workload);
    for (const auto& candidate : candidates) benchmark.validate(candidate.launch);
    if (spec.reduction()) {
        for (const auto fixture : {ReductionFixture::all_ones, ReductionFixture::tail_impulse}) {
            benchmark.set_reduction_fixture(fixture);
            for (const auto& candidate : candidates) benchmark.validate(candidate.launch);
        }
    }
    std::cout << "CHECK_PASS kernel=" << spec.name << " elements=" << workload.elements()
              << " configurations=" << candidates.size() << " fixtures=" << (spec.reduction() ? 3 : 1)
              << " device=\"" << device.properties.name << "\"\n";
}

}  // namespace gpu_launch
