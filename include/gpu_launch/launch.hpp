#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace gpu_launch {

struct Shape {
    unsigned x = 1;
    unsigned y = 1;
    unsigned z = 1;
    bool operator==(const Shape& rhs) const;
};

struct Launch {
    Shape block;
    Shape grid;
    std::size_t shared = 0;
    Shape final_block;
    Shape final_grid;
    std::size_t final_shared = 0;
    bool operator==(const Launch& rhs) const;
};

enum class KernelId { vadd, compute, reduce, transform };
enum class WorkClass { memory, compute, reduction };

struct KernelSpec {
    KernelId id;
    const char* name;
    WorkClass work_class;
    unsigned logical_bytes;
    unsigned flops;
    bool two_dimensional;
    bool grid_stride;
    bool reduction() const { return id == KernelId::reduce; }
    std::size_t shared_bytes(Shape block) const;
};

struct Workload {
    std::uint64_t n = 0;
    std::uint64_t width = 0;
    std::uint64_t height = 0;
    std::uint32_t seed = 3175;
    std::uint64_t elements() const;
};

struct DeviceInfo {
    int index;
    cudaDeviceProp properties;
    int driver;
    int runtime;
};

struct KernelFunction {
    const void* entry;
    cudaFuncAttributes attributes;
};

struct KernelFunctions {
    KernelFunction main;
    KernelFunction final;
};

struct Candidate {
    Launch launch;
    int resident_blocks;
    int final_resident_blocks;
};

std::uint64_t checked_product(std::uint64_t lhs, std::uint64_t rhs);
std::uint64_t ceil_div(std::uint64_t numerator, std::uint64_t denominator);
std::uint64_t volume(Shape shape);
std::string describe(Shape shape);
const KernelSpec& kernel_spec(KernelId id);
const KernelSpec& kernel_spec(const std::string& name);
Launch make_launch(const KernelSpec& spec, Shape block, Shape grid);
std::vector<Shape> block_shapes(const KernelSpec& spec);
Launch coverage_launch(const DeviceInfo& device, const KernelSpec& spec, const Workload& workload, Shape block);
std::vector<Launch> generate_launches(const DeviceInfo& device, const KernelSpec& spec,
                                      const Workload& workload);
bool legal_shape(const DeviceInfo& device, const KernelSpec& spec, const Launch& launch);
std::vector<Candidate> feasible_launches(const DeviceInfo& device, const KernelSpec& spec,
                                         const KernelFunctions& functions, const std::vector<Launch>& launches);

void cuda_check(cudaError_t status, const std::string& operation);
DeviceInfo select_device(int index);
KernelFunction query_function(const void* entry);
int resident_blocks(const DeviceInfo& device, const KernelFunction& function, Shape block,
                    std::size_t dynamic_shared);
std::string device_uuid(const DeviceInfo& device);

}  // namespace gpu_launch
