#include "gpu_launch/cli.hpp"

#include <algorithm>
#include <limits>

namespace gpu_launch {
namespace {

constexpr KernelSpec specs[] = {
    {KernelId::vadd, "vadd", WorkClass::memory, 12, 1, false, true},
    {KernelId::compute, "compute", WorkClass::compute, 16, 256, false, true},
    {KernelId::reduce, "reduce", WorkClass::reduction, 4, 1, false, true},
    {KernelId::transform, "transform", WorkClass::memory, 12, 3, true, true},
};

bool within(Shape shape, const int (&limits)[3]) {
    return shape.x && shape.y && shape.z &&
           shape.x <= static_cast<unsigned>(limits[0]) &&
           shape.y <= static_cast<unsigned>(limits[1]) &&
           shape.z <= static_cast<unsigned>(limits[2]);
}

}  // namespace

bool Shape::operator==(const Shape& rhs) const {
    return x == rhs.x && y == rhs.y && z == rhs.z;
}

bool Launch::operator==(const Launch& rhs) const {
    return block == rhs.block && grid == rhs.grid && shared == rhs.shared &&
           final_block == rhs.final_block && final_grid == rhs.final_grid && final_shared == rhs.final_shared;
}

std::uint64_t checked_product(std::uint64_t lhs, std::uint64_t rhs) {
    if (rhs && lhs > std::numeric_limits<std::uint64_t>::max() / rhs) {
        throw std::overflow_error("unsigned 64-bit multiplication overflow");
    }
    return lhs * rhs;
}

std::uint64_t ceil_div(std::uint64_t numerator, std::uint64_t denominator) {
    if (!denominator) throw std::invalid_argument("zero divisor");
    return numerator / denominator + (numerator % denominator != 0);
}

std::uint64_t volume(Shape shape) {
    return checked_product(checked_product(shape.x, shape.y), shape.z);
}

std::string describe(Shape shape) {
    return std::to_string(shape.x) + "," + std::to_string(shape.y) + "," + std::to_string(shape.z);
}

const KernelSpec& kernel_spec(KernelId id) {
    for (const auto& spec : specs) if (spec.id == id) return spec;
    throw std::logic_error("unknown kernel ID");
}

const KernelSpec& kernel_spec(const std::string& name) {
    for (const auto& spec : specs) if (name == spec.name) return spec;
    throw UsageError("unknown kernel: " + name);
}

std::uint64_t Workload::elements() const {
    return n ? n : checked_product(width, height);
}

std::size_t KernelSpec::shared_bytes(Shape block) const {
    return reduction() ? checked_product(block.x, sizeof(float)) : 0;
}

Launch make_launch(const KernelSpec& spec, Shape block, Shape grid) {
    Launch launch{block, grid, spec.shared_bytes(block), {}, {}, 0};
    if (spec.reduction()) {
        launch.final_block = {256, 1, 1};
        launch.final_shared = spec.shared_bytes(launch.final_block);
    }
    return launch;
}

std::vector<Shape> block_shapes(const KernelSpec& spec) {
    return spec.two_dimensional
        ? std::vector<Shape>{{8,8}, {16,8}, {32,4}, {16,16}, {32,8}}
        : std::vector<Shape>{{64}, {128}, {256}, {512}};
}

Launch coverage_launch(const DeviceInfo& device, const KernelSpec& spec, const Workload& workload, Shape block) {
    const auto& p = device.properties;
    auto gx = std::min<std::uint64_t>(
        ceil_div(spec.two_dimensional ? workload.width : workload.n, block.x), std::max(0, p.maxGridSize[0]));
    const auto gy = spec.two_dimensional
        ? std::min<std::uint64_t>(ceil_div(workload.height, block.y), std::max(0, p.maxGridSize[1])) : 1;
    if (spec.reduction()) gx = std::min<std::uint64_t>(gx, 4ull * std::max(0, p.multiProcessorCount));
    return make_launch(spec, block, {static_cast<unsigned>(gx), static_cast<unsigned>(gy)});
}

std::vector<Launch> generate_launches(const DeviceInfo& device, const KernelSpec& spec,
                                      const Workload& workload) {
    const auto& p = device.properties;
    if (!workload.elements() || p.multiProcessorCount <= 0 ||
        p.maxGridSize[0] <= 0 || p.maxGridSize[1] <= 0) return {};
    const auto sms = static_cast<std::uint64_t>(p.multiProcessorCount);
    std::vector<Launch> launches;
    for (const auto block : block_shapes(spec)) {
        const auto full = coverage_launch(device, spec, workload, block);
        const std::uint64_t tx = full.grid.x;
        const std::uint64_t ty = full.grid.y;
        const auto coverage = checked_product(tx, ty);
        for (const auto budget : {sms, 2 * sms, 4 * sms, coverage}) {
            const auto gx = std::min(tx, budget);
            if (!gx) continue;
            const auto gy = std::min(ty, budget / gx);
            if (!gy) continue;
            const auto launch = make_launch(spec, block, {static_cast<unsigned>(gx), static_cast<unsigned>(gy)});
            if (std::find(launches.begin(), launches.end(), launch) == launches.end()) {
                launches.push_back(launch);
            }
        }
    }
    return launches;
}

bool legal_shape(const DeviceInfo& device, const KernelSpec& spec, const Launch& launch) {
    const auto& p = device.properties;
    if (!within(launch.block, p.maxThreadsDim) || !within(launch.grid, p.maxGridSize) ||
        launch.block.z != 1 || launch.grid.z != 1 ||
        volume(launch.block) > static_cast<unsigned>(p.maxThreadsPerBlock) ||
        launch.shared != spec.shared_bytes(launch.block)) return false;
    if (!spec.two_dimensional && (launch.block.y != 1 || launch.grid.y != 1)) return false;
    if (!spec.reduction()) return launch.final_shared == 0;
    const auto threads = launch.block.x;
    return !(threads & (threads - 1)) &&
           launch.grid.x <= 4ull * p.multiProcessorCount &&
           launch.final_block == Shape{256} && launch.final_grid == Shape{} &&
           launch.final_shared == spec.shared_bytes(launch.final_block) &&
           within(launch.final_block, p.maxThreadsDim) && within(launch.final_grid, p.maxGridSize);
}

std::vector<Candidate> feasible_launches(const DeviceInfo& device, const KernelSpec& spec,
                                         const KernelFunctions& functions, const std::vector<Launch>& launches) {
    std::vector<Candidate> result;
    for (const auto& launch : launches) {
        if (!legal_shape(device, spec, launch)) continue;
        const auto resident = resident_blocks(device, functions.main, launch.block, launch.shared);
        const auto final_resident = spec.reduction()
            ? resident_blocks(device, functions.final, launch.final_block, launch.final_shared) : 0;
        if (resident && (!spec.reduction() || final_resident)) result.push_back({launch, resident, final_resident});
    }
    return result;
}

}  // namespace gpu_launch
