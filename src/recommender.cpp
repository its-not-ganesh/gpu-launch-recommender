#include "gpu_launch/recommender.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <ostream>
#include <stdexcept>
#include <tuple>

namespace gpu_launch {
namespace {

auto shape_key(const Shape& shape) {
    return std::tie(shape.x, shape.y, shape.z);
}

auto launch_key(const Launch& launch) {
    return std::make_tuple(volume(launch.grid), shape_key(launch.block), shape_key(launch.grid));
}

struct Parameters {
    double occupancy_target;
    double loop_weight;
    double dispatch_weight;
};

Parameters parameters(WorkClass work) {
    switch (work) {
        case WorkClass::memory: return {0.50, 0.10, 0.18};
        case WorkClass::compute: return {0.75, 0.15, 0.05};
        case WorkClass::reduction: return {0.50, 0.10, 0.05};
    }
    throw std::logic_error("unknown workload class");
}

void add_role(std::vector<EvaluationLaunch>& result, const Candidate& candidate, LaunchRole role) {
    auto prior = std::find_if(result.begin(), result.end(), [&](const auto& entry) {
        return entry.candidate.launch == candidate.launch;
    });
    if (prior != result.end()) prior->roles |= role_bit(role);
    else result.push_back({candidate, role_bit(role)});
}

}  // namespace

std::optional<LaunchScore> score_launch(const DeviceInfo& device, const KernelSpec& spec,
                                       const Workload& workload, const Candidate& candidate) {
    const auto& p = device.properties;
    const auto& launch = candidate.launch;
    if (candidate.resident_blocks <= 0 || p.multiProcessorCount <= 0 || p.warpSize <= 0 ||
        p.maxThreadsPerMultiProcessor < p.warpSize || !legal_shape(device, spec, launch)) {
        return std::nullopt;
    }
    const auto threads = volume(launch.block);
    const auto grid = volume(launch.grid);
    const auto tiles_x = ceil_div(spec.two_dimensional ? workload.width : workload.n, launch.block.x);
    const auto tiles_y = spec.two_dimensional ? ceil_div(workload.height, launch.block.y) : 1;
    const auto tiles = checked_product(tiles_x, tiles_y);
    if (!tiles || !workload.elements()) return std::nullopt;

    const auto warps = ceil_div(threads, p.warpSize);
    const auto max_warps_per_sm = p.maxThreadsPerMultiProcessor / p.warpSize;
    const auto slots = checked_product(p.multiProcessorCount, candidate.resident_blocks);
    const auto resident_grid = std::min(grid, slots);
    const auto waves = ceil_div(grid, slots);

    LaunchScore score{};
    score.logical_tiles = tiles;
    score.grid_waves = waves;
    score.warps_per_block = warps;
    score.estimated_occupancy = static_cast<double>(checked_product(resident_grid, warps)) /
                                checked_product(p.multiProcessorCount, max_warps_per_sm);
    score.resident_occupancy = static_cast<double>(checked_product(candidate.resident_blocks, warps)) /
                               max_warps_per_sm;
    score.wave_utilization = static_cast<double>(grid) / checked_product(resident_grid, waves);
    score.tile_efficiency = static_cast<double>(workload.elements()) / checked_product(tiles, threads);
    score.coalescing_factor = spec.two_dimensional
        ? std::min(1.0, static_cast<double>(launch.block.x) / p.warpSize) : 1.0;
    score.grid_coverage_fraction = std::min(1.0, static_cast<double>(grid) / tiles);

    const auto weights = parameters(spec.work_class);
    score.occupancy_factor = std::min(1.0, score.estimated_occupancy / weights.occupancy_target);
    score.loop_penalty = weights.loop_weight * (1.0 - score.grid_coverage_fraction);
    score.dispatch_penalty = weights.dispatch_weight * score.grid_coverage_fraction;
    score.reduction_penalty = spec.reduction() ? 0.20 * grid / (4.0 * p.multiProcessorCount) : 0.0;
    score.cost = (1.0 + score.loop_penalty + score.dispatch_penalty + score.reduction_penalty) /
                 (score.occupancy_factor * (0.75 + 0.25 * score.wave_utilization) *
                  score.tile_efficiency * score.coalescing_factor);
    if (!std::isfinite(score.cost) || score.cost <= 0.0 || score.resident_occupancy > 1.0 ||
        score.wave_utilization > 1.0 || score.tile_efficiency > 1.0) return std::nullopt;
    return score;
}

Recommendation recommend_launch(const DeviceInfo& device, const KernelSpec& spec, const Workload& workload,
                                const std::vector<Candidate>& candidates) {
    std::optional<Recommendation> best;
    for (const auto& candidate : candidates) {
        const auto score = score_launch(device, spec, workload, candidate);
        if (score && (!best || score->cost < best->score.cost ||
                      (score->cost == best->score.cost &&
                       launch_key(candidate.launch) < launch_key(best->candidate.launch)))) {
            best = Recommendation{candidate, *score};
        }
    }
    if (!best) throw std::runtime_error("no analytically scorable launches for " + std::string(spec.name));
    return *best;
}

std::optional<Candidate> occupancy_baseline(const DeviceInfo& device,
                                            const std::vector<Candidate>& coverage_candidates) {
    std::optional<Candidate> best;
    std::uint64_t best_warps = 0;
    for (const auto& candidate : coverage_candidates) {
        const auto block = candidate.launch.block;
        const auto warps = checked_product(candidate.resident_blocks,
                                           ceil_div(volume(block), device.properties.warpSize));
        if (!best || warps > best_warps ||
            (warps == best_warps &&
             std::make_tuple(volume(block), shape_key(block)) <
             std::make_tuple(volume(best->launch.block), shape_key(best->launch.block)))) {
            best = candidate;
            best_warps = warps;
        }
    }
    return best;
}

std::vector<EvaluationLaunch> plan_evaluation(const DeviceInfo& device, const KernelSpec& spec,
                                             const Workload& workload, const KernelFunctions& functions,
                                             const std::vector<Candidate>& candidates,
                                             const Recommendation& recommendation) {
    std::vector<EvaluationLaunch> result;
    for (const auto& candidate : candidates) add_role(result, candidate, LaunchRole::candidate);
    add_role(result, recommendation.candidate, LaunchRole::recommended);

    std::vector<Launch> coverage;
    for (const auto block : block_shapes(spec)) coverage.push_back(coverage_launch(device, spec, workload, block));
    const auto feasible_coverage = feasible_launches(device, spec, functions, coverage);
    if (const auto baseline = occupancy_baseline(device, feasible_coverage)) {
        add_role(result, *baseline, LaunchRole::occupancy);
    }

    const std::array<Shape, 2> fixed = spec.two_dimensional ? std::array<Shape, 2>{{{16, 8}, {16, 16}}}
                                                          : std::array<Shape, 2>{{{128}, {256}}};
    for (std::size_t i = 0; i < fixed.size(); ++i) {
        const auto launch = coverage_launch(device, spec, workload, fixed[i]);
        const auto feasible = feasible_launches(device, spec, functions, {launch});
        if (!feasible.empty()) {
            add_role(result, feasible.front(), i == 0 ? LaunchRole::fixed_1 : LaunchRole::fixed_2);
        }
    }
    return result;
}

void write_recommendation(std::ostream& output, const DeviceInfo& device, const KernelSpec& spec,
                          const Workload& workload, const KernelFunctions& functions,
                          const Recommendation& recommendation) {
    const auto& p = device.properties;
    const auto& candidate = recommendation.candidate;
    const auto& launch = candidate.launch;
    const auto& score = recommendation.score;
    output << std::setprecision(17)
           << "model_revision=" << model_revision
           << "\ndevice_index=" << device.index << "\ndevice_uuid=" << device_uuid(device)
           << "\ndevice_name=" << std::quoted(p.name) << "\ncompute_capability=" << p.major << '.' << p.minor
           << "\nkernel=" << spec.name << "\nelements=" << workload.elements();
    if (spec.two_dimensional) output << "\nwidth=" << workload.width << "\nheight=" << workload.height;
    output << "\nblockDim=" << describe(launch.block) << "\ngridDim=" << describe(launch.grid)
           << "\nshared_bytes=" << launch.shared << "\nstatic_shared_bytes=" << functions.main.attributes.sharedSizeBytes
           << "\nregisters_per_thread=" << functions.main.attributes.numRegs
           << "\nregisters_per_sm=" << p.regsPerMultiprocessor << "\nshared_bytes_per_sm=" << p.sharedMemPerMultiprocessor
           << "\nsm_count=" << p.multiProcessorCount << "\nwarp_size=" << p.warpSize
           << "\nresident_blocks_per_sm=" << candidate.resident_blocks
           << "\nwarps_per_block=" << score.warps_per_block << "\nresident_occupancy=" << score.resident_occupancy
           << "\nestimated_occupancy=" << score.estimated_occupancy << "\noccupancy_factor=" << score.occupancy_factor
           << "\nscore=" << score.cost << "\nlogical_tiles=" << score.logical_tiles << "\ngrid_waves=" << score.grid_waves
           << "\nwave_utilization=" << score.wave_utilization << "\ntile_efficiency=" << score.tile_efficiency
           << "\ncoalescing_factor=" << score.coalescing_factor << "\ngrid_coverage_fraction=" << score.grid_coverage_fraction
           << "\nloop_penalty=" << score.loop_penalty << "\ndispatch_penalty=" << score.dispatch_penalty
           << "\nreduction_penalty=" << score.reduction_penalty << '\n';
    if (spec.reduction()) {
        output << "final_blockDim=" << describe(launch.final_block) << "\nfinal_gridDim=" << describe(launch.final_grid)
               << "\nfinal_shared_bytes=" << launch.final_shared
               << "\nfinal_resident_blocks_per_sm=" << candidate.final_resident_blocks << '\n';
    }
}

}  // namespace gpu_launch
