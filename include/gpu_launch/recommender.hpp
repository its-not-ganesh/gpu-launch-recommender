#pragma once

#include "gpu_launch/launch.hpp"

#include <iosfwd>
#include <optional>

namespace gpu_launch {

inline constexpr const char* model_revision = "analytical-v1";

struct LaunchScore {
    double cost;
    double estimated_occupancy;
    double resident_occupancy;
    double occupancy_factor;
    double wave_utilization;
    double tile_efficiency;
    double coalescing_factor;
    double grid_coverage_fraction;
    double loop_penalty;
    double dispatch_penalty;
    double reduction_penalty;
    std::uint64_t logical_tiles;
    std::uint64_t grid_waves;
    std::uint64_t warps_per_block;
};

struct Recommendation {
    Candidate candidate;
    LaunchScore score;
};

enum class LaunchRole : unsigned {
    candidate = 1, fixed_1 = 2, fixed_2 = 4, occupancy = 8, recommended = 16, empirical_best = 32
};

constexpr unsigned role_bit(LaunchRole role) { return static_cast<unsigned>(role); }

struct EvaluationLaunch {
    Candidate candidate;
    unsigned roles;
};

std::optional<LaunchScore> score_launch(const DeviceInfo& device, const KernelSpec& spec,
                                       const Workload& workload, const Candidate& candidate);
Recommendation recommend_launch(const DeviceInfo& device, const KernelSpec& spec, const Workload& workload,
                                const std::vector<Candidate>& candidates);
std::optional<Candidate> occupancy_baseline(const DeviceInfo& device,
                                            const std::vector<Candidate>& coverage_candidates);
std::vector<EvaluationLaunch> plan_evaluation(const DeviceInfo& device, const KernelSpec& spec,
                                             const Workload& workload, const KernelFunctions& functions,
                                             const std::vector<Candidate>& candidates,
                                             const Recommendation& recommendation);
void write_recommendation(std::ostream& output, const DeviceInfo& device, const KernelSpec& spec,
                          const Workload& workload, const KernelFunctions& functions,
                          const Recommendation& recommendation);

}  // namespace gpu_launch
