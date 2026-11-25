#include "gpu_launch/cli.hpp"
#include "gpu_launch/recommender.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

using namespace gpu_launch;

__global__ void occupancy_probe(unsigned* output) {
    extern __shared__ unsigned scratch[];
    if (threadIdx.x == 0 && output) {
        scratch[0] = 1;
        output[blockIdx.x] = scratch[0];
    }
}

void require(bool condition, const char* reason) {
    if (!condition) throw std::runtime_error(reason);
}

void rejected(std::initializer_list<const char*> args) {
    try {
        parse_options(static_cast<int>(args.size()), args.begin());
    } catch (const UsageError&) {
        return;
    }
    throw std::runtime_error("invalid arguments accepted");
}

void check_model(const DeviceInfo& device, const KernelFunction& probe) {
    auto synthetic = device;
    synthetic.properties.multiProcessorCount = 2;
    const auto& vadd = kernel_spec(KernelId::vadd);
    const Workload irregular{129};
    const Candidate small{make_launch(vadd, {64}, {1}), 16, 0};
    const auto score = score_launch(synthetic, vadd, irregular, small);
    const double expected = (1 + 0.10 * 2 / 3 + 0.18 / 3) / (0.0625 * (129.0 / 192));
    require(score && std::abs(score->cost - expected) < 1e-12, "analytical formula differs from design");
    require(score->logical_tiles == 3 && score->grid_waves == 1 &&
            std::abs(score->tile_efficiency - 129.0 / 192) < 1e-12, "irregular tile accounting failed");
    auto bad = small;
    bad.resident_blocks = 0;
    require(!score_launch(synthetic, vadd, irregular, bad), "zero residency was scored");
    bad.resident_blocks = 99;
    require(!score_launch(synthetic, vadd, irregular, bad), "impossible occupancy was scored");
    const Candidate wave_tail{make_launch(vadd, {64}, {65}), 1, 0};
    const auto tail = score_launch(synthetic, vadd, {4097}, wave_tail);
    require(tail && tail->grid_waves == 33 && std::abs(tail->wave_utilization - 65.0 / 66) < 1e-12,
            "partial wave accounting failed");

    const auto& transform = kernel_spec(KernelId::transform);
    const Workload matrix{0, 128, 8};
    const Candidate narrow{make_launch(transform, {8, 8}, {16, 1}), 16, 0};
    const Candidate wide{make_launch(transform, {32, 4}, {4, 2}), 4, 0};
    const std::vector<Candidate> shapes{narrow, wide};
    const auto occupancy = occupancy_baseline(synthetic, shapes);
    require(occupancy && occupancy->launch == narrow.launch, "baseline did not maximize resident warps");
    require(recommend_launch(synthetic, transform, matrix, shapes).candidate.launch == wide.launch,
            "analytical ranker did not account for the 2D coalescing factor");
    const Candidate narrow_tied{make_launch(transform, {8, 8}, {16, 1}), 8, 0};
    require(occupancy_baseline(synthetic, {wide, narrow_tied})->launch == narrow_tied.launch,
            "occupancy tie did not prefer fewer threads");
    const Candidate grid_x{make_launch(transform, {32, 4}, {2, 1}), 4, 0};
    const Candidate grid_y{make_launch(transform, {32, 4}, {1, 2}), 4, 0};
    require(recommend_launch(synthetic, transform, matrix, {grid_x, grid_y}).candidate.launch == grid_y.launch,
            "analytical grid tie-break is not lexicographic");

    const Workload workload{1537};
    const KernelFunctions functions{probe, {}};
    const auto seed = feasible_launches(device, vadd, functions, {coverage_launch(device, vadd, workload, {64})});
    require(seed.size() == 1, "model probe was infeasible");
    const auto selected = recommend_launch(device, vadd, workload, seed);
    const auto plan = plan_evaluation(device, vadd, workload, functions, seed, selected);
    require(plan.front().roles & role_bit(LaunchRole::recommended), "recommended role did not merge with candidate");
    unsigned combined = 0;
    for (const auto& entry : plan) {
        require(std::count_if(plan.begin(), plan.end(), [&](const auto& other) {
                    return entry.candidate.launch == other.candidate.launch;
                }) == 1, "evaluation plan duplicated a launch");
        if (entry.roles & role_bit(LaunchRole::fixed_1)) require(entry.candidate.launch.block == Shape{128}, "wrong fixed 1");
        if (entry.roles & role_bit(LaunchRole::fixed_2)) require(entry.candidate.launch.block == Shape{256}, "wrong fixed 2");
        combined |= entry.roles;
    }
    require(combined == 31 && plan.size() >= 3, "evaluation did not add omitted feasible baselines");
    auto constrained = device;
    constrained.properties.maxThreadsPerBlock = 64;
    const auto limited = plan_evaluation(constrained, vadd, workload, functions, seed, selected);
    require(limited.size() == 1 &&
            !(limited.front().roles & (role_bit(LaunchRole::fixed_1) | role_bit(LaunchRole::fixed_2))),
            "infeasible fixed launch included");
    std::cout << "ANALYTICAL_AND_BASELINE_PASS\n";
}

int main() {
    try {
        const auto limit = std::numeric_limits<std::uint64_t>::max();
        require(ceil_div(limit, 2) == limit / 2 + 1, "ceiling division overflowed");
        bool overflow = false;
        try { checked_product(limit, 2); } catch (const std::overflow_error&) { overflow = true; }
        require(overflow, "overflow not rejected");
        rejected({"test", "unsupported"});
        rejected({"test", "check", "--kernel", "vadd", "--n", "0"});
        rejected({"test", "check", "--kernel", "vadd", "--n", "18446744073709551616"});
        rejected({"test", "check", "--kernel", "vadd", "--n", "1", "--block", "32"});
        rejected({"test", "check", "--kernel", "transform", "--width", "16777216", "--height", "16777216"});
        rejected({"test", "evaluate", "--kernel", "reduce", "--n", "17", "--csv", "test.csv",
                  "--split", "calibration", "--rounds", "0"});
        rejected({"test", "recommend", "--kernel", "vadd", "--n", "1", "--csv", "test.csv"});

        const auto device = select_device(0);
        const auto probe = query_function(reinterpret_cast<const void*>(occupancy_probe));
        require(resident_blocks(device, probe, {128}, 512) > 0, "valid launch is infeasible");
        require(resident_blocks(device, probe, {128}, device.properties.sharedMemPerBlock + 1) == 0,
                "excess shared memory accepted");
        auto restricted = probe;
        restricted.attributes.maxThreadsPerBlock = 64;
        require(resident_blocks(device, restricted, {128}, 0) == 0, "function thread cap ignored");

        const auto& transform = kernel_spec(KernelId::transform);
        auto constrained = device;
        constrained.properties.maxGridSize[0] = 7;
        constrained.properties.maxGridSize[1] = 3;
        const auto tiles = generate_launches(constrained, transform, {0, 4097, 257});
        require(tiles.size() == 5, "clipped duplicate launches not removed");
        for (const auto& launch : tiles) {
            require(legal_shape(constrained, transform, launch), "clipped candidate is illegal");
            require(launch.grid == Shape{7, 3}, "axis clipping is incorrect");
            require(std::count(tiles.begin(), tiles.end(), launch) == 1, "duplicate 2D candidate");
        }
        require(!legal_shape(constrained, transform, make_launch(transform, {8, 8}, {8, 3})),
                "grid axis limit ignored");
        require(!legal_shape(constrained, transform, make_launch(transform, {1024, 2}, {1, 1})),
                "thread volume limit ignored");

        const auto& reduce = kernel_spec(KernelId::reduce);
        const auto launches = generate_launches(device, reduce, {16777216});
        require(launches.size() == 12, "unexpected reduction budget or duplicate");
        for (const auto& launch : launches) {
            require(launch.grid.x <= 4u * device.properties.multiProcessorCount, "reduction grid cap ignored");
            require(launch.shared == 4 * launch.block.x && launch.final_shared == 1024,
                    "reduction shared bytes incorrect");
            require(std::count(launches.begin(), launches.end(), launch) == 1, "duplicate reduction candidate");
        }
        require(feasible_launches(device, reduce, {probe, probe}, launches).size() == launches.size(),
                "reduction stage feasibility failed");
        require(!legal_shape(device, reduce, make_launch(reduce, {96}, {1})), "non-power-of-two reduction accepted");
        check_model(device, probe);
        std::cout << "INFRASTRUCTURE_PASS device=" << device.properties.name << " sm="
                  << device.properties.multiProcessorCount << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
