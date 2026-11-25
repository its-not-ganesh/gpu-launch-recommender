#pragma once

#include "gpu_launch/cli.hpp"
#include "gpu_launch/recommender.hpp"

namespace gpu_launch {

void evaluate_benchmark(const DeviceInfo& device, const KernelSpec& spec, const Options& options,
                        const std::vector<EvaluationLaunch>& launches, int argc, char** argv);

}  // namespace gpu_launch
