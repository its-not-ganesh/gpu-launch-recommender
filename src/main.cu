#include "gpu_launch/benchmark.hpp"
#include "gpu_launch/cli.hpp"
#include "gpu_launch/evaluate.hpp"

#include <iostream>
#include <string_view>

int main(int argc, char** argv) {
    using namespace gpu_launch;
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
        std::cout << usage() << '\n';
        return 0;
    }
    try {
        const auto options = parse_options(argc, argv);
        const auto device = select_device(options.device);
        const auto& spec = kernel_spec(options.kernel);
        const auto functions = benchmark_functions(spec.id);
        const auto launches = options.explicit_launch ? std::vector<Launch>{*options.explicit_launch}
            : generate_launches(device, spec, options.workload);
        const auto candidates = feasible_launches(device, spec, functions, launches);
        if (candidates.empty()) {
            if (options.explicit_launch) throw UsageError("invalid or infeasible explicit launch");
            throw std::runtime_error("no feasible launches for " + std::string(spec.name));
        }
        if (options.mode == Mode::check) {
            check_benchmark(device, spec, options.workload, candidates);
            return 0;
        }
        const auto recommendation = recommend_launch(device, spec, options.workload, candidates);
        if (options.mode == Mode::recommend) {
            write_recommendation(std::cout, device, spec, options.workload, functions, recommendation);
        } else {
            const auto plan = plan_evaluation(device, spec, options.workload, functions, candidates, recommendation);
            evaluate_benchmark(device, spec, options, plan, argc, argv);
        }
        return 0;
    } catch (const UsageError& error) {
        std::cerr << error.what() << '\n' << usage() << '\n';
        return 2;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
