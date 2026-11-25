#pragma once

#include "gpu_launch/launch.hpp"

#include <optional>
#include <stdexcept>

namespace gpu_launch {

struct UsageError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

enum class Mode { check, recommend, evaluate };

struct Options {
    Mode mode = Mode::check;
    KernelId kernel = KernelId::vadd;
    Workload workload;
    int device = 0;
    std::optional<Launch> explicit_launch;
    std::string csv;
    std::string split;
    unsigned warmups = 5;
    unsigned rounds = 11;
    unsigned batch = 20;
};

constexpr std::uint64_t max_elements = 1ull << 24;
const char* usage();
Options parse_options(int argc, const char* const* argv);

}  // namespace gpu_launch
