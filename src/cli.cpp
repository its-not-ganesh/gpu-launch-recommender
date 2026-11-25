#include "gpu_launch/cli.hpp"

#include <charconv>
#include <limits>
#include <map>
#include <string_view>

namespace gpu_launch {
namespace {

std::uint64_t number(std::string_view text, const std::string& flag, std::uint64_t minimum,
                     std::uint64_t maximum) {
    std::uint64_t value = 0;
    if (text.empty()) throw UsageError("missing number for " + flag);
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() ||
        value < minimum || value > maximum) {
        throw UsageError("invalid value for " + flag + ": " + std::string(text));
    }
    return value;
}

Shape shape(std::string_view text, const std::string& flag, bool two_dimensional) {
    const auto comma = text.find(',');
    const auto limit = std::numeric_limits<unsigned>::max();
    if ((!two_dimensional && comma != text.npos) || (two_dimensional && comma == text.npos)) {
        throw UsageError("incorrect dimension count for " + flag);
    }
    const auto x = static_cast<unsigned>(number(text.substr(0, comma), flag, 1, limit));
    if (!two_dimensional) return {x, 1, 1};
    return {x, static_cast<unsigned>(number(text.substr(comma + 1), flag, 1, limit)), 1};
}

}  // namespace

const char* usage() {
    return "usage: gpu-launch <check|recommend|evaluate> --kernel <vadd|compute|reduce|transform> "
           "[--n N | --width W --height H] [--device I]\n"
           "  check: [--block X[,Y] --grid X[,Y]]\n"
           "  evaluate: --csv PATH --split <calibration|heldout> "
           "[--seed N --warmups N --rounds N --batch N]";
}

Options parse_options(int argc, const char* const* argv) {
    if (argc < 2) throw UsageError("missing mode");
    Options options;
    const std::string mode = argv[1];
    if (mode == "check") options.mode = Mode::check;
    else if (mode == "recommend") options.mode = Mode::recommend;
    else if (mode == "evaluate") options.mode = Mode::evaluate;
    else throw UsageError("unknown mode: " + mode);

    std::map<std::string, std::string> flags;
    for (int i = 2; i < argc; i += 2) {
        if (i + 1 >= argc) throw UsageError(std::string("missing value for ") + argv[i]);
        if (!flags.emplace(argv[i], argv[i + 1]).second) {
            throw UsageError(std::string("duplicate option: ") + argv[i]);
        }
    }
    const auto take = [&](const std::string& flag) -> std::optional<std::string> {
        const auto item = flags.find(flag);
        if (item == flags.end()) return std::nullopt;
        auto value = item->second;
        flags.erase(item);
        return value;
    };
    const auto required = [&](const std::string& flag) {
        const auto value = take(flag);
        if (!value || value->empty()) throw UsageError("required option: " + flag);
        return *value;
    };
    const auto& spec = kernel_spec(required("--kernel"));
    options.kernel = spec.id;
    if (spec.two_dimensional) {
        options.workload.width = number(required("--width"), "--width", 1, max_elements);
        options.workload.height = number(required("--height"), "--height", 1, max_elements);
        if (options.workload.elements() > max_elements) throw UsageError("W*H exceeds 2^24 elements");
    } else {
        options.workload.n = number(required("--n"), "--n", 1, max_elements);
    }
    if (const auto value = take("--device")) {
        options.device = static_cast<int>(number(*value, "--device", 0, std::numeric_limits<int>::max()));
    }
    if (options.mode == Mode::check) {
        const auto block = take("--block");
        const auto grid = take("--grid");
        if (block.has_value() != grid.has_value()) throw UsageError("--block and --grid must be paired");
        if (block) options.explicit_launch = make_launch(spec, shape(*block, "--block", spec.two_dimensional),
                                                       shape(*grid, "--grid", spec.two_dimensional));
    }
    if (options.mode == Mode::evaluate) {
        options.csv = required("--csv");
        options.split = required("--split");
        if (options.split != "calibration" && options.split != "heldout") throw UsageError("invalid --split");
        if (const auto value = take("--seed")) {
            options.workload.seed = static_cast<std::uint32_t>(
                number(*value, "--seed", 0, std::numeric_limits<std::uint32_t>::max()));
        }
        for (const auto& field : {std::pair{"--warmups", &options.warmups},
                                  std::pair{"--rounds", &options.rounds},
                                  std::pair{"--batch", &options.batch}}) {
            if (const auto value = take(field.first)) {
                *field.second = static_cast<unsigned>(number(*value, field.first, 1,
                                                               std::numeric_limits<unsigned>::max()));
            }
        }
    }
    if (!flags.empty()) throw UsageError("unexpected option: " + flags.begin()->first);
    return options;
}

}  // namespace gpu_launch
