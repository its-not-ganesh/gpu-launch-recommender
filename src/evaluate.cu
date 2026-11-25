#include "gpu_launch/evaluate.hpp"
#include "gpu_launch/benchmark.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <utility>
#include <unistd.h>

namespace gpu_launch {
namespace {

constexpr char csv_header[] =
    "schema_version,run_id,split,kernel,n,width,height,seed,launch_id,roles,block_x,block_y,block_z,"
    "grid_x,grid_y,grid_z,shared_bytes,final_block_x,final_grid_x,final_shared_bytes,validation_status,"
    "error_code,round,order,warmups,batch_iters,elapsed_ms,ms_per_operation\n";

constexpr std::pair<LaunchRole, const char*> roles[] = {
    {LaunchRole::candidate, "candidate"}, {LaunchRole::fixed_1, "fixed_1"},
    {LaunchRole::fixed_2, "fixed_2"}, {LaunchRole::occupancy, "occupancy"},
    {LaunchRole::recommended, "recommended"}, {LaunchRole::empirical_best, "empirical_best"},
};

std::string role_labels(unsigned mask) {
    std::string labels;
    for (const auto& [role, name] : roles) {
        if (!(mask & role_bit(role))) continue;
        if (!labels.empty()) labels += '|';
        labels += name;
    }
    return labels;
}

using File = std::unique_ptr<std::FILE, decltype(&std::fclose)>;

File open_fresh(const std::filesystem::path& path) {
    auto* file = std::fopen(path.c_str(), "wx");
    if (!file) throw std::runtime_error("create " + path.string() + ": " + std::strerror(errno));
    return File(file, &std::fclose);
}

void write_file(std::FILE* file, const std::string& content) {
    if (std::fwrite(content.data(), 1, content.size(), file) != content.size() || std::fflush(file) != 0) {
        throw std::runtime_error("write measurement evidence: " + std::string(std::strerror(errno)));
    }
}

std::string capture(const std::string& command) {
    auto* pipe = popen(command.c_str(), "r");
    if (!pipe) return "unavailable";
    std::string result;
    char buffer[256];
    while (std::fgets(buffer, sizeof buffer, pipe)) result += buffer;
    if (pclose(pipe) != 0) return "unavailable";
    while (!result.empty() && (result.back() == '\n' || result.back() == '\r')) result.pop_back();
    std::replace(result.begin(), result.end(), '\n', '|');
    return result;
}

std::string gpu_snapshot(const std::string& uuid) {
    return capture("nvidia-smi --id=" + uuid +
                   " --query-gpu=driver_version,utilization.gpu,memory.used,temperature.gpu,clocks.current.graphics"
                   " --format=csv,noheader,nounits 2>/dev/null");
}

std::string utc_timestamp() {
    const auto now = std::time(nullptr);
    std::tm utc{};
    if (!gmtime_r(&now, &utc)) throw std::runtime_error("read UTC clock");
    std::ostringstream timestamp;
    timestamp << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
    return timestamp.str();
}

std::string csv_field(const std::string& value) {
    std::string quoted = "\"";
    for (const char ch : value) {
        if (ch == '"') quoted += '"';
        quoted += ch;
    }
    return quoted + '"';
}

class RunOutput {
public:
    explicit RunOutput(const std::string& csv_path)
        : csv_path_(csv_path), metadata_path_(csv_path), run_id_(csv_path_.stem().string()),
          csv_(open_fresh(csv_path_)), metadata_(nullptr, &std::fclose) {
        metadata_path_.replace_extension(".meta.txt");
        try {
            metadata_ = open_fresh(metadata_path_);
        } catch (...) {
            csv_.reset();
            std::error_code ignored;
            std::filesystem::remove(csv_path_, ignored);
            throw;
        }
        csv(csv_header);
    }

    void csv(const std::string& text) { write_file(csv_.get(), text); }
    void metadata(const std::string& text) { write_file(metadata_.get(), text); }
    const std::string& run_id() const { return run_id_; }
    const std::filesystem::path& metadata_path() const { return metadata_path_; }

private:
    std::filesystem::path csv_path_;
    std::filesystem::path metadata_path_;
    std::string run_id_;
    File csv_;
    File metadata_;
};

class Event {
public:
    Event() { cuda_check(cudaEventCreate(&event_), "cudaEventCreate"); }
    ~Event() { cudaEventDestroy(event_); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;

    void record(cudaStream_t stream) const {
        cuda_check(cudaEventRecord(event_, stream), "cudaEventRecord");
    }
    double elapsed_since(const Event& start) const {
        cuda_check(cudaEventSynchronize(event_), "cudaEventSynchronize");
        float milliseconds = 0;
        cuda_check(cudaEventElapsedTime(&milliseconds, start.event_, event_), "cudaEventElapsedTime");
        if (!std::isfinite(milliseconds) || milliseconds <= 0) {
            throw std::runtime_error("CUDA event interval is nonpositive or nonfinite");
        }
        return milliseconds;
    }
private:
    cudaEvent_t event_ = nullptr;
};

struct Sample {
    std::size_t index;
    unsigned round;
    std::size_t order;
    double milliseconds;
};

struct EmpiricalBest {
    std::size_t index;
    double median_ms;
};

EmpiricalBest empirical_best(const std::vector<Sample>& samples, std::size_t count, unsigned batch) {
    std::vector<std::vector<double>> by_launch(count);
    for (const auto& sample : samples) by_launch[sample.index].push_back(sample.milliseconds / batch);
    std::optional<EmpiricalBest> best;
    for (std::size_t index = 0; index < count; ++index) {
        auto& values = by_launch[index];
        if (values.empty()) continue;
        std::sort(values.begin(), values.end());
        const auto middle = values.size() / 2;
        const double median = values.size() % 2 ? values[middle] : (values[middle - 1] + values[middle]) / 2.0;
        if (!best || median < best->median_ms) best = EmpiricalBest{index, median};
    }
    if (!best) throw std::runtime_error("no valid measured configurations");
    return *best;
}

void write_metadata(RunOutput& output, const DeviceInfo& device, const KernelSpec& spec,
                    const Options& options, const std::vector<EvaluationLaunch>& launches, int argc, char** argv) {
    char hostname[256]{};
    if (gethostname(hostname, sizeof hostname - 1) != 0) throw std::runtime_error("gethostname failed");
    const auto uuid = device_uuid(device);
    const auto revision = capture("git rev-parse HEAD 2>/dev/null");
    const auto status = capture("git status --porcelain=v1 --untracked-files=all 2>/dev/null");
    std::ostringstream meta;
    meta << "schema_version=1\nrun_id=" << output.run_id() << "\nutc=" << utc_timestamp()
         << "\nhostname=" << hostname << "\ndevice_index=" << device.index << "\ndevice_uuid=" << uuid
         << "\ndevice_name=" << device.properties.name
         << "\ncompute_capability=" << device.properties.major << '.' << device.properties.minor
         << "\nsm_count=" << device.properties.multiProcessorCount
         << "\ncuda_driver_api_version=" << device.driver << "\ncuda_runtime_version=" << device.runtime
         << "\nnvcc_version=" << __CUDACC_VER_MAJOR__ << '.' << __CUDACC_VER_MINOR__ << '.' << __CUDACC_VER_BUILD__
         << "\nhost_compiler_version=" << __VERSION__ << "\ncuda_flags=" << GPU_LAUNCH_CUDA_FLAGS
         << "\ngit_revision=" << revision << "\ngit_dirty="
         << (status == "unavailable" ? "unknown" : status.empty() ? "false" : "true")
         << "\nmodel_revision=" << model_revision << "\nkernel=" << spec.name << "\nsplit=" << options.split
         << "\nelements=" << options.workload.elements() << "\ninput_seed=" << options.workload.seed
         << "\nordering_seed=" << options.workload.seed << "\nordering_algorithm=mt19937-std-shuffle"
         << "\nwarmups=" << options.warmups << "\nrounds=" << options.rounds << "\nbatch_iters=" << options.batch
         << "\nconfiguration_count=" << launches.size();
    for (const auto& label : roles) {
        const auto role = label.first;
        if (role == LaunchRole::candidate || role == LaunchRole::empirical_best) continue;
        const auto selected = std::find_if(launches.begin(), launches.end(), [role](const auto& entry) {
            return entry.roles & role_bit(role);
        });
        meta << '\n' << label.second << "_launch_id=";
        if (selected == launches.end()) meta << "unavailable";
        else meta << (selected - launches.begin() + 1);
    }
    meta << "\ngpu_load_fields=driver_version,utilization_gpu_pct,memory_used_mib,temperature_c,graphics_clock_mhz"
         << "\ngpu_load_before=" << gpu_snapshot(uuid) << "\ninvocation=";
    for (int i = 0; i < argc; ++i) {
        if (i) meta << ' ';
        meta << std::quoted(argv[i]);
    }
    meta << '\n';
    output.metadata(meta.str());
}

void write_row(RunOutput& output, const KernelSpec& spec, const Options& options, const Launch& launch,
               std::size_t index, unsigned role_mask, const Sample* sample, const char* error) {
    const auto& workload = options.workload;
    std::ostringstream row;
    row << "1," << csv_field(output.run_id()) << ',' << options.split << ',' << spec.name << ',';
    if (spec.two_dimensional) row << ',' << workload.width << ',' << workload.height;
    else row << workload.n << ",,";
    row << ',' << workload.seed << ',' << index + 1 << ',' << role_labels(role_mask) << ',' << describe(launch.block) << ','
        << describe(launch.grid) << ',' << launch.shared << ',';
    if (spec.reduction()) row << launch.final_block.x << ',' << launch.final_grid.x << ',' << launch.final_shared;
    else row << ",,";
    if (sample) {
        row << ",passed,," << sample->round << ',' << sample->order << ',' << options.warmups << ','
            << options.batch << ',' << std::setprecision(17) << sample->milliseconds << ','
            << sample->milliseconds / options.batch;
    } else {
        row << ",failed," << error << ",,," << options.warmups << ',' << options.batch << ",,";
    }
    row << '\n';
    output.csv(row.str());
}

}  // namespace

void evaluate_benchmark(const DeviceInfo& device, const KernelSpec& spec, const Options& options,
                        const std::vector<EvaluationLaunch>& launches, int argc, char** argv) {
    RunOutput output(options.csv);
    write_metadata(output, device, spec, options, launches, argc, argv);

    std::vector<Sample> samples;
    std::optional<std::size_t> active;
    const char* error_code = "";
    try {
        Benchmark benchmark(device, spec, options.workload);
        error_code = "validation_failed";
        for (std::size_t i = 0; i < launches.size(); ++i) {
            active = i;
            benchmark.validate(launches[i].candidate.launch);
        }

        error_code = "warmup_failed";
        for (std::size_t i = 0; i < launches.size(); ++i) {
            active = i;
            for (unsigned j = 0; j < options.warmups; ++j) benchmark.run(launches[i].candidate.launch);
            cuda_check(cudaStreamSynchronize(benchmark.stream()), "warmup synchronization");
        }
        active.reset();

        const Event start;
        const Event stop;
        std::vector<std::size_t> order(launches.size());
        std::iota(order.begin(), order.end(), 0);
        std::mt19937 random(options.workload.seed);
        error_code = "timing_failed";
        for (unsigned round = 0; round < options.rounds; ++round) {
            std::shuffle(order.begin(), order.end(), random);
            for (std::size_t position = 0; position < order.size(); ++position) {
                const auto index = order[position];
                active = index;
                start.record(benchmark.stream());
                for (unsigned j = 0; j < options.batch; ++j) benchmark.run(launches[index].candidate.launch);
                stop.record(benchmark.stream());
                samples.push_back({index, round + 1, position + 1, stop.elapsed_since(start)});
                active.reset();
            }
        }
    } catch (const std::exception& error) {
        for (const auto& sample : samples) {
            if (active && sample.index == *active) continue;
            const auto& entry = launches[sample.index];
            write_row(output, spec, options, entry.candidate.launch, sample.index, entry.roles, &sample, "");
        }
        output.metadata("gpu_load_after=" + gpu_snapshot(device_uuid(device)) + "\nrun_status=failed\n");
        std::string details = "evaluate " + std::string(spec.name) + " elements=" +
                              std::to_string(options.workload.elements());
        if (active) {
            const auto& entry = launches[*active];
            const auto& launch = entry.candidate.launch;
            write_row(output, spec, options, launch, *active, entry.roles, nullptr, error_code);
            details += " block=" + describe(launch.block) + " grid=" + describe(launch.grid);
        }
        throw std::runtime_error(details + ": " + error.what());
    }

    const auto best = empirical_best(samples, launches.size(), options.batch);
    for (const auto& sample : samples) {
        const auto& entry = launches[sample.index];
        const unsigned empirical_role = sample.index == best.index ? role_bit(LaunchRole::empirical_best) : 0;
        write_row(output, spec, options, entry.candidate.launch, sample.index, entry.roles | empirical_role, &sample, "");
    }
    std::ostringstream summary;
    summary << "empirical_best_launch_id=" << best.index + 1
            << "\nempirical_best_median_ms=" << std::setprecision(17) << best.median_ms
            << "\ngpu_load_after=" << gpu_snapshot(device_uuid(device)) << "\nrun_status=passed\n";
    output.metadata(summary.str());
    std::cout << "EVALUATE_PASS run_id=" << std::quoted(output.run_id()) << " configurations=" << launches.size()
              << " samples=" << samples.size() << " csv=" << std::quoted(options.csv)
              << " metadata=" << output.metadata_path() << " empirical_best_launch_id=" << best.index + 1
              << " empirical_best_median_ms=" << std::setprecision(17) << best.median_ms << '\n';
}

}  // namespace gpu_launch
