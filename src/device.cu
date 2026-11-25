#include "gpu_launch/launch.hpp"

#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace gpu_launch {

void cuda_check(cudaError_t status, const std::string& operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(operation + ": " + cudaGetErrorName(status) + " (" + cudaGetErrorString(status) + ")");
    }
}

DeviceInfo select_device(int index) {
    int count = 0;
    cuda_check(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
    if (index < 0 || index >= count) throw std::runtime_error("CUDA device index is unavailable: " + std::to_string(index));
    DeviceInfo info{};
    info.index = index;
    cuda_check(cudaSetDevice(index), "cudaSetDevice");
    cuda_check(cudaGetDeviceProperties(&info.properties, index), "cudaGetDeviceProperties");
    cuda_check(cudaDriverGetVersion(&info.driver), "cudaDriverGetVersion");
    cuda_check(cudaRuntimeGetVersion(&info.runtime), "cudaRuntimeGetVersion");
    return info;
}

KernelFunction query_function(const void* entry) {
    KernelFunction function{entry, {}};
    cuda_check(cudaFuncGetAttributes(&function.attributes, entry), "cudaFuncGetAttributes");
    return function;
}

int resident_blocks(const DeviceInfo& device, const KernelFunction& function, Shape block,
                    std::size_t dynamic_shared) {
    const auto threads = volume(block);
    const auto& p = device.properties;
    const auto& f = function.attributes;
    if (!threads || threads > static_cast<unsigned>(p.maxThreadsPerBlock) ||
        threads > static_cast<unsigned>(f.maxThreadsPerBlock) ||
        f.sharedSizeBytes > p.sharedMemPerBlock ||
        dynamic_shared > p.sharedMemPerBlock - f.sharedSizeBytes ||
        dynamic_shared > static_cast<std::size_t>(f.maxDynamicSharedSizeBytes)) return 0;
    int resident = 0;
    cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident, function.entry,
               static_cast<int>(threads), dynamic_shared), "cudaOccupancyMaxActiveBlocksPerMultiprocessor");
    return resident;
}

std::string device_uuid(const DeviceInfo& device) {
    std::ostringstream result;
    result << "GPU-" << std::hex << std::setfill('0');
    for (int i = 0; i < 16; ++i) {
        if (i == 4 || i == 6 || i == 8 || i == 10) result << '-';
        result << std::setw(2) << static_cast<unsigned>(static_cast<unsigned char>(device.properties.uuid.bytes[i]));
    }
    return result.str();
}

}  // namespace gpu_launch
