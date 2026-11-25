#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

[[noreturn]] void reject(const char* api) {
    std::fprintf(stderr, "RECOMMEND_GUARD intercepted %s\n", api);
    std::exit(86);
}

bool allowed(const char* value) {
    const char* mode = std::getenv("GPU_LAUNCH_GUARD_CONTROL");
    return mode && std::strcmp(mode, value) == 0;
}

}  // namespace

extern "C" cudaError_t __real_cudaStreamCreateWithFlags(cudaStream_t*, unsigned);
extern "C" cudaError_t __real___cudaLaunchKernel(const void*, dim3, dim3, void**, std::size_t, cudaStream_t);

extern "C" cudaError_t __wrap_cudaStreamCreateWithFlags(cudaStream_t* stream, unsigned flags) {
    if (!allowed("kernel") && !allowed("event")) reject("cudaStreamCreateWithFlags");
    return __real_cudaStreamCreateWithFlags(stream, flags);
}

extern "C" cudaError_t __wrap___cudaLaunchKernel(const void* function, dim3 grid, dim3 block, void** arguments,
                                                std::size_t shared, cudaStream_t stream) {
    if (!allowed("event")) reject("__cudaLaunchKernel");
    return __real___cudaLaunchKernel(function, grid, block, arguments, shared, stream);
}

extern "C" cudaError_t __wrap_cudaEventCreate(cudaEvent_t*) {
    reject("cudaEventCreate");
}
