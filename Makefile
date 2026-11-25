NVCC ?= nvcc
CUDA_ARCH ?= 75
CUDA_FLAGS := -O3 -std=c++17 -gencode arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH) -lineinfo
CPPFLAGS := -Iinclude
COMMON := src/cli.cpp src/candidates.cpp src/device.cu src/recommender.cpp
BENCHMARKS := src/benchmark.cu $(wildcard src/kernels/*.cu)
APP := src/main.cu src/evaluate.cu $(COMMON) $(BENCHMARKS)
BUILD_INFO := -DGPU_LAUNCH_CUDA_FLAGS='"$(CUDA_FLAGS)"'
HEADERS := $(wildcard include/gpu_launch/*.hpp)

.PHONY: all check clean FORCE
all: build/gpu-launch

build/.compile-config: FORCE
	@mkdir -p build
	@printf '%s\n' '$(NVCC) $(CXX) $(CUDA_FLAGS) $(CPPFLAGS)' > $@.tmp
	@cmp -s $@.tmp $@ || mv -f $@.tmp $@
	@rm -f $@.tmp

build/gpu-launch: $(APP) $(HEADERS) Makefile build/.compile-config
	@mkdir -p build
	$(NVCC) -ccbin "$(CXX)" $(CUDA_FLAGS) $(CPPFLAGS) $(BUILD_INFO) $(APP) -o $@

build/infrastructure-test: tests/infrastructure.cu $(COMMON) $(HEADERS) Makefile build/.compile-config
	@mkdir -p build
	$(NVCC) -ccbin "$(CXX)" $(CUDA_FLAGS) $(CPPFLAGS) tests/infrastructure.cu $(COMMON) -o $@

build/recommend-guard: $(APP) tests/recommend-guard.cu $(HEADERS) Makefile build/.compile-config
	$(NVCC) -ccbin "$(CXX)" $(CUDA_FLAGS) $(CPPFLAGS) $(BUILD_INFO) \
		-Xlinker=--wrap=cudaStreamCreateWithFlags,--wrap=__cudaLaunchKernel,--wrap=cudaEventCreate \
		$(APP) tests/recommend-guard.cu -o $@

check: all build/infrastructure-test build/recommend-guard
	./build/infrastructure-test
	bash tests/check.sh
	bash tests/check-recommend.sh

clean:
	rm -f build/gpu-launch build/infrastructure-test build/recommend-guard build/.compile-config build/.compile-config.tmp
