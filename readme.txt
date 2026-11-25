Architecture-Aware CUDA Kernel Launch Configuration Recommender

This CUDA/C++ tool recommends legal block and grid dimensions using device
properties, compiled kernel attributes, and workload size. It supports FP32
vector addition (vadd), dependent arithmetic (compute), two-stage reduction
(reduce), and a row-major transform (transform).

BUILD AND TEST
On CIMS cuda2, use an RTX 2080 Ti, the NVIDIA driver, GNU Make, Bash, GCC/G++
and the cuda-13.0 module. From the extracted directory, run:

  module load cuda-13.0
  make clean
  make CUDA_ARCH=75
  make check

Make uses nvcc with -O3 -std=c++17 -gencode arch=compute_75,code=sm_75
and -lineinfo. The correctness tests exercise all four kernels against
CPU references and check that recommendations are legal and deterministic.

RECOMMEND
  build/gpu-launch recommend --kernel reduce --n 16777216 --device 0
  build/gpu-launch recommend --kernel transform --width 4096 --height 3072 --device 0

Recommendation prints key=value fields for the launch dimensions, resource
estimates, and score. Reduction also reports its final-stage launch.
Recommendation does not execute the workload.

EVALUATE
  mkdir -p build/evaluation
  build/gpu-launch evaluate --kernel vadd --n 262144 --device 0 --split calibration --seed 3175 --warmups 5 --rounds 11 --batch 20 --csv build/evaluation/vadd-262144.csv
  build/gpu-launch evaluate --kernel transform --width 4096 --height 3072 --device 0 --split heldout --seed 3175 --warmups 5 --rounds 11 --batch 20 --csv build/evaluation/transform-4096x3072.csv

Use a fresh CSV path for each run. To reproduce the other workloads, use
the kernel, dimensions, and split from the report's workload table.
Use --n for 1D kernels and --width/--height for the transform. Keep device
0, seed 3175, five warmups, 11 rounds, and 20 operations per timing batch.

Evaluation checks each candidate against a CPU reference before timing.
The CSV records launch IDs, geometry, roles, and milliseconds per
operation. Both reduction stages are timed together. The .meta.txt file
records the device, model, and best-tested launch. configurations.csv
provides the measured launch-ID mapping and per-configuration statistics
used in the report.
