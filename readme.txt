Architecture-Aware CUDA Kernel Launch Configuration Recommender
NYU Courant GPU Architecture and Programming, Project 6

PURPOSE
The CUDA/C++ executable recommends legal grid and block dimensions from
device properties, kernel attributes and workload dimensions. The recommend
mode performs no workload launches. Check verifies output against a CPU
reference. Evaluate validates each candidate and times it with CUDA events.
Supported kernels are vadd, compute, reduce and transform.

REQUIREMENTS
On CIMS cuda2 use the cuda-13.0 environment module, NVIDIA driver, GNU Make,
Bash, GNU GCC/G++ and an RTX 2080 Ti visible as CUDA device 0. The measured
configuration used nvcc 13.0.88, GCC 11.5.0 and sm_75. Python 3.8 or later
with the standard library is needed for the audit and numerical tables.
Matplotlib 3.6+ is optional for plots. Rebuilding the report also uses
ReportLab and Pillow. Poppler is needed only for PDF verification.

BUILD AND CORRECTNESS ON CIMS
From the extracted directory containing this file, run:

  module load cuda-13.0
  make clean
  make CUDA_ARCH=75
  make check

Make compiles with nvcc -O3 -std=c++17 -gencode
arch=compute_75,code=sm_75 -lineinfo. The build and check need no Git
checkout. The build/gpu-launch executable supports --help. The correctness
suite exercises all kernels, irregular sizes, the two reduction stages,
legality, and deterministic recommendation without workload execution in
recommend mode. Expect BENCHMARK_CORRECTNESS_PASS and RECOMMEND_PASS.

RECOMMENDATION EXAMPLES
  build/gpu-launch recommend --kernel reduce --n 16777216 --device 0
  build/gpu-launch recommend --kernel transform --width 4096 --height 3072 --device 0

Expected output is key=value metadata including blockDim, gridDim, estimated
occupancy, the score and explanatory factors. The reduction also reports its
final stage. On the measured GPU the examples select block/grid 512/68 and
(32,8)/(128,2) respectively.

EVALUATION EXAMPLES
  mkdir -p build/submission-check
  build/gpu-launch evaluate --kernel vadd --n 4103 --device 0 --split calibration --seed 3175 --warmups 1 --rounds 3 --batch 5 --csv build/submission-check/vadd.csv
  build/gpu-launch evaluate --kernel transform --width 129 --height 67 --device 0 --split calibration --seed 3175 --warmups 1 --rounds 3 --batch 5 --csv build/submission-check/transform.csv

Each command needs a fresh CSV path and creates a matching .meta.txt. The
CSV contains one row per validated launch and round. The role field is
pipe-delimited. GPU batches, reported medians and speedups use milliseconds
per complete operation. Both stages are timed for reduction. The examples
are smoke tests. The preserved experiment used 5 warmups, 11 rounds and
20 operations per batch for every configuration. Extracted copies without
Git report git_revision=unavailable and git_dirty=unknown in new smoke
metadata. Preserved campaign metadata records the original clean revision.

REPRODUCE THE ANALYSIS
  python3 scripts/audit-phase4.py results/raw/phase4 build/submission-check/audit --verify-manifest
  python3 scripts/analyze.py --out build/submission-check/tables
  cmp results/summary/phase4/workloads.csv build/submission-check/audit/workloads.csv
  cmp results/summary/phase5/configurations.csv build/submission-check/tables/configurations.csv
  cmp results/summary/phase5/aggregates.json build/submission-check/tables/aggregates.json

For plots, add --figures build/submission-check/figures to analyze.py.
To rebuild the report locally after refreshing its default inputs, run
python report/build_report.py. For an automated extracted-package check,
run bash scripts/gate-c.sh on a fresh extraction.

FILES AND PROVENANCE
report/Final_Report.pdf and report/build_report.py contain the final
report and its editable source. results/raw/phase4/ contains the original
CSV, metadata, GPU observations, source revision and SHA-256 manifest.
results/summary/phase4/ holds the Gate D outputs. results/summary/phase5/
holds per-configuration, workload, split, and failure-case tables.
results/figures/ contains publication figures as PNG and SVG.
SHA256SUMS at the ZIP root covers each other packaged file.
Companion distribution URL: https://github.com/its-not-ganesh/gpu-launch-recommender

LIMITATIONS
The assessment covers four controlled FP32 kernels and one Turing GPU.
The best-tested launches come from a finite search space. The model can
lose to fixed defaults. No hardware counters were collected, and sampling
cannot exclude brief contention. The smoke commands are independent of
the authoritative Phase 4 dataset.
