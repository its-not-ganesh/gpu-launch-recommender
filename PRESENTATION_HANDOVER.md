# Presentation handover

The research question is whether a metadata-only CUDA launch ranker can approach the best tested configuration across FP32 workloads. The tool uses candidate block shapes, 1D/2D grid-stride traversal, CUDA resource feasibility, and a dimensionless score. The score saturates an occupancy term and considers wave fill, tile usage, row width, and dispatch and loop penalties. Evaluation compares two fixed defaults, occupancy-only, and the best tested median. It times both reduction stages and performs correctness checks before timing.

The CIMS RTX 2080 Ti has 68 Turing SMs, 32-thread warps, and compute capability 7.5. CIMS ran nvcc 13.0.88 and GCC 11.5.0. Seed 3175, five warmups and 11 randomized rounds of 20 operations were fixed. Four calibration cases contributed 63 configurations and 693 batches. Eight held-out cases contributed 122 configurations and 1,342 batches. No tuning followed calibration. Held-out median slowdown was 6.72% with a 0–46.67% range. The recommendation won 3/8 versus fixed 1, 1/8 versus fixed 2, and 4/8 with one tie versus occupancy-only.

Use the required eight slides for the eight-minute talk and allow two minutes for Q&A.

1. **Title**. Project 6, author, architecture-aware launch selection.
2. **Problem and motivation**. Show how identical work maps to different physical grids and why block geometry affects resident warps and access patterns.
3. **Literature survey**. Contrast resource heuristics (Ryoo et al. and NVIDIA), analytical parallelism models (Hong and Kim), and measurement-driven tuning (Nugteren and Codreanu). The exact citations are in the report.
4. **Proposed idea**. Show device/function attributes → feasible tuples → scoring → recommendation. Explain occupancy saturation and the absence of benchmark launches during recommendation.
5. **Experimental setup**. Use a compact device and workload diagram. Explain the two splits, CPU references, event batches, and the best-tested comparator.
6. **Results and analysis**. Use `01-heldout-slowdown` and `02-baseline-comparison` in `results/figures/`. Report exact win counts and mention fixed defaults remain competitive. `03-representative-configurations` is optional support for kernel-specific sensitivity.
7. **Results and analysis**. Use `04-large-reduction` and `05-large-transform`. At 512 threads, the reduction changed 68 to 136 CTAs, 50% to 100% analytical grid-limited occupancy, and 482 to 241 maximum tile visits per CTA. Time fell from 191.80 to 130.76 µs. For the large transform, (32,8)/(128,2) took 230.50 µs, the same block at full coverage took 191.43 µs, and the best tested (16,8)/(256,384) took 189.54 µs. These comparisons support grid effects. Cache, transaction and latency mechanisms remain hypotheses.
8. **Conclusions**. A legal fast-to-compute ranker, competitive defaults and meaningful failure cases, and the scope of the evidence. Distinguish best tested from global optimum.

The report is `report/Final_Report.pdf`. Use `results/summary/phase5/aggregates.json`, `workloads.csv`, `failure_cases.csv`, and `configurations.csv` to check numbers. SVG figures are available for resizing. Limitations include one GPU architecture, controlled workloads, bounded candidate space, no hardware counters, and possible unobserved contention between GPU load samples.
