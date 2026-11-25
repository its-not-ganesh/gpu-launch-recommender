"""Build the editable academic report from verified statistics and figures."""

from pathlib import Path
import csv
import json
import os
from datetime import datetime, timezone

import matplotlib
from PIL import Image as Raster
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results/summary/phase5"
FIGURES = ROOT / "results/figures"
URL = "https://github.com/its-not-ganesh/gpu-launch-recommender"
INK = colors.HexColor("#152b40")
MUTED = colors.HexColor("#405971")
BLUE = colors.HexColor("#1d566d")
FONT = "ReportSans"
fonts = Path(matplotlib.get_data_path()) / "fonts/ttf"
for name, file in (("ReportSans", "DejaVuSans.ttf"), ("ReportSansBold", "DejaVuSans-Bold.ttf"),
                   ("ReportSerif", "DejaVuSerif.ttf"), ("ReportSerifBold", "DejaVuSerif-Bold.ttf"),
                   ("ReportMono", "DejaVuSansMono.ttf")):
    pdfmetrics.registerFont(TTFont(name, str(fonts / file)))
pdfmetrics.registerFontFamily("ReportSans", normal="ReportSans", bold="ReportSansBold")
pdfmetrics.registerFontFamily("ReportSerif", normal="ReportSerif", bold="ReportSerifBold")
os.environ.setdefault("SOURCE_DATE_EPOCH", str(int(datetime(2026, 11, 25, 12, tzinfo=timezone.utc).timestamp())))


def rows(name):
    with (SUMMARY / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


agg = json.loads((SUMMARY / "aggregates.json").read_text(encoding="utf-8"))
config = rows("configurations.csv")
workloads = rows("workloads.csv")
failures = rows("failure_cases.csv")
freeze = dict(line.split("=", 1) for line in
              (ROOT / "results/raw/phase4/protocol.txt").read_text(encoding="utf-8").splitlines())
held = agg["heldout"]
cal = agg["calibration"]
styles = {
    "title": ParagraphStyle("title", fontName="ReportSansBold", fontSize=23, leading=27, textColor=INK, spaceAfter=10),
    "meta": ParagraphStyle("meta", fontName=FONT, fontSize=10, leading=14, textColor=MUTED, spaceAfter=12),
    "h1": ParagraphStyle("h1", fontName="ReportSansBold", fontSize=17, leading=21, textColor=INK,
                         spaceBefore=8, spaceAfter=7, keepWithNext=True),
    "h2": ParagraphStyle("h2", fontName="ReportSansBold", fontSize=11, leading=15, textColor=BLUE,
                         spaceBefore=6, spaceAfter=4, keepWithNext=True),
    "body": ParagraphStyle("body", fontName="ReportSerif", fontSize=9.6, leading=12.5, textColor=INK, spaceAfter=6),
    "caption": ParagraphStyle("caption", fontName=FONT, fontSize=8.5, leading=11.4, textColor=MUTED, spaceAfter=8),
    "math": ParagraphStyle("math", fontName=FONT, fontSize=10, leading=15, textColor=INK,
                           leftIndent=12, spaceAfter=6),
    "cell": ParagraphStyle("cell", fontName=FONT, fontSize=8.4, leading=11, textColor=INK),
    "th": ParagraphStyle("th", fontName="ReportSansBold", fontSize=8.4, leading=10, textColor=colors.white),
}
story = []


def p(text, kind="body"):
    story.append(Paragraph(text.replace("−", "-").replace("–", "-"), styles[kind]))


def h(text):
    p(text, "h1")


def sub(text):
    p(text, "h2")


def fig(name, caption):
    path = FIGURES / (name + ".png")
    with Raster.open(path) as raster:
        w, height = raster.size
    width = 450 if name == "01-heldout-slowdown" else 490
    picture = Image(str(path), width=width, height=width * height / w)
    picture.hAlign = TA_LEFT
    story.append(KeepTogether([picture, Paragraph(caption, styles["caption"])]))


def table(data, widths):
    content = [[Paragraph(str(cell), styles["th" if index == 0 else "cell"]) for cell in row]
               for index, row in enumerate(data)]
    item = Table(content, colWidths=widths, repeatRows=1, hAlign="LEFT")
    item.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BLUE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), (colors.white, colors.HexColor("#ecf3f6"))),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
        ("LINEBELOW", (0, -1), (-1, -1), .55, colors.HexColor("#cad8df")),
    ]))
    story.extend([item, Spacer(1, 7)])


def ms_to_us(number):
    return 1000 * float(number)


def case_gap(kernel, size):
    return float(next(r for r in workloads if r["split"] == "heldout" and r["kernel"] == kernel and
                      (r["n"] or r["width"]) == size)["slowdown_vs_empirical_best_pct"])


def match_transform_full():
    return next(r for r in config if r["kernel"] == "transform" and r["width"] == "4096"
                and (r["block_x"], r["block_y"], r["grid_x"], r["grid_y"]) == ("32", "8", "128", "384"))


def link(text, target):
    return f'<link href="{target}" color="#1d566d">{text}</link>'


if __name__ == "__main__":
    p("Architecture-Aware CUDA Kernel<br/>Launch Configuration Recommender", "title")
    p("Ganesh Gamanagatti<br/>NYU Courant · GPU Architecture and Programming · Project 6<br/>Professor Mohamed Zahran · November 25, 2026", "meta")
    h("Abstract")
    p(f"We implement a CUDA/C++ tool that recommends block and grid dimensions using device properties, "
      f"kernel resource usage and problem shape. Recommendation requires no kernel executions. A controlled evaluation "
      f"on an RTX 2080 Ti compared fixed launches, a CUDA occupancy-only baseline and the fastest configuration tested. "
      f"Twelve workloads contributed {len(config)} unique workload-launch pairs and "
      f"{cal['valid_batches'] + held['valid_batches']:,} validated timing batches. On eight held-out workloads the "
      f"median slowdown versus best tested was {held['median_slowdown_pct']:.2f}% "
      f"(range {held['min_slowdown_pct']:.2f}–{held['max_slowdown_pct']:.2f}%). "
      "The large reduction and transform expose distinct weaknesses in occupancy saturation and grid budgeting.")
    h("Introduction")
    p("CUDA block dimensions determine how threads form warps, how many blocks can reside on a streaming multiprocessor "
      "(SM), and how adjacent lanes map to memory addresses. Grid dimensions distribute those blocks across SMs. "
      "With grid-stride traversal, a small grid can still cover a large problem by increasing work per thread. "
      "The resulting choices affect latency hiding, loop overhead, boundary waste and load balance.")
    p("This project asks how close an interpretable, measurement-free recommendation comes to the fastest tested launch "
      "for four FP32 workloads. The contributions are a bounded candidate policy with CUDA legality checks, an explainable "
      "ranker, and a reproducible evaluator with correctness gates and independent dataset verification.")
    h("Literature Survey")
    sub("Resource heuristics")
    p("Ryoo et al. [1] describe resource allocation and memory-access tradeoffs in early CUDA applications. "
      "NVIDIA’s guides [2, 5] explain occupancy, coalescing and architecture-specific constraints. Resource rules "
      "quickly exclude impossible launches, but a resident-warp count alone does not describe memory transactions or "
      "how much work each grid-stride loop performs.")
    sub("Analytical performance models")
    p("Hong and Kim [3] model the interaction of memory-level and thread-level parallelism. Their approach explains "
      "latency overlap in more detail than a scalar occupancy heuristic. Such predictions need kernel and architecture "
      "inputs beyond the runtime attributes used here. This project ranks launches with dimensionless factors and makes "
      "no absolute time prediction.")
    sub("Measurement-driven autotuning")
    p("CLTune [4] benchmarks parameter combinations using configurable search strategies. Timed search captures effects "
      "that simple models miss, with a cost in target-device runs and possible sensitivity to load. We use timed search "
      "only to evaluate a bounded candidate set, while normal use relies on device and kernel metadata.")

    story.append(PageBreak())
    h("Proposed Idea")
    sub("Candidate generation and feasibility")
    p("The 1D block set is {64, 128, 256, 512} threads. The 2D set is {(8,8), (16,8), (32,4), (16,16), (32,8)}. "
      "For each block, the generator tests CTA budgets of M, 2M and 4M, plus full logical coverage, where M is the SM count. "
      "Grid axes are clipped to device limits. A 2D budget fills the x tile dimension first, then adds y tile rows when "
      "the budget permits. Reduction stage 1 is additionally capped at 4M partial CTAs. Identical complete launches share one ID.")
    p("CUDA runtime properties and compiled function attributes constrain block dimensions, threads, register and "
      "shared-memory residency. The occupancy API calculates viable resident CTAs per SM for the exact function and "
      "shared allocation. The final reduction stage is checked separately. The command rejects invalid geometry and "
      "reports the grid, block, shared bytes, occupancy, residency and score factors for the chosen launch.")
    sub("Grid-stride traversal")
    p("A 1D thread walks indices separated by blockDim.x × gridDim.x. The 2D kernel walks both axes, with blockDim and "
      "gridDim determining each stride. Boundary predicates keep partial tiles correct. This separates logical tiles from "
      "the physical CTAs launched and permits several grids for an identical block shape.")
    sub("Analytical score")
    p("Let G be the physical CTA count, t the threads per CTA, R the CUDA-reported resident CTAs per SM, "
      "S = M × R, T the logical tile count, W the warp size, and W<sub>max</sub> the maximum resident warps per SM. "
      "For N logical elements, the model computes the following quantities.")
    p("A = min(G,S) × ceil(t/W) / (M × W<sub>max</sub>)<br/>"
      "U = G / [min(G,S) × ceil(G/S)]<br/>"
      "E = N/(T × t), q = min(1,G/T), C = min(1, blockDim.x/W) for 2D and 1 otherwise", "math")
    p("<b>score = [1 + l(1 − q) + dq + p] / [min(1,A/a) × (0.75 + 0.25U) × E × C]</b>", "math")
    table([["Work class", "Occupancy target a", "Loop weight l", "Dispatch weight d"],
           ["Vector add and transform", "0.50", "0.10", "0.18"],
           ["Dependent FMA", "0.75", "0.15", "0.05"], ["Reduction", "0.50", "0.10", "0.05"]],
          [175, 107, 103, 105])
    p("The score trades off grid-limited resident warps (A), scheduled-wave fill (U), logical tile efficiency (E), "
      "and a row-contiguity preference (C). Saturation limits the reward for additional occupancy once the target is met. "
      "The numerator penalizes loop work and dispatch. For reduction, p = 0.20 × G/(4M) represents partial-result overhead. "
      "Otherwise p is zero. The lowest score wins, with smaller G then lexicographic block and grid dimensions breaking ties.")
    p("Fixed 1 and fixed 2 use 128 and 256 threads for 1D, or (16,8) and (16,16) for 2D, with coverage grids subject "
      "to the reduction cap. The occupancy-only baseline maximizes R × ceil(t/W) using coverage grids and favors fewer "
      "threads on ties. The empirical reference is the minimum measured median among all valid launches in the case.")

    story.append(PageBreak())
    h("Experimental Setup")
    sub("Hardware, toolchain and provenance")
    p("Measurements ran on CIMS cuda2, GPU 0, an NVIDIA GeForce RTX 2080 Ti (Turing, compute capability 7.5) "
      "with 68 SMs and 11,264 MiB reported by nvidia-smi. The device supports 32-thread warps, 32 resident warps "
      "and up to 16 CTAs per SM [5], 1,024 threads per CTA and per SM, 65,536 registers per SM and 64 KiB shared "
      "memory per SM. The default per-block shared limit is 48 KiB. Another visible GPU was outside the experiment.")
    p("CIMS ran RHEL 9.9, NVIDIA driver 595.104.02, GCC 11.5.0 and CUDA 13.0 (nvcc 13.0.88). "
      "The build used C++17 and the flags below. No fast-math flag was applied.")
    p("-O3 -std=c++17 -gencode arch=compute_75,code=sm_75 -lineinfo", "math")
    p(f"Frozen measurement source revision <font name='ReportMono' size='9'>{freeze['source_revision']}</font>. "
      "The candidate-generator and model Git blob IDs and binary SHA-256 are in the raw protocol record. "
      "Documentation, analysis and packaging commits are later revisions. The model and candidate policy stayed frozen "
      f"after calibration. Companion distribution URL {link(URL, URL)}.")
    sub("Benchmarks and workload matrix")
    table([["Kernel and operation", "Calibration", "Held-out"],
           ["Vector add, FP32 x + y", "262,144", "1,048,583 and 16,777,216"],
           ["Compute, 128 dependent FP32 fused multiply-adds per element", "131,072", "32,771 and 1,048,583"],
           ["Shared-memory tree reduction with a second kernel for partials", "262,144", "1,048,583 and 16,777,216"],
           ["Row-major weighted sum of direct and horizontally mirrored pixels", "512×512", "1009×769 and 4096×3072"]],
          [249, 81, 160])
    sub("Measurement and validation protocol")
    p("The evaluator initializes deterministic FP32 inputs with seed 3175. A CPU reference and output sentinels "
      "verify every candidate before timing. Vector addition and the transform are compared exactly for the chosen "
      "dyadic inputs. The compute kernel uses an absolute-plus-relative tolerance. Reduction validates all partials "
      "and the final sum using exact integer-valued FP32 inputs.")
    p("Each candidate receives five untimed warmups, then one batch per round in 11 rounds with a seeded shuffled "
      "order. A CUDA event pair brackets 20 complete operations on a single stream. Both reduction stages are inside "
      "the interval. Allocations, copies, CPU reference work and correctness checks are outside it. The elapsed time "
      "divided by 20 is one observation, so a candidate contributes 11 observations.")
    p("Preflight waits for GPU utilization below 10% and no competing CUDA compute PID. Sampling continued "
      "during the application and after exit. Five held-out cases waited for preflight. The monitor observed no "
      "competing compute process in accepted samples. Load and process sampling cannot detect every transient overlap.")

    story.append(PageBreak())
    h("Experiments &amp; Analysis")
    p(f"Calibration produced {cal['configurations']} configurations and {cal['valid_batches']} valid batches. "
      f"Held-out produced {held['configurations']} configurations and {held['valid_batches']:,} valid batches. "
      "All 12 case datasets passed the checksum and schema audit. Every unique configuration had 11 positive "
      "normalized observations, and the empirical ID was independently recomputed from those observations.")
    p("For each configuration we report the median and IQR (75th minus 25th percentile, with linear interpolation). "
      "Slowdown is 100 × (recommended median / best-tested median − 1). Baseline speedup is baseline median divided "
      "by recommended median. Split-level medians weight each workload equally. Win and tie counts use the original "
      "unrounded medians. Counts are descriptive and do not test significance.")
    tab = [["Split", "Kernel", "Input", "Cfg.", "Rec. µs", "Best µs", "Gap"]]
    for row in workloads:
        size = f"{int(row['n']):,}" if row["n"] else f"{row['width']}×{row['height']}"
        tab.append(["Cal." if row["split"] == "calibration" else "Held", row["kernel"], size, row["configurations"],
                    f"{ms_to_us(row['recommended_median_ms']):.2f}",
                    f"{ms_to_us(row['empirical_best_median_ms']):.2f}",
                    f"{float(row['slowdown_vs_empirical_best_pct']):.2f}%"])
    table(tab, [46, 65, 115, 44, 73, 73, 74])
    p("Table 1. One row per workload. Cfg. counts distinct full launch tuples within the workload. "
      "All 185 configuration medians, quartiles and IQRs are included in configurations.csv.", "caption")
    fig("01-heldout-slowdown", "Figure 1. Held-out slowdown relative to the smallest valid median in the tested set. "
        "The irregular 1009×769 transform matched the best-tested launch; both large-workload gaps are retained.")

    story.append(PageBreak())
    sub("Comparisons with fixed and occupancy-only launches")
    p(f"Calibration median slowdown was {cal['median_slowdown_pct']:.2f}%. The recommendation beat fixed 1 in 0/4, "
      "fixed 2 in 1/4 with one tie, and occupancy-only in 3/4. Held-out median slowdown was "
      f"{held['median_slowdown_pct']:.2f}%. Held-out wins were 3/8 against fixed 1, 1/8 against fixed 2, and "
      "4/8 against the occupancy-only baseline, with one exact occupancy tie. Median held-out speedups were "
      f"{held['baselines']['fixed_1']['median_speedup']:.3f}×, {held['baselines']['fixed_2']['median_speedup']:.3f}× "
      f"and {held['baselines']['occupancy']['median_speedup']:.3f}× respectively.")
    fig("02-baseline-comparison", "Figure 2. Each point is a baseline median divided by the recommendation median. "
        "The dashed line marks equality. The uneven results include favorable and unfavorable workload cases.")
    fig("03-representative-configurations", "Figure 3. Measured timing for all candidate IDs on two held-out "
        "1,048,583-element workloads. Bars show Q1 through Q3. The markers identify the recommendation and the "
        "empirical winner. Launch IDs map to block and grid dimensions in configurations.csv.")
    p("The two kernels use the same input size but differ in instructions and memory traffic. Their different timing "
      "profiles are evidence that launch choice should account for workload behavior. Event timings describe total "
      "kernel duration and cannot identify the contribution of instructions, cache behavior or memory stalls.")

    story.append(PageBreak())
    sub("Failure case 1 · Reduction with 16,777,216 elements")
    red, red_best, transform, transform_best = failures
    p(f"<b>Measured.</b> With both stages timed, block 512 and grid 68 took {ms_to_us(red['median_ms']):.2f} µs "
      f"(IQR {ms_to_us(red['iqr_ms']):.3f} µs). Block 512 and grid 136 took {ms_to_us(red_best['median_ms']):.2f} µs "
      f"(IQR {ms_to_us(red_best['iqr_ms']):.3f} µs). The recommendation was {case_gap('reduce','16777216'):.2f}% slower. "
      "The block size and compiled stage-1 kernel are identical. The grid change also doubles the partial count sent to stage 2.")
    fig("04-large-reduction", "Figure 4. Large reduction across all three tested stage-1 grid sizes and four block sizes. "
        "Markers show the chosen 512-thread launches. Error bars span the middle half of batches.")
    p("<b>Analytical explanation.</b> CUDA reports two resident 512-thread CTAs per SM. Each CTA contributes 16 warps. "
      "With 68 blocks, the model estimates one resident CTA per SM on average, giving A = 0.50. With 136 blocks, "
      "A reaches 1.00. The maximum tile visits per CTA drop from 482 to 241. The shared allocation is 2,048 bytes per CTA "
      "and the function uses 14 registers per thread. The 1,024-thread SM limit already caps residency at two CTAs.")
    p("The 0.50 occupancy target gives both grids the same saturated occupancy factor. The reduction penalty rises "
      "from 0.05 to 0.10 as the partial count doubles, while the bounded loop term changes only slightly. The ranker therefore "
      "prefers 68 CTAs despite the much longer per-CTA work and the extra resident warps available at 136 CTAs.")
    p("<b>Hypothesis.</b> More concurrent warps and shorter grid-stride loops likely improve latency hiding or memory "
      "throughput. The events include the final stage and cannot separate its cost or establish the dominant bottleneck. "
      "Stage-resolved timing and hardware-counter profiling would test this explanation.")
    sub("Measurement dispersion")
    p(f"The largest configuration IQR relative to its median was "
      f"{max(float(r['max_candidate_iqr_pct']) for r in workloads):.2f}% across the campaign. The largest single batch "
      f"relative to its own configuration median was {max(float(r['max_batch_to_candidate_median_ratio']) for r in workloads):.3f}×. "
      "No numerical or timing failures were recorded, and no attempt required replacement. IQR describes observed "
      "within-run variation. A single campaign on one shared device does not estimate cross-day variability.")

    story.append(PageBreak())
    sub("Failure case 2 · 4096×3072 row-major transform")
    full = match_transform_full()
    p(f"<b>Measured.</b> Block (32,8) with grid (128,2) took {ms_to_us(transform['median_ms']):.2f} µs. "
      f"The best-tested (16,8) with grid (256,384) took {ms_to_us(transform_best['median_ms']):.2f} µs, "
      f"a {case_gap('transform','4096'):.2f}% slowdown for the recommendation. Holding block (32,8) fixed and "
      f"switching to full coverage (128,384) gave {ms_to_us(full['median_ms']):.2f} µs, much closer to the winner.")
    fig("05-large-transform", "Figure 5. All large-transform configurations by physical grid CTA count and block shape. "
        "The horizontal axis is logarithmic. Error bars span Q1 through Q3.")
    p("<b>Analytical explanation.</b> The recommended grid launches 256 of 49,152 logical tiles (0.521% physical "
      "coverage). Its x stride covers the width in one visit, but each CTA handles 192 tile visits down y. The "
      "best-tested launch uses 98,304 CTAs for 98,304 smaller tiles and one tile visit per CTA. The ranker gives the "
      "32-wide block a higher coalescing factor and favors the capped grid through saturation and dispatch terms.")
    p("<b>Hypothesis.</b> The same-block comparison supports a large grid-dependent effect in this case. The smaller "
      "gap between the full-coverage block shapes may reflect transaction organization, scheduling, cache locality or "
      "overhead. A 16-wide block can split a warp across rows, while each row segment can still access adjacent words. "
      "The results do not isolate coalescing or establish a cache mechanism. No hardware counters were collected.")

    story.append(PageBreak())
    h("Conclusions")
    p("<b>1. A metadata-only ranker is practical.</b> The implementation returned legal, deterministic 1D and 2D launches "
      "without workload launches during recommendation. Its held-out median slowdown versus the best tested was 6.72%.")
    p("<b>2. Fixed defaults remain competitive.</b> The analytical method won only three held-out cases against fixed 1 "
      "and one against fixed 2. Occupancy-only was also competitive. The large reduction and transform expose "
      "how occupancy saturation and limited grid-loop penalties can prefer too little physical parallelism.")
    p("<b>3. The evidence is scoped.</b> Four controlled kernels on one Turing GPU and a finite candidate set bound the "
      "conclusions. Event timing and sampled load leave possible short-lived contention and unmeasured microarchitectural "
      "effects. A future cross-device study with stage timing and hardware counters could separate those effects.")
    h("References")
    for citation in [
        "Shane Ryoo, Christopher I. Rodrigues, Sara S. Baghsorkhi, Sam S. Stone, David B. Kirk, and Wen-mei W. Hwu. "
        f"Optimization Principles and Application Performance Evaluation of a Multithreaded GPU Using CUDA. PPoPP, pp. 73–82, 2008. {link('doi:10.1145/1345206.1345220','https://doi.org/10.1145/1345206.1345220')}.",
        f"NVIDIA. CUDA C++ Best Practices Guide, CUDA 13.0. {link('CUDA documentation','https://docs.nvidia.com/cuda/archive/13.0.3/cuda-c-best-practices-guide/index.html')}.",
        "Sunpyo Hong and Hyesoon Kim. An Analytical Model for a GPU Architecture with Memory-Level and Thread-Level "
        f"Parallelism Awareness. ISCA, pp. 152–163, 2009. {link('doi:10.1145/1555754.1555775','https://doi.org/10.1145/1555754.1555775')}.",
        "Cedric Nugteren and Valeriu Codreanu. CLTune: A Generic Auto-Tuner for OpenCL Kernels. MCSoC, 2015. "
        f"{link('doi:10.1109/MCSoC.2015.10','https://doi.org/10.1109/MCSoC.2015.10')}.",
        f"NVIDIA. Turing Tuning Guide. {link('Streaming multiprocessor and occupancy limits','https://docs.nvidia.com/cuda/archive/13.2.0/turing-tuning-guide/index.html')}.",
    ]:
        p(f"[{len([s for s in story if isinstance(s, Paragraph) and s.getPlainText().startswith('[')]) + 1}] {citation}")
    sub("Artifact reproduction")
    p("The submitted readme.txt documents the CIMS build, correctness checks, recommendation and evaluation examples. "
      "scripts/analyze.py revalidates the raw manifest and regenerates all statistical tables and five figures. "
      "The editable report source embeds those outputs. The raw CSV timings are retained with original precision.")

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#c4d3df"))
        canvas.line(58, 45, 554, 45)
        canvas.setFont(FONT, 8)
        canvas.setFillColor(MUTED)
        canvas.drawString(58, 32, "NYU Courant · GPU Architecture and Programming")
        canvas.drawRightString(554, 32, f"{doc.page:02}")
        canvas.restoreState()

    doc = SimpleDocTemplate(str(ROOT / "report/Final_Report.pdf"), pagesize=letter, leftMargin=56, rightMargin=56,
                            topMargin=49, bottomMargin=57, title="Architecture-Aware CUDA Kernel Launch Configuration Recommender",
                            author="Ganesh Gamanagatti", pageCompression=1, invariant=1)
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
