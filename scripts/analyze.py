from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ROLES = ("fixed_1", "fixed_2", "occupancy")
NAMES = {"vadd": "Vector add", "compute": "Compute", "reduce": "Reduction", "transform": "2D transform"}


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def label(row):
    size = f"{int(row['n']):,}" if row["n"] else f"{row['width']}×{row['height']}"
    return f"{NAMES[row['kernel']]} {size}"


def analyze(raw, out):
    subprocess.run([sys.executable, str(ROOT / "scripts/audit-phase4.py"), str(raw), str(out),
                    "--verify-manifest"], check=True)
    cases = read_csv(out / "workloads.csv")
    configurations = []
    for case in cases:
        groups = defaultdict(list)
        for row in read_csv(raw / case["split"] / (case["run_id"] + ".csv")):
            groups[int(row["launch_id"])].append(row)
        for launch, rows in sorted(groups.items()):
            values = sorted(float(row["ms_per_operation"]) for row in rows)
            q1, median, q3 = statistics.quantiles(values, n=4, method="inclusive")
            row = rows[0]
            configurations.append({
                **{key: row[key] for key in ("split", "kernel", "n", "width", "height", "run_id", "launch_id",
                   "roles", "block_x", "block_y", "block_z", "grid_x", "grid_y", "grid_z", "shared_bytes",
                   "final_block_x", "final_grid_x", "final_shared_bytes")},
                "batch_count": len(values), "median_ms": median, "q1_ms": q1, "q3_ms": q3, "iqr_ms": q3-q1,
            })
        per_id = {int(row["launch_id"]): row for row in configurations if row["run_id"] == case["run_id"]}
        for role in (*ROLES, "recommended", "empirical_best"):
            row = per_id[int(case[f"{role}_launch_id"])]
            if row["median_ms"] != float(case[f"{role}_median_ms"]):
                raise ValueError(f"median mismatch: {case['run_id']} {role}")
    write_csv(out / "configurations.csv", configurations)

    aggregates = {}
    for split in ("calibration", "heldout"):
        subset = [row for row in cases if row["split"] == split]
        slow = [float(row["slowdown_vs_empirical_best_pct"]) for row in subset]
        comparison = {}
        for role in ROLES:
            ratios = [float(row[f"speedup_vs_{role}"]) for row in subset]
            comparison[role] = {"wins": sum(x > 1 for x in ratios), "ties": ratios.count(1),
                                "losses": sum(x < 1 for x in ratios), "median_speedup": statistics.median(ratios)}
        aggregates[split] = {"cases": len(subset), "configurations": sum(int(r["configurations"]) for r in subset),
                             "valid_batches": sum(int(r["valid_batches"]) for r in subset),
                             "median_slowdown_pct": statistics.median(slow), "min_slowdown_pct": min(slow),
                             "max_slowdown_pct": max(slow), "baselines": comparison}
    (out / "aggregates.json").write_bytes((json.dumps(aggregates, indent=2) + "\n").encode())

    failures = []
    for kernel, size in (("reduce", "16777216"), ("transform", "4096")):
        case = next(r for r in cases if r["split"] == "heldout" and r["kernel"] == kernel and
                    (r["n"] or r["width"]) == size)
        metadata = dict(line.split("=", 1) for line in
                        (raw / "heldout" / (case["run_id"] + ".recommend.txt")).read_text().splitlines())
        for role in ("recommended", "empirical_best"):
            item = next(r for r in configurations if r["run_id"] == case["run_id"] and
                        r["launch_id"] == case[f"{role}_launch_id"])
            bx, by, gx, gy = (int(item[k]) for k in ("block_x", "block_y", "grid_x", "grid_y"))
            tx = (int(case["n"] or case["width"]) + bx - 1) // bx
            ty = (int(case["height"] or 1) + by - 1) // by
            resident = int(metadata["resident_blocks_per_sm"])
            sms = int(metadata["sm_count"])
            warp = int(metadata["warp_size"])
            grid_occupancy = (min(gx * gy, sms * resident) * ((bx * by + warp - 1) // warp)
                              / (sms * 32))
            failures.append({"kernel": kernel, "role": role, "block": f"{bx}x{by}", "grid": f"{gx}x{gy}",
                             "logical_tiles": tx * ty, "launched_ctas": gx * gy, "coverage": gx * gy / (tx * ty),
                             "max_tile_visits_per_cta": ((tx + gx - 1) // gx) * ((ty + gy - 1) // gy),
                             "resident_ctas_per_sm": metadata["resident_blocks_per_sm"] if role == "recommended"
                             or kernel == "reduce" else "",
                             "grid_limited_occupancy_estimate": grid_occupancy if role == "recommended"
                             or kernel == "reduce" else "",
                             "median_ms": item["median_ms"], "iqr_ms": item["iqr_ms"]})
    write_csv(out / "failure_cases.csv", failures)
    if len(configurations) != 185 or sum(r["batch_count"] for r in configurations) != 2035:
        raise ValueError("unexpected totals in analysis")
    return cases, configurations


def plots(cases, configs, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.labelcolor": "#203248",
                         "text.color": "#17283d", "savefig.dpi": 240, "svg.hashsalt": "gpu-launch-v1"})
    blue, teal, orange, gray = "#2463a8", "#127c77", "#bf581f", "#678096"

    def save(fig, name):
        fig.savefig(out / f"{name}.png", bbox_inches="tight", pad_inches=0.1)
        fig.savefig(out / f"{name}.svg", bbox_inches="tight", pad_inches=0.1, metadata={"Date": None})
        plt.close(fig)

    held = [r for r in cases if r["split"] == "heldout"]
    names = [label(r) for r in held]
    values = [float(r["slowdown_vs_empirical_best_pct"]) for r in held]
    fig, ax = plt.subplots(figsize=(8, 3.45), layout="constrained")
    ax.barh(names, values, color=[orange if v > 20 else blue for v in values], height=.62)
    ax.invert_yaxis()
    ax.set_xlim(0, 53)
    ax.grid(axis="x", alpha=.24)
    ax.set_axisbelow(True)
    ax.set_xlabel("Slowdown vs best tested (%) · lower is better")
    for i, v in enumerate(values):
        ax.text(v + .55, i, f"{v:.2f}%", va="center", fontsize=8)
    save(fig, "01-heldout-slowdown")

    fig, ax = plt.subplots(figsize=(8, 3.6), layout="constrained")
    for offset, role, color, marker, name in zip((-.22, 0, .22), ROLES, (blue, orange, teal), ("o", "s", "^"),
                                                 ("Fixed 1", "Fixed 2", "Occupancy-only")):
        ax.scatter([float(r[f"speedup_vs_{role}"]) for r in held], [i+offset for i in range(8)],
                   s=26, marker=marker, c=color, label=name)
    ax.set_yticks(range(8), names)
    ax.invert_yaxis()
    ax.axvline(1, color=gray, linestyle="--", linewidth=1)
    ax.set_xlabel("Baseline median / recommended median · values > 1 favor recommendation")
    ax.grid(axis="x", alpha=.22)
    ax.legend(loc="upper right", frameon=False)
    save(fig, "02-baseline-comparison")

    def selected(kernel, size):
        return [r for r in configs if r["split"] == "heldout" and r["kernel"] == kernel and
                (r["n"] or r["width"]) == size]

    def markers(ax, groups, x):
        for role, shape, color in (("recommended", "D", orange), ("empirical_best", "*", teal)):
            for index, row in enumerate(groups):
                if role in row["roles"].split("|"):
                    ax.scatter(x[index], row["median_ms"]*1000, marker=shape, c=color, s=68, zorder=5,
                               label=role.replace("_", " "))

    def errorbars(ax, rows, x, color=blue, **kwargs):
        y = [r["median_ms"]*1000 for r in rows]
        bounds = [[(r["median_ms"]-r["q1_ms"])*1000 for r in rows],
                  [(r["q3_ms"]-r["median_ms"])*1000 for r in rows]]
        ax.errorbar(x, y, yerr=bounds, fmt="o", color=color, markersize=4, capsize=2, **kwargs)
        ax.grid(axis="y", alpha=.22)
        ax.set_ylabel("Median time (µs)")

    fig, axes = plt.subplots(1, 2, figsize=(8, 2.65), layout="constrained")
    for ax, kernel, title in zip(axes, ("vadd", "compute"), ("Vector add", "Dependent FP32 FMA")):
        rows = selected(kernel, "1048583")
        x = list(range(1, len(rows)+1))
        errorbars(ax, rows, x)
        markers(ax, rows, x)
        ax.set_title(title)
        ax.set_xlabel("Launch ID (block/grid in configurations.csv)")
    axes[0].legend(loc="best", frameon=False, fontsize=7)
    save(fig, "03-representative-configurations")

    fig, ax = plt.subplots(figsize=(8, 3.0), layout="constrained")
    rows = selected("reduce", "16777216")
    for block, color in zip(("64", "128", "256", "512"), (gray, blue, "#77559d", "#283c55")):
        group = [r for r in rows if r["block_x"] == block]
        x = [int(r["grid_x"]) for r in group]
        errorbars(ax, group, x, color=color, linestyle="-", label=f"{block} threads")
    markers(ax, rows, [int(r["grid_x"]) for r in rows])
    ax.set_xticks([68, 136, 272])
    ax.set_xlabel("Stage 1 grid (CTAs) · both stages timed")
    ax.legend(ncol=2, frameon=False, fontsize=8)
    save(fig, "04-large-reduction")

    fig, ax = plt.subplots(figsize=(8, 3.05), layout="constrained")
    rows = selected("transform", "4096")
    for i, shape in enumerate(dict.fromkeys((r["block_x"], r["block_y"]) for r in rows)):
        group = [r for r in rows if (r["block_x"], r["block_y"]) == shape]
        x = [int(r["grid_x"])*int(r["grid_y"]) for r in group]
        errorbars(ax, group, x, color=(gray, blue, "#77559d", "#83712f", "#283c55")[i],
                  linestyle=":", label="×".join(shape))
    markers(ax, rows, [int(r["grid_x"])*int(r["grid_y"]) for r in rows])
    ax.set_xscale("log")
    ax.set_xlabel("Grid CTAs (log scale) · legend gives block dimensions")
    ax.legend(ncol=3, frameon=False, fontsize=8)
    save(fig, "05-large-transform")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regenerate validated configuration statistics and figures")
    parser.add_argument("--raw", type=Path, default=ROOT / "results/raw/phase4")
    parser.add_argument("--out", type=Path, default=ROOT / "results/summary/phase5")
    parser.add_argument("--figures", type=Path)
    args = parser.parse_args()
    cases, configs = analyze(args.raw, args.out)
    if args.figures:
        plots(cases, configs, args.figures)
