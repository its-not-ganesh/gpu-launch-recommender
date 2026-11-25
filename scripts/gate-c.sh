#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
unset CUDA_VISIBLE_DEVICES
export CUDA_DEVICE_ORDER=PCI_BUS_ID
[[ -f readme.txt && -f SHA256SUMS && ! -e .git && ! -e build ]]
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo 'Gate C needs an isolated archive extraction outside current Git worktrees.' >&2
    exit 1
fi

sha256sum --quiet -c SHA256SUMS
set +u
module load cuda-13.0
set -u
hostname
nvcc --version
make clean
make CUDA_ARCH=75
make check

mkdir -p build/submission-check
out=build/submission-check
build/gpu-launch recommend --kernel reduce --n 16777216 --device 0 | tee "$out/reduce.recommend.txt"
build/gpu-launch recommend --kernel transform --width 4096 --height 3072 --device 0 | tee "$out/transform.recommend.txt"
grep -Fx 'blockDim=512,1,1' "$out/reduce.recommend.txt"
grep -Fx 'gridDim=68,1,1' "$out/reduce.recommend.txt"
grep -Fx 'final_blockDim=256,1,1' "$out/reduce.recommend.txt"
grep -Fx 'blockDim=32,8,1' "$out/transform.recommend.txt"
grep -Fx 'gridDim=128,2,1' "$out/transform.recommend.txt"
build/gpu-launch evaluate --kernel vadd --n 4103 --device 0 --split calibration --seed 3175 --warmups 1 --rounds 3 --batch 5 --csv "$out/vadd.csv"
build/gpu-launch evaluate --kernel transform --width 129 --height 67 --device 0 --split calibration --seed 3175 --warmups 1 --rounds 3 --batch 5 --csv "$out/transform.csv"
python3 scripts/audit-phase4.py results/raw/phase4 "$out/audit" --verify-manifest
python3 scripts/analyze.py --out "$out/tables"
for file in audit.json workloads.csv; do cmp "results/summary/phase4/$file" "$out/audit/$file"; done
for file in audit.json workloads.csv configurations.csv aggregates.json failure_cases.csv; do
    cmp "results/summary/phase5/$file" "$out/tables/$file"
done

pdfinfo report/Final_Report.pdf | grep -E '^(Pages|PDF version):'
pdftotext -layout report/Final_Report.pdf "$out/report.txt"
python3 - "$out" <<'PYTHON'
import csv
import importlib.util
import json
import math
import statistics
import sys
from pathlib import Path

out = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("audit", "scripts/audit-phase4.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
for kernel in ("vadd", "transform"):
    meta = audit.fields(out / (kernel + ".meta.txt"))
    assert meta["run_status"] == "passed" and meta["kernel"] == kernel
    assert meta["device_name"] == "NVIDIA GeForce RTX 2080 Ti" and meta["compute_capability"] == "7.5"
    assert meta["git_revision"] == "unavailable" and meta["git_dirty"] == "unknown"
    assert meta["model_revision"] == "analytical-v1" and meta["gpu_load_before"] != "unavailable"
    count = int(meta["configuration_count"])
    batches = {}
    with (out / (kernel + ".csv")).open(newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == audit.SCHEMA
        for row in reader:
            assert row["validation_status"] == "passed" and not row["error_code"]
            assert row["batch_iters"] == "5" and row["kernel"] == kernel
            key = int(row["launch_id"]), int(row["round"])
            ms = float(row["ms_per_operation"])
            assert key not in batches and ms > 0 and math.isfinite(ms)
            assert math.isclose(ms, float(row["elapsed_ms"])/5, rel_tol=1e-13)
            batches[key] = ms
    assert set(batches) == {(i, r) for i in range(1, count+1) for r in range(1, 4)}
    medians = {i: statistics.median(batches[i, r] for r in range(1, 4)) for i in range(1, count+1)}
    best = min(medians, key=lambda i: (medians[i], i))
    assert best == int(meta["empirical_best_launch_id"]) and medians[best] == float(meta["empirical_best_median_ms"])
    print(f"EXTRACTED_EVALUATION_PASS kernel={kernel} candidates={count} batches={len(batches)}")

report = (out / "report.txt").read_text(encoding="utf-8")
sections = ("Abstract", "Introduction", "Literature Survey", "Proposed Idea", "Experimental Setup",
            "Experiments & Analysis", "Conclusions", "References")
assert all(section in report for section in sections)
assert all(f"Figure {i}." in report for i in range(1, 6))
assert all(len(page.strip()) > 120 for page in report.split("\f") if page)
aggregates = json.loads((out / "tables/aggregates.json").read_text())
held = aggregates["heldout"]
summary = audit.fields(Path("results/raw/phase4/protocol.txt"))
for value in ("185", "2,035", f"{held['median_slowdown_pct']:.2f}%", f"{held['max_slowdown_pct']:.2f}%",
              "21.61%", "191.80", "130.76", "230.50", "189.54", summary["source_revision"], "10.1109/MCSoC.2015.10"):
    assert value in report, f"report is missing verified value {value}"
for image in ("01-heldout-slowdown", "02-baseline-comparison", "03-representative-configurations",
              "04-large-reduction", "05-large-transform"):
    assert (Path("results/figures") / (image + ".png")).read_bytes().startswith(b"\x89PNG")
    assert "<svg" in (Path("results/figures") / (image + ".svg")).read_text(encoding="utf-8")
print("REPORT_AND_DATA_PASS sections=8 figures=5")
PYTHON
echo 'GATE_C_PASS isolated_zip=yes cuda_build=yes correctness=yes analysis=yes report=yes'
