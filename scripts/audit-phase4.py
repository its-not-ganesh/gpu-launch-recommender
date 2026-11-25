#!/usr/bin/env python3
"""Audit Phase 4 event batches and regenerate the workload summary."""

import argparse
import csv
import hashlib
import json
import math
import re
import shlex
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path


WORKLOADS = {
    "calibration": {"vadd": [(262144,)], "compute": [(131072,)], "reduce": [(262144,)],
                    "transform": [(512, 512)]},
    "heldout": {"vadd": [(1048583,), (16777216,)], "compute": [(32771,), (1048583,)],
                "reduce": [(1048583,), (16777216,)], "transform": [(1009, 769), (4096, 3072)]},
}
SEED, WARMUPS, ROUNDS, BATCH, SMS = 3175, 5, 11, 20, 68
SCHEMA = ("schema_version,run_id,split,kernel,n,width,height,seed,launch_id,roles,block_x,block_y,block_z,"
          "grid_x,grid_y,grid_z,shared_bytes,final_block_x,final_grid_x,final_shared_bytes,validation_status,"
          "error_code,round,order,warmups,batch_iters,elapsed_ms,ms_per_operation").split(",")
GEOMETRY = SCHEMA[10:20]
SELECTIONS = ("fixed_1", "fixed_2", "occupancy", "recommended", "empirical_best")
DEVICE = "NVIDIA GeForce RTX 2080 Ti"


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def fields(path):
    data = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, marker, value = line.partition("=")
        require(marker and key not in data, f"{path}: malformed or duplicate key {key}")
        data[key] = value
    return data


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_manifest(root):
    manifest = root / "SHA256SUMS"
    entries = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9._/-]+)", line)
        require(match is not None, f"bad manifest entry: {line}")
        checksum, name = match.groups()
        require(name not in entries, f"duplicate manifest name: {name}")
        entries[name] = checksum
    payload = {p.relative_to(root).as_posix(): p for p in root.rglob("*") if p.is_file() and p != manifest}
    require(entries.keys() == payload.keys(), "manifest does not match the raw evidence inventory")
    for name, path in payload.items():
        require(digest(path) == entries[name], f"checksum mismatch: {name}")
    return {"files": len(entries), "manifest_sha256": digest(manifest)}


def ceil_div(a, b):
    return (a + b - 1) // b


def shapes(kernel):
    if kernel == "transform":
        return ((8, 8), (16, 8), (32, 4), (16, 16), (32, 8))
    return ((64, 1), (128, 1), (256, 1), (512, 1))


def coverage(kernel, dims, bx, by):
    gx = min(ceil_div(dims[0], bx), 2147483647)
    gy = min(ceil_div(dims[1], by), 65535) if len(dims) == 2 else 1
    if kernel == "reduce":
        gx = min(gx, 4 * SMS)
    return gx, gy


def expected_geometries(kernel, dims):
    result = set()
    for bx, by in shapes(kernel):
        tx, ty = coverage(kernel, dims, bx, by)
        for budget in (SMS, 2 * SMS, 4 * SMS, tx * ty):
            gx = min(tx, budget)
            gy = min(ty, budget // gx)
            resources = (4 * bx, 256, 1, 1024) if kernel == "reduce" else (0, None, None, None)
            result.add((bx, by, 1, gx, gy, 1) + resources)
    return result


def iqr(values):
    ordered = sorted(values)

    def quantile(p):
        x = p * (len(ordered) - 1)
        lower = int(x)
        return ordered[lower] + (x - lower) * (ordered[math.ceil(x)] - ordered[lower])

    return quantile(0.75) - quantile(0.25)


def invocation_matches(meta, run_id, kernel, split, dims):
    args = shlex.split(meta["invocation"])
    require(args[:2] == ["./build/gpu-launch", "evaluate"] and len(args) % 2 == 0, "invalid invocation")
    flags = dict(zip(args[2::2], args[3::2]))
    csv_path = flags.pop("--csv")
    require(csv_path.startswith("build/phase4/") and Path(csv_path).stem == run_id,
            "invocation CSV was not staged under build/phase4")
    expected = {"--kernel": kernel, "--split": split, "--seed": str(SEED), "--warmups": str(WARMUPS),
                "--rounds": str(ROUNDS), "--batch": str(BATCH), "--device": "0"}
    expected.update({"--n": str(dims[0])} if len(dims) == 1
                    else {"--width": str(dims[0]), "--height": str(dims[1])})
    require(flags == expected and len(flags) * 2 + 4 == len(args), "incorrect invocation flags")


def gpu_monitor(path, uuid):
    records = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        prefix, marker, stats = line.partition(" stats=")
        info = dict(item.split("=", 1) for item in prefix.split())
        phase = info["phase"]
        require(phase in ("pre", "during", "post"), "unknown GPU monitor phase")
        datetime.fromisoformat(info["time"].replace("Z", "+00:00"))
        if info["status"] == "query_error":
            require(phase == "pre" and not marker, "accepted run lost GPU visibility")
            records[phase].append((info["time"], None, "unknown"))
            continue
        require(info["status"] == "ok" and marker, "invalid GPU monitor record")
        fields_ = next(csv.reader([stats], skipinitialspace=True))
        require(len(fields_) == 6 and fields_[:2] == [uuid, DEVICE], "monitored GPU differs")
        utilization = int(fields_[2])
        foreign = info["foreign_compute_pids"]
        require(0 <= utilization <= 100, "bad GPU utilization")
        if phase != "pre":
            require(foreign == "none", "accepted run has sampled GPU contention")
        records[phase].append((info["time"], utilization, foreign))
    require(all(records[p] for p in ("pre", "during", "post")), "incomplete GPU monitoring")
    before, after = records["pre"][-1], records["post"][-1]
    require(before[1] is not None and before[1] < 10 and before[2] == "none", "GPU was unavailable before run")
    require(before[0] <= records["during"][0][0] <= after[0], "GPU monitoring timestamps are unordered")
    return {
        "gpu_pre_utc": before[0], "gpu_post_utc": after[0], "preflight_retries": len(records["pre"]) - 1,
        "during_gpu_samples": len(records["during"]),
        "peak_sampled_gpu_util_pct": max(r[1] for r in records["during"]),
    }


def audit_case(path, protocol, split, kernel, dims):
    run_id = path.stem
    sidecar = lambda suffix: path.with_name(run_id + suffix)
    meta = fields(sidecar(".meta.txt"))
    expected_meta = {
        "schema_version": "1", "run_id": run_id, "split": split, "kernel": kernel, "elements": str(math.prod(dims)),
        "hostname": "cuda2.cims.nyu.edu", "device_index": "0", "device_uuid": protocol["device_uuid"],
        "device_name": DEVICE, "compute_capability": "7.5", "sm_count": str(SMS),
        "git_revision": protocol["source_revision"], "git_dirty": "false", "model_revision": "analytical-v1",
        "input_seed": str(SEED), "ordering_seed": str(SEED), "ordering_algorithm": "mt19937-std-shuffle",
        "warmups": str(WARMUPS), "rounds": str(ROUNDS), "batch_iters": str(BATCH), "run_status": "passed",
        "nvcc_version": "13.0.88", "cuda_runtime_version": "13000",
        "cuda_flags": "-O3 -std=c++17 -gencode arch=compute_75,code=sm_75 -lineinfo",
    }
    for key, value in expected_meta.items():
        require(meta.get(key) == value, f"metadata {key} differs: {meta.get(key)!r}")
    for key in ("utc", "cuda_driver_api_version", "host_compiler_version", "gpu_load_fields"):
        require(meta.get(key), f"missing metadata {key}")
    snapshots = [next(csv.reader([meta[key]], skipinitialspace=True)) for key in ("gpu_load_before", "gpu_load_after")]
    require(all(len(snapshot) == 5 and snapshot[0] and 0 <= int(snapshot[1]) <= 100 for snapshot in snapshots),
            "invalid qualitative GPU load snapshots")
    require(snapshots[0][0] == snapshots[1][0], "driver changed within one run")
    datetime.strptime(meta["utc"], "%Y-%m-%dT%H:%M:%SZ")
    invocation_matches(meta, run_id, kernel, split, dims)
    monitor = gpu_monitor(sidecar(".gpu.log"), protocol["device_uuid"])
    rec = fields(sidecar(".recommend.txt"))
    require(rec["kernel"] == kernel and rec["model_revision"] == "analytical-v1" and
            rec["device_uuid"] == protocol["device_uuid"] and rec["device_index"] == "0" and
            rec["elements"] == str(math.prod(dims)) and 0 < float(rec["score"]) < math.inf,
            "recommendation record is inconsistent")
    require(sidecar(".stdout.log").read_text(encoding="utf-8").startswith("EVALUATE_PASS "), "evaluator failed")

    launch_data, times, orders = {}, defaultdict(dict), defaultdict(dict)
    dims_csv = (str(dims[0]), "", "") if len(dims) == 1 else ("", str(dims[0]), str(dims[1]))
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames == SCHEMA, "CSV schema differs")
        for row in reader:
            require(None not in row and all(value is not None for value in row.values()), "incomplete CSV row")
            require((row["n"], row["width"], row["height"]) == dims_csv, "wrong workload dimensions")
            require((row["schema_version"], row["validation_status"], row["error_code"]) == ("1", "passed", ""),
                    "sample did not pass correctness")
            require((row["run_id"], row["split"], row["kernel"]) == (run_id, split, kernel), "wrong sample identity")
            require(tuple(int(row[k]) for k in ("seed", "warmups", "batch_iters")) == (SEED, WARMUPS, BATCH),
                    "sample timing protocol differs")
            launch, round_, order = (int(row[k]) for k in ("launch_id", "round", "order"))
            require(launch > 0 and 1 <= round_ <= ROUNDS and order > 0, "invalid sample index")
            geometry = tuple(int(row[key]) if row[key] else None for key in GEOMETRY)
            roles = tuple(row["roles"].split("|"))
            require(len(set(roles)) == len(roles) and set(roles) <= {"candidate", *SELECTIONS}, "invalid roles")
            require(launch_data.setdefault(launch, (geometry, roles)) == (geometry, roles), "launch or roles changed")
            require(round_ not in times[launch] and order not in orders[round_], "duplicate launch batch or position")
            elapsed, normalized = (float(row[k]) for k in ("elapsed_ms", "ms_per_operation"))
            require(0 < elapsed < math.inf and 0 < normalized < math.inf and
                    math.isclose(elapsed / BATCH, normalized, rel_tol=1e-13), "invalid event time or units")
            times[launch][round_] = normalized
            orders[round_][order] = launch

    count = int(meta["configuration_count"])
    ids, rounds = set(range(1, count + 1)), set(range(1, ROUNDS + 1))
    geometry_set = {entry[0] for entry in launch_data.values()}
    expected = expected_geometries(kernel, dims)
    require(set(launch_data) == ids and len(geometry_set) == count and geometry_set == expected,
            "missing or duplicate launch geometry")
    require(set(orders) == rounds and all(set(order) == ids and set(order.values()) == ids for order in orders.values())
            and all(set(batch) == rounds for batch in times.values()), "incomplete measurement rounds")
    require(len({tuple(v[k] for k in sorted(v)) for v in orders.values()}) > 1, "configuration order did not vary")

    selections = {role: int(meta[f"{role}_launch_id"]) for role in SELECTIONS}
    for role, launch in selections.items():
        require([i for i, (_, roles) in launch_data.items() if role in roles] == [launch], f"inconsistent {role} role")
    fixed_blocks = ((16, 8), (16, 16)) if len(dims) == 2 else ((128, 1), (256, 1))
    for role in ("fixed_1", "fixed_2", "occupancy"):
        geom = launch_data[selections[role]][0]
        require(geom[3:5] == coverage(kernel, dims, *geom[:2]), f"{role} used a non-coverage grid")
    for role, block in zip(("fixed_1", "fixed_2"), fixed_blocks):
        require(launch_data[selections[role]][0][:2] == block, f"incorrect {role} block")
    rec_id = selections["recommended"]
    rec_geo, rec_roles = launch_data[rec_id]
    require("candidate" in rec_roles and rec_geo[:3] == tuple(map(int, rec["blockDim"].split(","))) and
            rec_geo[3:6] == tuple(map(int, rec["gridDim"].split(","))) and rec_geo[6] == int(rec["shared_bytes"]),
            "CLI recommendation did not match the evaluated candidate")
    if kernel == "reduce":
        require((rec["final_blockDim"], rec["final_gridDim"], rec["final_shared_bytes"]) ==
                ("256,1,1", "1,1,1", "1024"), "CLI final reduction stage was inconsistent")

    medians = {i: statistics.median(times[i].values()) for i in ids}
    oracle = min(ids, key=lambda i: (medians[i], i))
    require(oracle == selections["empirical_best"] and medians[oracle] == float(meta["empirical_best_median_ms"]),
            "reported empirical best differs from measured medians")
    spreads = {i: iqr(times[i].values()) for i in ids}
    max_iqr_pct = max(100 * spreads[i] / medians[i] for i in ids)
    max_sample_ratio = max(max(times[i].values()) / medians[i] for i in ids)
    notes = []
    if max_iqr_pct > 20:
        notes.append("candidate_iqr_over_20_percent")
    if max_sample_ratio > 2:
        notes.append("batch_over_twice_candidate_median")
    if monitor["preflight_retries"]:
        notes.append("preflight_waited")
    summary = {
        "split": split, "kernel": kernel, "n": dims_csv[0], "width": dims_csv[1], "height": dims_csv[2],
        "run_id": run_id, "configurations": count, "valid_batches": count * ROUNDS,
        "recommended_block": rec["blockDim"], "recommended_grid": rec["gridDim"],
        **{f"{role}_launch_id": i for role, i in selections.items()},
        **{f"{role}_median_ms": medians[i] for role, i in selections.items()},
        **{f"{role}_iqr_ms": spreads[i] for role, i in selections.items()},
        **{f"speedup_vs_{role}": medians[selections[role]] / medians[rec_id]
           for role in ("fixed_1", "fixed_2", "occupancy")},
        "slowdown_vs_empirical_best_pct": 100 * (medians[rec_id] / medians[oracle] - 1),
        "max_candidate_iqr_pct": max_iqr_pct, "max_batch_to_candidate_median_ratio": max_sample_ratio,
        "quality_flags": "|".join(notes),
    }
    detail = {"run_id": run_id, "split": split, "kernel": kernel, "dimensions": dims,
              "configurations": count, "valid_batches": count * ROUNDS, "source_revision": meta["git_revision"],
              "selections": selections, "gpu_driver_version": snapshots[0][0], **monitor, "quality_flags": notes}
    return summary, detail


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--split", choices=tuple(WORKLOADS))
    parser.add_argument("--verify-manifest", action="store_true")
    args = parser.parse_args()
    protocol = fields(args.protocol or args.raw / "protocol.txt")
    require(all(re.fullmatch(r"[0-9a-f]{40}", protocol[key])
                for key in ("source_revision", "candidate_policy_blob", "analytical_model_blob")),
            "bad source freeze identifiers")
    require(re.fullmatch(r"[0-9a-f]{64}", protocol["binary_sha256"]), "missing frozen binary SHA-256")
    require(tuple(protocol[key] for key in ("model_revision", "device_index", "compute_capability")) ==
            ("analytical-v1", "0", "7.5"), "unexpected model or device freeze")
    require(tuple(int(protocol[key]) for key in ("seed", "warmups", "rounds", "batch_iters")) ==
            (SEED, WARMUPS, ROUNDS, BATCH), "unexpected protocol freeze")
    manifest = check_manifest(args.raw) if args.verify_manifest else None
    summaries, details = [], []
    for split in (args.split,) if args.split else WORKLOADS:
        selected = set()
        for kernel, workloads in WORKLOADS[split].items():
            for dims in workloads:
                label = "x".join(map(str, dims))
                matches = list((args.raw / split).glob(f"{split}-{kernel}-{label}-*.csv"))
                require(len(matches) == 1, f"{split}/{kernel}/{label}: expected exactly one accepted run")
                path = matches[0]
                selected.add(path)
                try:
                    summary, detail = audit_case(path, protocol, split, kernel, dims)
                except (ValueError, KeyError, OSError) as error:
                    raise ValueError(f"{path.name}: {error}") from error
                summaries.append(summary)
                details.append(detail)
        require(selected == set((args.raw / split).glob("*.csv")), f"unexpected accepted runs in {split}")
    if not args.split:
        require(max(d["gpu_post_utc"] for d in details if d["split"] == "calibration") <=
                min(d["gpu_pre_utc"] for d in details if d["split"] == "heldout"), "held-out ran before calibration")
        require(len({d["gpu_driver_version"] for d in details}) == 1, "driver changed across the campaign")
    superseded = []
    for path in sorted((args.raw / "superseded").glob("*/*.disposition.txt")):
        outcome = fields(path)
        superseded.append({"run_id": path.parent.name, "reason": outcome["reason"]})
    audit = {
        "status": "passed", "source_revision": protocol["source_revision"], "model_revision": protocol["model_revision"],
        "device": DEVICE, "device_uuid": protocol["device_uuid"], "seed": SEED, "warmups": WARMUPS,
        "rounds": ROUNDS, "batch_iters": BATCH, "cases": len(details),
        "configurations": sum(d["configurations"] for d in details),
        "valid_batches": sum(d["valid_batches"] for d in details),
        "manifest": manifest, "superseded": superseded, "workloads": details,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "workloads.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summaries)
    (args.out / "audit.json").write_bytes((json.dumps(audit, indent=2) + "\n").encode("utf-8"))
    print(f"AUDIT_PASS cases={audit['cases']} configs={audit['configurations']} batches={audit['valid_batches']}")
    for summary in summaries:
        print(summary["run_id"], f"configs={summary['configurations']}",
              f"slowdown_pct={summary['slowdown_vs_empirical_best_pct']:.3f}",
              f"max_iqr_pct={summary['max_candidate_iqr_pct']:.3f}",
              f"max_batch_ratio={summary['max_batch_to_candidate_median_ratio']:.3f}",
              f"flags={summary['quality_flags'] or 'none'}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError) as exc:
        raise SystemExit(str(exc))
