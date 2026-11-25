#!/usr/bin/env bash
set -euo pipefail

binary=./build/gpu-launch
guard=./build/recommend-guard
out=$(mktemp -d build/recommend-check.XXXXXX)
trap 'rm -f -- "$out"/event.csv "$out"/event.meta.txt; rmdir -- "$out"' EXIT
unset GPU_LAUNCH_GUARD_CONTROL

for kernel in vadd compute reduce transform; do
    if [[ $kernel == transform ]]; then
        sizes=(--width 129 --height 67)
    else
        sizes=(--n 4103)
    fi
    output=$("$binary" recommend --kernel "$kernel" "${sizes[@]}")
    [[ $output == "$("$binary" recommend --kernel "$kernel" "${sizes[@]}")" ]]
    [[ $output == "$("$guard" recommend --kernel "$kernel" "${sizes[@]}")" ]]
    grep -Fxq "kernel=$kernel" <<< "$output"
    grep -Fxq 'model_revision=analytical-v1' <<< "$output"
    grep -Eiq '^device_uuid=GPU-[a-f0-9-]+$' <<< "$output"
    awk -F= '
       $1 == "score" { if ($2 !~ /^[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$/ || $2 + 0 <= 0) exit 1; score++ }
       $1 == "estimated_occupancy" { if ($2 + 0 <= 0 || $2 + 0 > 1) exit 1; occupancy++ }
       $1 == "resident_blocks_per_sm" { if ($2 + 0 <= 0) exit 1; residency++ }
       END { if (score != 1 || occupancy != 1 || residency != 1) exit 1 }
    ' <<< "$output"
    block=$(sed -n 's/^blockDim=//p' <<< "$output")
    grid=$(sed -n 's/^gridDim=//p' <<< "$output")
    if [[ $kernel == transform ]]; then
        block=${block%,1}
        grid=${grid%,1}
    else
        block=${block%,1,1}
        grid=${grid%,1,1}
    fi
    if [[ $kernel == reduce ]]; then
        grep -Fxq 'final_blockDim=256,1,1' <<< "$output"
        grep -Fxq 'final_gridDim=1,1,1' <<< "$output"
        grep -Fxq 'final_shared_bytes=1024' <<< "$output"
    fi
    "$binary" check --kernel "$kernel" "${sizes[@]}" --block "$block" --grid "$grid"
done

# Positive controls prove the guarded binary detects the real runtime calls.
for api in cudaStreamCreateWithFlags __cudaLaunchKernel cudaEventCreate; do
    status=0
    if [[ $api == cudaStreamCreateWithFlags ]]; then
        message=$("$guard" check --kernel vadd --n 33 2>&1) || status=$?
    elif [[ $api == __cudaLaunchKernel ]]; then
        message=$(GPU_LAUNCH_GUARD_CONTROL=kernel "$guard" check --kernel vadd --n 33 2>&1) || status=$?
    else
        message=$(GPU_LAUNCH_GUARD_CONTROL=event "$guard" evaluate --kernel vadd --n 33 --split calibration \
            --warmups 1 --rounds 1 --batch 5 --csv "$out/event.csv" 2>&1) || status=$?
    fi
    if [[ $status != 86 ]] || ! grep -Fxq "RECOMMEND_GUARD intercepted $api" <<< "$message"; then
        printf '%s\n' "$message" >&2
        echo "recommendation guard failed positive control for $api (status $status)" >&2
        exit 1
    fi
done
echo RECOMMEND_PASS kernels=4 deterministic=yes benchmark_and_timing_guards=yes
