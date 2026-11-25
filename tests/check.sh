#!/usr/bin/env bash
set -euo pipefail
binary=./build/gpu-launch
for kernel in vadd compute reduce; do
    for n in 1 31 32 33 257 4103 139267; do
        "$binary" check --kernel "$kernel" --n "$n"
    done
done
for size in 1x1 31x33 129x67 1009x769; do
    "$binary" check --kernel transform --width "${size%x*}" --height "${size#*x}"
done
"$binary" check --kernel transform --width 129 --height 67 --block 8,8 --grid 1,1
"$binary" check --kernel reduce --n 139267 --block 512 --grid 272
echo BENCHMARK_CORRECTNESS_PASS
