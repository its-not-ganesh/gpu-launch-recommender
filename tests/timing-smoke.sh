#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 1 || ! -d $1 ]]; then
    echo 'usage: bash tests/timing-smoke.sh EXISTING_EMPTY_DIRECTORY' >&2
    exit 2
fi
out=$1
binary=./build/gpu-launch
reduce=(evaluate --kernel reduce --n 17411 --device 0 --split calibration --seed 3175 --warmups 2 --rounds 3 --batch 5)
"$binary" "${reduce[@]}" --csv "$out/reduce-a.csv"
"$binary" "${reduce[@]}" --csv "$out/reduce-b.csv"
"$binary" evaluate --kernel transform --width 17 --height 19 --device 0 --split calibration \
    --seed 3175 --warmups 1 --rounds 1 --batch 5 --csv "$out/transform.csv"

header=schema_version,run_id,split,kernel,n,width,height,seed,launch_id,roles,block_x,block_y,block_z,grid_x,grid_y,grid_z,shared_bytes,final_block_x,final_grid_x,final_shared_bytes,validation_status,error_code,round,order,warmups,batch_iters,elapsed_ms,ms_per_operation
for file in "$out/reduce-a.csv" "$out/reduce-b.csv" "$out/transform.csv"; do
    [[ $(head -n 1 "$file") == "$header" ]]
    name=$(basename "$file" .csv)
    meta="$out/$name.meta.txt"
    if [[ $name == transform ]]; then
        configurations=5
        rounds=1
        warmups=1
    else
        configurations=9
        rounds=3
        warmups=2
    fi
    awk -F, -v name="$name" -v configs="$configurations" -v rounds="$rounds" -v warmups="$warmups" '
        NR == 1 { next }
        {
            key = $11 FS $12 FS $13 FS $14 FS $15 FS $16
            delta = $27 / $26 - $28
            if (NF != 28 || $1 != 1 || $2 != "\"" name "\"" || $3 != "calibration" ||
                $8 != 3175 || $9 < 1 || $9 > configs || $10 !~ /^candidate([|][a-z_12]+)*$/ ||
                $21 != "passed" || $22 != "" || $23 < 1 || $23 > rounds ||
                $24 < 1 || $24 > configs || $25 != warmups || $26 != 5 ||
                $27 + 0 <= 0 || $28 + 0 <= 0 || delta < -1e-9 || delta > 1e-9 ||
                seen[$23, $9]++ || position[$23, $24]++) exit 1
            if ((key in launch) && launch[key] != $9) exit 1
            if (($9 in geometry) && geometry[$9] != key) exit 1
            launch[key] = $9; geometry[$9] = key
            if (name == "transform") {
                if ($4 != "transform" || $5 != "" || $6 != 17 || $7 != 19 ||
                    $17 != 0 || $18 != "" || $19 != "" || $20 != "") exit 1
            } else {
                if ($4 != "reduce" || $5 != 17411 || $6 != "" || $7 != "" || $17 != 4*$11 ||
                    $18 != 256 || $19 != 1 || $20 != 1024) exit 1
            }
        }
        END { if (NR != configs * rounds + 1) exit 1 }
    ' "$file"
    grep -Fx "run_id=$name" "$meta"
    grep -Fx 'device_name=NVIDIA GeForce RTX 2080 Ti' "$meta"
    grep -Fx 'compute_capability=7.5' "$meta"
    grep -Fx 'sm_count=68' "$meta"
    grep -Fx 'input_seed=3175' "$meta"
    grep -Fx 'ordering_seed=3175' "$meta"
    grep -Fx "configuration_count=$configurations" "$meta"
    grep -Fx "git_revision=$(git rev-parse HEAD)" "$meta"
    grep -Fx 'git_dirty=false' "$meta"
    grep -E '^gpu_load_before=[0-9]' "$meta"
    grep -E '^gpu_load_after=[0-9]' "$meta"
    grep -Fx 'run_status=passed' "$meta"
done

sed 's/|empirical_best//g' "$out/reduce-a.csv" | cut -d, -f3-26 > "$out/reduce-a.order"
sed 's/|empirical_best//g' "$out/reduce-b.csv" | cut -d, -f3-26 > "$out/reduce-b.order"
cmp "$out/reduce-a.order" "$out/reduce-b.order"

expect_error() {
    local expected=$1 message=$2 actual=0
    shift 2
    "$@" > "$out/expected-error.log" 2>&1 || actual=$?
    if [[ $actual != "$expected" ]] || ! grep -Fq -- "$message" "$out/expected-error.log"; then
        cat "$out/expected-error.log"
        return 1
    fi
}
old=$(sha256sum "$out/reduce-a.csv" "$out/reduce-a.meta.txt")
expect_error 1 'File exists' "$binary" "${reduce[@]}" --csv "$out/reduce-a.csv"
[[ $old == "$(sha256sum "$out/reduce-a.csv" "$out/reduce-a.meta.txt")" ]]
printf 'pre-existing sidecar\n' > "$out/conflict.meta.txt"
expect_error 1 'File exists' "$binary" "${reduce[@]}" --csv "$out/conflict.csv"
[[ ! -e $out/conflict.csv && $(cat "$out/conflict.meta.txt") == 'pre-existing sidecar' ]]
expect_error 2 'invalid or infeasible explicit launch' "$binary" check --kernel reduce --n 17 --block 63 --grid 1
expect_error 2 'W*H exceeds' "$binary" check --kernel transform --width 16777216 --height 16777216
echo TIMING_SMOKE_PASS reduce_samples=54 transform_samples=5 ordering_reproduced=yes
