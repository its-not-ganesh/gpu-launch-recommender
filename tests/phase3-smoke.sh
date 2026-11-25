#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 1 || ! -d $1 ]]; then
    echo 'usage: bash tests/phase3-smoke.sh EXISTING_EMPTY_DIRECTORY' >&2
    exit 2
fi
out=$1
binary=./build/gpu-launch
revision=$(git rev-parse HEAD)
header=schema_version,run_id,split,kernel,n,width,height,seed,launch_id,roles,block_x,block_y,block_z,grid_x,grid_y,grid_z,shared_bytes,final_block_x,final_grid_x,final_shared_bytes,validation_status,error_code,round,order,warmups,batch_iters,elapsed_ms,ms_per_operation

for kernel in vadd compute reduce transform; do
    n=0 w=0 h=0
    case $kernel in
        vadd) n=65539; sizes=(--n "$n") ;;
        compute) n=4103; sizes=(--n "$n") ;;
        reduce) n=17411; sizes=(--n "$n") ;;
        transform) w=129 h=67; sizes=(--width "$w" --height "$h") ;;
    esac
    rec="$out/$kernel.recommend.txt"
    csv="$out/$kernel.csv"
    meta="$out/$kernel.meta.txt"
    "$binary" recommend --kernel "$kernel" "${sizes[@]}" --device 0 > "$rec"
    "$binary" evaluate --kernel "$kernel" "${sizes[@]}" --device 0 --split calibration \
        --seed 3175 --warmups 1 --rounds 3 --batch 5 --csv "$csv"
    [[ $(head -n 1 "$csv") == "$header" ]]
    grep -Fxq "git_revision=$revision" "$meta"
    grep -Fxq 'git_dirty=false' "$meta"
    grep -Fxq 'run_status=passed' "$meta"
    grep -Fxq 'model_revision=analytical-v1' "$meta"
    grep -Fxq 'device_name=NVIDIA GeForce RTX 2080 Ti' "$meta"
    grep -Fxq 'compute_capability=7.5' "$meta"
    grep -Eiq '^device_uuid=GPU-[a-f0-9-]+$' "$meta"
    grep -Eiq '^gpu_load_before=[0-9]' "$meta"
    grep -Eiq '^gpu_load_after=[0-9]' "$meta"

    awk -F, -v kernel="$kernel" -v n="$n" -v w="$w" -v h="$h" \
        -v block="$(sed -n 's/^blockDim=//p' "$rec")" -v grid="$(sed -n 's/^gridDim=//p' "$rec")" \
        -v configs="$(sed -n 's/^configuration_count=//p' "$meta")" \
        -v first="$(sed -n 's/^fixed_1_launch_id=//p' "$meta")" \
        -v second="$(sed -n 's/^fixed_2_launch_id=//p' "$meta")" \
        -v occupancy="$(sed -n 's/^occupancy_launch_id=//p' "$meta")" \
        -v recommended="$(sed -n 's/^recommended_launch_id=//p' "$meta")" \
        -v oracle="$(sed -n 's/^empirical_best_launch_id=//p' "$meta")" \
        -v oracle_ms="$(sed -n 's/^empirical_best_median_ms=//p' "$meta")" '
        function fail(reason) { print kernel ": " reason > "/dev/stderr"; failed = 1; exit 1 }
        function ceil_div(a,b) { return int((a + b - 1) / b) }
        BEGIN {
            expected["fixed_1"] = first; expected["fixed_2"] = second
            expected["occupancy"] = occupancy; expected["recommended"] = recommended
            expected["empirical_best"] = oracle
            for (role in expected) if (expected[role] !~ /^[1-9][0-9]*$/ || expected[role] > configs) fail(role " missing")
        }
        NR == 1 { next }
        {
            id = $9
            b = $11 "," $12 "," $13; g = $14 "," $15 "," $16
            key = b "," g "," $17 "," $18 "," $19 "," $20
            delta = $27 / $26 - $28
            if (NF != 28 || $1 != 1 || $3 != "calibration" || $4 != kernel || $8 != 3175 ||
                id < 1 || id > configs || $21 != "passed" || $22 != "" || $23 < 1 || $23 > 3 ||
                $24 < 1 || $24 > configs || $25 != 1 || $26 != 5 ||
                $27 + 0 <= 0 || $28 + 0 <= 0 || delta < -1e-9 || delta > 1e-9 ||
                seen[id, $23]++ || position[$23, $24]++) fail("invalid or repeated observation")
            if ((key in launch) && launch[key] != id) fail("one launch has multiple IDs")
            if ((id in geometry) && (geometry[id] != key || labels[id] != $10)) fail("launch ID changed")
            launch[key] = id; geometry[id] = key; labels[id] = $10
            values[id, $23] = $28 + 0
            if (n && ($5 != n || $6 != "" || $7 != "")) fail("wrong 1D workload")
            if (!n && ($5 != "" || $6 != w || $7 != h)) fail("wrong 2D workload")
            if (kernel == "reduce") {
                if ($17 != 4 * $11 || $18 != 256 || $19 != 1 || $20 != 1024) fail("invalid reduction stages")
            } else if ($17 != 0 || $18 != "" || $19 != "" || $20 != "") fail("unexpected final stage")

            fields = split($10, tokens, /[|]/)
            if (fields > 1) shared = 1
            for (i = 1; i <= fields; ++i) {
                role = tokens[i]
                if (role == "candidate") continue
                if (!(role in expected) || expected[role] != id || count[role, $23]++) fail("wrong role: " role)
            }
            if (id == recommended && (b != block || g != grid)) fail("recommendation differs from CLI")
            if (id == first && b != (n ? "128,1,1" : "16,8,1")) fail("wrong fixed_1 block")
            if (id == second && b != (n ? "256,1,1" : "16,16,1")) fail("wrong fixed_2 block")
            if (id == first || id == second || id == occupancy) {
                gx = n ? ceil_div(n, $11) : ceil_div(w, $11)
                if (kernel == "reduce" && gx > 272) gx = 272
                if ($14 != gx || $15 != (n ? 1 : ceil_div(h, $12)) || $16 != 1) fail("wrong baseline grid")
            }
        }
        END {
            if (failed) exit 1
            if (NR != configs * 3 + 1 || !shared) fail("sample count or shared roles")
            for (role in expected) for (round = 1; round <= 3; ++round) {
                if (count[role, round] != 1) fail("role not uniquely sampled: " role)
            }
            for (id = 1; id <= configs; ++id) {
                a = values[id, 1]; b = values[id, 2]; c = values[id, 3]
                if (a > b) { tmp = a; a = b; b = tmp }
                if (b > c) b = c
                median = a > b ? a : b
                if (id == 1 || median < fastest) { fastest = median; winner = id }
            }
            difference = fastest - oracle_ms
            if (winner != oracle || difference < -1e-12 || difference > 1e-12) fail("reported oracle differs from CSV medians")
            printf "PHASE3_SMOKE_PASS kernel=%s configurations=%d batch_samples=%d oracle_verified=yes\n", kernel, configs, NR-1
        }
    ' "$csv"
done
