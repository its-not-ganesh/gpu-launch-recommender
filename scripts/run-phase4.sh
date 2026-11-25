#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C GIT_PAGER=cat

if [[ $# != 2 || ! $1 =~ ^[0-9a-f]{40}$ || ! $2 =~ ^(calibration|heldout)$ ]]; then
    echo 'usage: bash scripts/run-phase4.sh FROZEN_SOURCE_SHA calibration|heldout' >&2
    exit 2
fi
revision=$1
split=$2
out=build/phase4
binary=./build/gpu-launch
model=analytical-v1
seed=3175
warmups=5
rounds=11
batch=20
device=0
idle_limit=10

verify_source() {
    [[ $(hostname -s) == cuda2 && $(git rev-parse HEAD) == "$revision" ]]
    [[ -z $(git status --porcelain=v1 --untracked-files=all) ]]
    sha256sum -c "$out/binary.sha256" >/dev/null
}

phase3_meta=results/raw/phase3-gate/vadd.meta.txt
uuid=$(sed -n 's/^device_uuid=//p' "$phase3_meta")
[[ $uuid =~ ^GPU-[0-9a-f-]+$ ]]
unset CUDA_VISIBLE_DEVICES
export CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p "$out"/{accepted/calibration,accepted/heldout,superseded,pending}
verify_source

gpu_stats() {
    nvidia-smi --id="$uuid" --query-gpu=uuid,name,utilization.gpu,memory.used,temperature.gpu,clocks.current.graphics \
        --format=csv,noheader,nounits
}

gpu_pids() {
    nvidia-smi --id="$uuid" --query-compute-apps=pid --format=csv,noheader,nounits
}

monitor() {
    local log=$1 phase=$2 owner=$3 stats pids pid foreign='' timestamp
    timestamp=$(date -u +%FT%T.%3NZ)
    if ! stats=$(gpu_stats) || ! pids=$(gpu_pids); then
        printf 'time=%s phase=%s status=query_error\n' "$timestamp" "$phase" >> "$log"
        return 2
    fi
    for pid in $pids; do
        if [[ ! $pid =~ ^[0-9]+$ ]]; then
            printf 'time=%s phase=%s status=query_error\n' "$timestamp" "$phase" >> "$log"
            return 2
        fi
        if [[ $pid != "$owner" ]]; then foreign+=${foreign:+,}$pid; fi
    done

    local utilization
    utilization=$(cut -d, -f3 <<< "$stats" | tr -d ' ')
    if [[ $stats != "$uuid, NVIDIA GeForce RTX 2080 Ti,"* || ! $utilization =~ ^[0-9]+$ ]]; then
        printf 'time=%s phase=%s status=query_error\n' "$timestamp" "$phase" >> "$log"
        return 2
    fi
    printf 'time=%s phase=%s status=ok evaluator_pid=%s foreign_compute_pids=%s stats=%s\n' \
        "$timestamp" "$phase" "$owner" "${foreign:-none}" "$stats" >> "$log"
    [[ -z $foreign ]] || return 1
    if [[ $phase == pre ]] && (( utilization >= idle_limit )); then return 1; fi
}

run_case() {
    local kernel=$1 size=$2 case_id attempt timestamp run stage dest status issue eval_pid
    shift 2
    case_id="$split-$kernel-$size"
    if compgen -G "$out/accepted/$split/$case_id-*.csv" >/dev/null; then
        printf 'CAMPAIGN_CASE_EXISTS %s\n' "$case_id"
        return 0
    fi
    for attempt in 1 2 3; do
        verify_source
        timestamp=$(date -u +%Y%m%dT%H%M%S%NZ)
        run="$case_id-$timestamp-a$attempt"
        stage="$out/pending/$run"
        mkdir -- "$stage"
        local stem="$stage/$run"
        "$binary" recommend --kernel "$kernel" "$@" --device "$device" > "$stem.recommend.txt"
        grep -Fxq "device_uuid=$uuid" "$stem.recommend.txt"
        grep -Fxq 'device_index=0' "$stem.recommend.txt"
        grep -Fxq 'compute_capability=7.5' "$stem.recommend.txt"
        grep -Fxq "model_revision=$model" "$stem.recommend.txt"

        issue=''
        status=0
        for trial in {1..10}; do
            status=0
            monitor "$stem.gpu.log" pre none || status=$?
            if (( status == 0 )); then break; fi
            sleep 3
        done
        if (( status != 0 )); then
            issue=gpu_unavailable_before_run
        else
            printf 'CAMPAIGN_CASE_START %s %s\n' "$case_id" "$run"
            "$binary" evaluate --kernel "$kernel" "$@" --device "$device" --split "$split" \
                --seed "$seed" --warmups "$warmups" --rounds "$rounds" --batch "$batch" \
                --csv "$stem.csv" > "$stem.stdout.log" 2>&1 &
            eval_pid=$!
            while kill -0 "$eval_pid" 2>/dev/null; do
                if ! monitor "$stem.gpu.log" during "$eval_pid"; then issue=gpu_contention_or_query_error; fi
                sleep 0.25
            done
            status=0
            wait "$eval_pid" || status=$?
            if ! monitor "$stem.gpu.log" post none; then issue=gpu_contention_or_query_error; fi
            if (( status != 0 )); then issue="evaluator_exit_$status${issue:+_$issue}"; fi
            if [[ -z $issue ]]; then
                grep -Fxq "git_revision=$revision" "$stem.meta.txt"
                grep -Fxq 'git_dirty=false' "$stem.meta.txt"
                grep -Fxq "device_uuid=$uuid" "$stem.meta.txt"
                grep -Fxq 'run_status=passed' "$stem.meta.txt"
                verify_source
                dest="$out/accepted/$split"
                mv -- "$stage"/* "$dest/"
                rmdir -- "$stage"
                printf 'CAMPAIGN_CASE_PASS %s %s\n' "$case_id" "$run"
                cat "$dest/$run.stdout.log"
                return 0
            fi
        fi

        printf 'status=superseded\nreason=%s\n' "$issue" > "$stem.disposition.txt"
        mv -- "$stage" "$out/superseded/"
        printf 'CAMPAIGN_CASE_SUPERSEDED %s %s reason=%s\n' "$case_id" "$run" "$issue"
        if [[ $issue == evaluator_exit_* ]]; then
            cat "$out/superseded/$run/$run.stdout.log"
            return 1
        fi
        sleep 3
    done
    echo "GPU hardware or availability verification deferred $case_id" >&2
    return 1
}

if [[ $split == calibration ]]; then
    run_case vadd 262144 --n 262144
    run_case compute 131072 --n 131072
    run_case reduce 262144 --n 262144
    run_case transform 512x512 --width 512 --height 512
else
    run_case vadd 1048583 --n 1048583
    run_case vadd 16777216 --n 16777216
    run_case compute 32771 --n 32771
    run_case compute 1048583 --n 1048583
    run_case reduce 1048583 --n 1048583
    run_case reduce 16777216 --n 16777216
    run_case transform 1009x769 --width 1009 --height 769
    run_case transform 4096x3072 --width 4096 --height 3072
fi
printf 'CAMPAIGN_SPLIT_PASS %s\n' "$split"
