#!/usr/bin/env bash
# Refit both encoders with the published split on user-owned V100 hardware.
set -euo pipefail

usage() {
    printf 'Usage: bash %s RUN_DIR [--gpus 0[,1]] [--author-replay]\n' "$0"
    printf 'RUN_DIR must be new. Use one V100 for serial seeds or two V100s for parallel seeds.\n'
}

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    usage
    exit 0
fi
if (( $# == 0 )); then
    usage >&2
    exit 2
fi

requested_dir=$1
shift
gpus=0
author_replay=false
while (( $# )); do
    case "$1" in
        --gpus)
            if (( $# < 2 )); then usage >&2; exit 2; fi
            gpus=$2
            shift 2
            ;;
        --author-replay)
            author_replay=true
            shift
            ;;
        *) usage >&2; exit 2 ;;
    esac
done
if [[ ! $gpus =~ ^(0|[1-9][0-9]*)(,(0|[1-9][0-9]*))?$ ]]; then
    printf 'Use one or two comma-separated numeric GPU indices.\n' >&2
    exit 2
fi
IFS=, read -r gpu_primary gpu_secondary <<< "$gpus"
if [[ $gpu_primary == "$gpu_secondary" ]]; then
    printf 'Choose two distinct GPU indices for parallel training.\n' >&2
    exit 2
fi

tools_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$tools_dir/../.." && pwd)
if [[ $repo_dir == *:* || $requested_dir == *:* || $PWD == *:* ]]; then
    printf 'Docker bind mount paths must not contain a colon. Choose another checkout or run directory.\n' >&2
    exit 2
fi
programs="$repo_dir/public-programs"
data="$repo_dir/data/Patent-CE"
lf_code="$programs/paper-reproduction"
command -v docker >/dev/null
command -v python3 >/dev/null
docker info --format '{{.ServerVersion}}'
mkdir -p -- "$(dirname -- "$requested_dir")"
mkdir -- "$requested_dir"
run_dir=$(cd -- "$requested_dir" && pwd)
longformer="$run_dir/longformer"
modernbert="$run_dir/modernbert"
mb_code="$run_dir/modernbert-source"
cache="$run_dir/cache"
mkdir -- "$longformer" "$modernbert" "$cache"
run_id=$(python3 -c 'import uuid; print(uuid.uuid4().hex)')
run_label="patent-ce.reproduction=$run_id"
parent_image="patent-ce-reproduction-base:$run_id"
lf_image="patent-ce-reproduction-longformer:$run_id"
mb_image="patent-ce-reproduction-modernbert:$run_id"

stop_training() {
    trap - INT TERM
    local -a containers=()
    mapfile -t containers < <(docker ps --quiet --filter "label=$run_label")
    if (( ${#containers[@]} )); then
        docker stop --timeout 330 "${containers[@]}" >/dev/null || true
    fi
    exit 130
}
trap stop_training INT TERM

run_container() {
    local image=$1 code=$2 output=$3 gpu=$4
    shift 4
    local -a args=(--rm --init --label "$run_label" --user "$(id -u):$(id -g)" --shm-size 8g
        -e OMP_NUM_THREADS=6 -e HF_HOME=/cache/huggingface -e XDG_CACHE_HOME=/cache
        -e HOME=/tmp -e USER=encoder -e LOGNAME=encoder
        -e TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor -e TRITON_CACHE_DIR=/tmp/triton
        -v "$data:/data:ro" -v "$code:/work/code:ro" -v "$output:/output" -v "$cache:/cache")
    if [[ $gpu != cpu ]]; then args+=(--gpus "device=$gpu"); fi
    if [[ $image == "$mb_image" ]]; then args+=(-v "$longformer:/baseline:ro"); fi
    docker run "${args[@]}" "$image" "$@"
}

save_environment() {
    local image=$1 code=$2 output=$3 name=$4 transformer_version=$5 tokenizer_version=$6
    docker image inspect "$image" --format '{{.Id}}' > "$output/image_id.txt"
    run_container "$image" "$code" "$output" "$gpu_primary" -c '
import json
from pathlib import Path
import platform
import sys
import tokenizers
import torch
import transformers

name, transformer_version, tokenizer_version = sys.argv[1:]
assert torch.__version__ == "2.7.1+cu126" and torch.version.cuda == "12.6"
assert transformers.__version__ == transformer_version and tokenizers.__version__ == tokenizer_version
assert torch.cuda.is_available() and torch.cuda.get_device_capability() == (7, 0), "A V100 GPU is required"
lock = Path(f"/opt/{name}/requirements.lock.txt").read_bytes()
Path("/output/requirements.lock.txt").write_bytes(lock)
record = {
    "python": platform.python_version(), "torch": torch.__version__, "torch_cuda": torch.version.cuda,
    "transformers": transformers.__version__, "tokenizers": tokenizers.__version__,
    "gpu": torch.cuda.get_device_name(), "cuda_arch_list": torch.cuda.get_arch_list(),
    "requirements_lock": lock.decode("utf-8"),
}
Path("/output/runtime_environment.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
' "$name" "$transformer_version" "$tokenizer_version"
}

start_deadline() {
    python3 - "$1" <<'PY'
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import sys

folder = Path(sys.argv[1])
hours = json.loads((folder / "experiment_config.json").read_bytes())["runtime"]["budget_hours"]
started = datetime.now(timezone.utc)
deadline = started + timedelta(hours=hours)
clock = {"started_at": started.isoformat(), "deadline_utc": deadline.isoformat(), "budget_seconds": hours * 3600}
(folder / "run_clock.json").write_text(json.dumps(clock, indent=2) + "\n", encoding="utf-8")
print(deadline.isoformat())
PY
}

train_seed() {
    local image=$1 code=$2 output=$3 script=$4 deadline=$5 gpu=$6 seed=$7
    run_container "$image" "$code" "$output" "$gpu" "$script" \
        --output-dir /output --stage train --seed "$seed" --deadline-utc "$deadline" 2>&1 | tee "$output/seed_${seed}.log"
}

train_seeds() {
    if [[ -z $gpu_secondary ]]; then
        train_seed "$@" "$gpu_primary" 42
        train_seed "$@" "$gpu_primary" 43
        return
    fi
    local first second status=0
    train_seed "$@" "$gpu_primary" 42 &
    first=$!
    train_seed "$@" "$gpu_secondary" 43 &
    second=$!
    wait "$first" || status=$?
    wait "$second" || status=$?
    return "$status"
}

docker build -t "$parent_image" "$programs/encoder-ranking"
docker build --build-arg "BASE_IMAGE=$parent_image" -t "$lf_image" "$lf_code"
docker build --build-arg "BASE_IMAGE=$parent_image" -t "$mb_image" "$programs/paper-modernbert"

cp -- "$lf_code/experiment_config.json" "$longformer/experiment_config.json"
lf_deadline=$(start_deadline "$longformer")
save_environment "$lf_image" "$lf_code" "$longformer" paper-reproduction 4.46.2 0.20.3
run_container "$lf_image" "$lf_code" "$longformer" cpu prepare_original.py --data-dir /data --output-dir /output
run_container "$lf_image" "$lf_code" "$longformer" "$gpu_primary" run_reproduction.py \
    --output-dir /output --stage preflight --seed 42 --deadline-utc "$lf_deadline"
train_seeds "$lf_image" "$lf_code" "$longformer" run_reproduction.py "$lf_deadline"
if "$author_replay"; then
    run_container "$lf_image" "$lf_code" "$longformer" "$gpu_primary" run_reproduction.py \
        --output-dir /output --stage author --seed 42 --deadline-utc "$lf_deadline"
    run_container "$lf_image" "$lf_code" "$longformer" cpu report.py --data-dir /data --output-dir /output
    run_container "$lf_image" "$lf_code" "$longformer" cpu independent_verify.py --data-dir /data --output-dir /output
fi

python3 "$tools_dir/bind_baseline.py" --baseline-dir "$longformer" --data-dir "$data" --output-code-dir "$mb_code"
mb_inputs=(--baseline-dir /baseline --data-dir /data --output-dir /output)
save_environment "$mb_image" "$mb_code" "$modernbert" paper-modernbert 4.50.3 0.21.1
run_container "$mb_image" "$mb_code" "$modernbert" cpu prepare_inputs.py "${mb_inputs[@]}" --config /work/code/experiment_config.json
mb_deadline=$(start_deadline "$modernbert")
run_container "$mb_image" "$mb_code" "$modernbert" "$gpu_primary" run_modernbert.py \
    --output-dir /output --stage preflight --seed 42 --deadline-utc "$mb_deadline"
run_container "$mb_image" "$mb_code" "$modernbert" "$gpu_primary" replay_longformer.py "${mb_inputs[@]}"
train_seeds "$mb_image" "$mb_code" "$modernbert" run_modernbert.py "$mb_deadline"
run_container "$mb_image" "$mb_code" "$modernbert" cpu report.py "${mb_inputs[@]}"
run_container "$mb_image" "$mb_code" "$modernbert" cpu independent_verify.py "${mb_inputs[@]}"
printf 'Fresh comparison report: %s\n' "$modernbert/report.html"
