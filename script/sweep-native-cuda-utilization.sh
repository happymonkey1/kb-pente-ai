#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repository_root="$(cd -- "${script_dir}/.." && pwd -P)"
cd -- "${repository_root}"

usage() {
    cat <<'EOF'
Sweep native CUDA self-play across active-game/native-worker profiles.
Defaults: Standard19, model 6/128/256, 512 games (--games/SWEEP_GAMES), 64 simulations, one learner
step, and profiles 256:4 256:8 256:16 384:8 512:8:1 512:8:2
(active:workers[:cohorts], where omitted cohorts means one).

Options (also available as SWEEP_* environment variables):
  --games N                number of self-play games and main.py --batch-games value (default: 512)
  --profiles LIST          space- or comma-separated active:workers[:cohorts] profiles
  --prefix NAME            unique output prefix
  --checkpoint PATH        shared starting training checkpoint
  --resume-replay PATH     replay snapshot paired with --checkpoint
  --final-iteration N      final main.py training iteration target
  --python PATH            Python executable (default: .venv/bin/python)
  --compile / --no-compile compile the CUDA model (default: --compile)
  --dry-run                validate and print commands without running them
  -h, --help               show this help

With --checkpoint, all three checkpoint, replay, and final-iteration options
are required. The final target must exceed the checkpoint's stored iteration.
Every profile uses the strict self-play health failure policy so a threshold
breach fails the verification sweep.
EOF
}

die() { printf 'error: %s\n' "$*" >&2; exit 64; }
value() { (($# >= 2)) || die "$1 requires a value"; printf '%s' "$2"; }
file_path() {
    local label=$1 candidate=$2
    [[ -f ${candidate} ]] || die "${label} is not a regular file: ${candidate}"
    printf '%s/%s\n' "$(cd -- "$(dirname -- "${candidate}")" && pwd -P)" "$(basename -- "${candidate}")"
}
python_path() {
    local candidate=$1 resolved
    if [[ ${candidate} == */* ]]; then
        [[ -x ${candidate} ]] || die "Python executable is not executable: ${candidate}"
        resolved="$(cd -- "$(dirname -- "${candidate}")" && pwd -P)/$(basename -- "${candidate}")"
    else
        resolved="$(command -v "${candidate}" || true)"
        [[ -x ${resolved} ]] || die "Python executable was not found: ${candidate}"
    fi
    printf '%s' "${resolved}"
}

checkpoint="${SWEEP_CHECKPOINT:-}"
resume_replay="${SWEEP_RESUME_REPLAY:-}"
final_iteration="${SWEEP_FINAL_ITERATION:-1}"
games="${SWEEP_GAMES:-512}"
profiles="${SWEEP_PROFILES:-256:4 256:8 256:16 384:8 512:8:1 512:8:2}"
prefix="${SWEEP_PREFIX:-native-cuda-sweep-$(date -u '+%Y%m%dT%H%M%SZ')-$$}"
python_bin="${SWEEP_PYTHON:-${repository_root}/.venv/bin/python}"
compile_model="${SWEEP_COMPILE:-1}"
dry_run=0; checkpoint_supplied=0; final_iteration_supplied=0
[[ -n ${SWEEP_CHECKPOINT:-} ]] && checkpoint_supplied=1
[[ -n ${SWEEP_FINAL_ITERATION:-} ]] && final_iteration_supplied=1

while (($# > 0)); do
    case $1 in
        --profiles) profiles="$(value "$@")"; shift 2;;
        --prefix) prefix="$(value "$@")"; shift 2;;
        --checkpoint|--starting-checkpoint) checkpoint="$(value "$@")"; checkpoint_supplied=1; shift 2;;
        --resume-replay) resume_replay="$(value "$@")"; shift 2;;
        --final-iteration) final_iteration="$(value "$@")"; final_iteration_supplied=1; shift 2;;
        --games) games="$(value "$@")"; shift 2;;
        --python) python_bin="$(value "$@")"; shift 2;;
        --compile) compile_model=1; shift;;
        --no-compile) compile_model=0; shift;;
        --dry-run) dry_run=1; shift;;
        -h|--help) usage; exit 0;;
        *) die "unknown option: $1";;
    esac
done

[[ ${compile_model} == 0 || ${compile_model} == 1 ]] || die "SWEEP_COMPILE must be 0 or 1"
[[ ${prefix} =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die "invalid prefix: ${prefix}"
[[ ${final_iteration} =~ ^[1-9][0-9]*$ ]] || die "final-iteration must be positive: ${final_iteration}"
[[ ${games} =~ ^[1-9][0-9]*$ ]] || die "games must be a positive integer: ${games}"
if (( checkpoint_supplied )); then
    (( final_iteration_supplied )) || die "--checkpoint requires --final-iteration"
    [[ -n ${resume_replay} ]] || die "--checkpoint requires --resume-replay"
    checkpoint="$(file_path checkpoint "${checkpoint}")"
    resume_replay="$(file_path resume-replay "${resume_replay}")"
else
    [[ -z ${resume_replay} ]] || die "--resume-replay requires --checkpoint"
fi
python_bin="$(python_path "${python_bin}")"

profiles="${profiles//,/ }"
read -r -a profile_values <<< "${profiles}"
(( ${#profile_values[@]} > 0 )) || die "profile list cannot be empty"
declare -a active_values=() worker_values=()
declare -a cohort_values=() explicit_cohort_values=()
declare -A seen_profiles=()
for profile in "${profile_values[@]}"; do
    [[ ${profile} =~ ^([1-9][0-9]*):([1-9][0-9]*)(:([1-9][0-9]*))?$ ]] || die "invalid profile: ${profile}"
    active="${BASH_REMATCH[1]}"; workers="${BASH_REMATCH[2]}"
    explicit_cohort=0; cohorts=1
    if [[ -n ${BASH_REMATCH[3]} ]]; then
        explicit_cohort=1; cohorts="${BASH_REMATCH[4]}"
    fi
    (( cohorts == 1 || cohorts == 2 )) || die "cohorts must be 1 or 2: ${profile}"
    profile_key="${active}:${workers}:${cohorts}"
    [[ -z ${seen_profiles[${profile_key}]+present} ]] || die "duplicate profile: ${profile}"
    seen_profiles[${profile_key}]=1
    (( active <= games )) || die "active-games must not exceed games=${games}: ${active}"
    if (( cohorts == 2 )); then
        (( active >= 2 )) || die "cohorts=2 requires active-games >= 2: ${profile}"
        (( workers >= 2 )) || die "cohorts=2 requires native-workers >= 2: ${profile}"
    fi
    active_values+=("${active}"); worker_values+=("${workers}")
    cohort_values+=("${cohorts}"); explicit_cohort_values+=("${explicit_cohort}")
done

metrics_root="${repository_root}/metrics"; replays_root="${repository_root}/replays"; logs_root="${repository_root}/logs"
main_file="${repository_root}/main.py"; [[ -f ${main_file} ]] || die "main.py was not found"
for output_root in "${metrics_root}" "${replays_root}" "${logs_root}"; do
    [[ ! -e ${output_root} || -d ${output_root} ]] || die "output root is not a directory: ${output_root}"
done
assert_new() { [[ ! -e $1 && ! -L $1 ]] || die "output already exists: $1"; }
profile_name() {
    local index=$1 suffix=""
    if (( explicit_cohort_values[index] )); then
        suffix="-cohorts${cohort_values[index]}"
    fi
    printf '%s-active%s-workers%s%s' \
        "${prefix}" "${active_values[index]}" "${worker_values[index]}" "${suffix}"
}
for index in "${!active_values[@]}"; do
    name="$(profile_name "${index}")"
    assert_new "${repository_root}/pente-model-${name}"
    assert_new "${metrics_root}/${name}.jsonl"
    assert_new "${replays_root}/${name}.jsonl"
    assert_new "${logs_root}/${name}.log"
done

if (( ! dry_run )); then
    cuda_status="$("${python_bin}" -c 'import torch; print("available" if torch.cuda.is_available() else "unavailable")')" \
        || die "could not query CUDA availability"
    [[ ${cuda_status} == available ]] || die "CUDA is unavailable; this runner requires CUDA"
    "${python_bin}" -c 'import torch; import kb_pente_native' >/dev/null 2>&1 || die "kb_pente_native could not be imported"
fi

common_args=(
    --gpu --search-backend cpp --ruleset standard --board-size 19
    --model-blocks 6 --model-channels 128 --model-hidden-size 256
    --professional-iterations 0 --self-play-iterations "${final_iteration}" --batch-games "${games}"
    --mcts-sim 64 --temp-threshold 16 --batch-size 512 --learner-steps 1
    --max-training-examples 1000000 --replay-checkpoint-interval 1 --seed 103
    --minimum-batch-occupancy 0.80 --minimum-mean-root-children 4
    --maximum-search-collapse-rate 0.25 --maximum-invalid-policy-fallbacks 0
    --maximum-zero-visit-fallbacks 0 --self-play-health-failure-policy error
)
(( compile_model )) && common_args+=(--compile)
if (( checkpoint_supplied )); then common_args+=(--model "${checkpoint}" --resume-replay "${resume_replay}"); fi
if (( ! dry_run )); then mkdir -p "${metrics_root}" "${replays_root}" "${logs_root}"; fi
for index in "${!active_values[@]}"; do
    active=${active_values[index]}; workers=${worker_values[index]}; cohorts=${cohort_values[index]}
    name="$(profile_name "${index}")"
    model_dir="${repository_root}/pente-model-${name}"
    metric_file="${metrics_root}/${name}.jsonl"
    replay_file="${replays_root}/${name}.jsonl"
    log_file="${logs_root}/${name}.log"
    args=("${common_args[@]}" --active-games "${active}" --native-search-threads "${workers}"
        --native-search-cohorts "${cohorts}"
        --model-dir "${model_dir}" --telemetry-file "${metric_file}" --replay-sample-file "${replay_file}")
    printf '[%d/%d] active-games=%s native-workers=%s native-cohorts=%s batch-games=%s\n' "$((index + 1))" "${#active_values[@]}" "${active}" "${workers}" "${cohorts}" "${games}"
    if (( dry_run )); then printf '  command:'; printf ' %q' "${python_bin}" "${main_file}" "${args[@]}"; printf '\n'; continue; fi
    mkdir "${model_dir}"
    (set -o noclobber; : > "${metric_file}"; : > "${replay_file}"; : > "${log_file}") \
        || die "could not reserve outputs for active=${active}, workers=${workers}, cohorts=${cohorts}"
    printf 'active-games=%s native-workers=%s native-cohorts=%s batch-games=%s\ncommand:' "${active}" "${workers}" "${cohorts}" "${games}" >> "${log_file}"
    printf ' %q' "${python_bin}" "${main_file}" "${args[@]}" >> "${log_file}"; printf '\n' >> "${log_file}"
    if (cd -- "${model_dir}" && "${python_bin}" "${main_file}" "${args[@]}") 2>&1 | tee -a "${log_file}"; then
        printf '  model: %s\n  telemetry: %s\n  replay: %s\n  log: %s\n' "${model_dir}" "${metric_file}" "${replay_file}" "${log_file}"
    else
        status=${PIPESTATUS[0]}; (( status > 0 )) || status=1
        printf 'error: run active=%s workers=%s cohorts=%s batch-games=%s failed (status %s); log: %s\n' "${active}" "${workers}" "${cohorts}" "${games}" "${status}" "${log_file}" >&2
        exit "${status}"
    fi
done
if (( dry_run )); then printf 'Dry run complete: %d profiles validated; no outputs were created.\n' "${#active_values[@]}"; else printf 'Sweep complete: %d profiles.\n' "${#active_values[@]}"; fi
