#!/usr/bin/env bash
# Submit train.sh jobs to Slurm, picking the GPU type by the cluster's rules:
#   1. A40 (partition hexm), 2. L40 (partition hexm_l40), 3. RTX 2080 Ti (partition normal).
# A type is used when it can start soon (see pick_gpu). Waits while the user already has MAX_JOBS jobs queued/running.
#
# Usage: scripts/slurm/submit.sh "<train.sh args>" ["<train.sh args>" ...]
#   scripts/slurm/submit.sh "final 42"
#   scripts/slurm/submit.sh "size 42" "ablate 42" "bsx 42" "size 43" "ablate 43" "bsx 43"
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
mkdir -p slurm_logs

MAX_JOBS=${MAX_JOBS:-6}
USER_NAME=${USER:-$(id -un)}

# start_epoch SBATCH_ARGS...  -> estimated start time (epoch seconds) from `sbatch --test-only`.
# A GPU type can show idle nodes in sinfo yet be unschedulable for this account (the estimate then lies
# decades away), so the estimate is a better test of "available" than the idle-GPU count.
start_epoch() {
  local out ts
  out=$(sbatch --test-only -N 1 -n 1 -c 8 --mem=32G "$@" --wrap=hostname 2>&1)
  ts=$(sed -n 's/.*to start at \([0-9T:-]*\) .*/\1/p' <<<"$out")
  if [[ -n $ts ]]; then date -d "$ts" +%s; else echo 9999999999; fi
}

# pick_gpu -> sets SBATCH_GPU_ARGS and GPU_NAME.
# A40, then L40, are used when they can start within WAIT_MINUTES (default 30); otherwise the type with
# the earliest estimated start is used, which is normally the RTX 2080 Ti queue.
pick_gpu() {
  local wait=$(( ${WAIT_MINUTES:-30} * 60 )) now t_a40 t_l40 t_2080
  now=$(date +%s)
  A40_ARGS="-p hexm -A hexm --gres=gpu:NVIDIAA40:1 --time=0"
  L40_ARGS="-p hexm_l40 -A hexm --gres=gpu:NVIDIAL40:1 --time=0"
  TI_ARGS="-p normal -A hexm --gres=gpu:NVIDIAGeForceRTX2080Ti:1 --time=12:00:00"
  t_a40=$(start_epoch $A40_ARGS); t_l40=$(start_epoch $L40_ARGS)
  if ((t_a40 - now <= wait)); then
    GPU_NAME=A40; SBATCH_GPU_ARGS=$A40_ARGS
  elif ((t_l40 - now <= wait)); then
    GPU_NAME=L40; SBATCH_GPU_ARGS=$L40_ARGS
  else
    t_2080=$(start_epoch $TI_ARGS)
    if ((t_2080 < t_l40 && t_2080 < t_a40)); then GPU_NAME=2080Ti; SBATCH_GPU_ARGS=$TI_ARGS
    elif ((t_l40 <= t_a40)); then GPU_NAME=L40; SBATCH_GPU_ARGS=$L40_ARGS
    else GPU_NAME=A40; SBATCH_GPU_ARGS=$A40_ARGS; fi
  fi
}

for job_args in "$@"; do
  while (($(squeue -h -u "$USER_NAME" | wc -l) >= MAX_JOBS)); do sleep 30; done
  pick_gpu
  name=$(echo "$job_args" | tr ' ' '-')
  echo "[$GPU_NAME] train.sh $job_args"
  # shellcheck disable=SC2086
  sbatch -N 1 -n 1 -c 8 --mem=32G $SBATCH_GPU_ARGS -J "cs172-$name" \
    --output="slurm_logs/${name}_%j.out" --error="slurm_logs/${name}_%j.err" \
    scripts/slurm/job.sbatch $job_args
  sleep 5  # let Slurm account for the new allocation before the next idle-GPU check
done
