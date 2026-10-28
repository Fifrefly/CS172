#!/usr/bin/env bash
# Run the experiments reported in main.ipynb, one after another on the current GPU.
#
# Usage:
#   scripts/train.sh list                      # print all experiment names and groups
#   scripts/train.sh NAME_OR_GROUP [SEED ...]  # train, then evaluate on data/test/test0-4
#
# Examples:
#   scripts/train.sh final 42                  # final model (100k images, ResNet-18)
#   scripts/train.sh size 42 43 44             # training-set-size sweep, three seeds
#   scripts/train.sh bsx 42                    # batch-size sweep (runs back to back, same GPU)
#
# Every run is written to outputs/NAME/<timestamp>-<id>/ (config.json, metrics.json,
# checkpoints/best.pt, evaluation.json), so several seeds of one experiment share outputs/NAME/.
# Environment variables: OUTPUT_ROOT (default outputs), PYTHON (default python),
# TRAIN_DATA, VAL_DATA, TEST_ROOT, EPOCHS, WORKERS.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

OUTPUT_ROOT=${OUTPUT_ROOT:-outputs}
PYTHON=${PYTHON:-python}
TRAIN_DATA=${TRAIN_DATA:-data/training100k.zip}
VAL_DATA=${VAL_DATA:-data/validation.zip}
TEST_ROOT=${TEST_ROOT:-data/test}
EPOCHS=${EPOCHS:-20}
WORKERS=${WORKERS:-4}

# Baseline hyper-parameters; each experiment overrides exactly one of them.
BASE=(--model resnet18 --batch-size 64 --learning-rate 0.001 --weight-decay 0.0001 --augment)

# NAME -> extra train.py arguments (later arguments override BASE).
declare -A ARGS=(
  # final model and training-set size (6.1)
  [captcha100k-resnet18-aug]="--num-images 100000"
  [size-50000]="--num-images 50000"
  [size-20000]="--num-images 20000"
  [baseline-10k]="--num-images 10000"
  [size-5000]="--num-images 5000"
  [size-2000]="--num-images 2000"
  [size-1000]="--num-images 1000"
  # augmentation (6.2), weight decay (6.3)
  [noaug-10k]="--num-images 10000"  # also drops --augment, see run_one
  [wd-0]="--num-images 10000 --weight-decay 0"
  [wd-0.01]="--num-images 10000 --weight-decay 0.01"
  [wd-0.1]="--num-images 10000 --weight-decay 0.1"
  # batch size (6.4); the five runs of one seed must share a GPU, so they form one group
  [bsx16-10k]="--num-images 10000 --batch-size 16"
  [bsx32-10k]="--num-images 10000 --batch-size 32"
  [bsx64-10k]="--num-images 10000 --batch-size 64"
  [bsx128-10k]="--num-images 10000 --batch-size 128"
  [bsx256-10k]="--num-images 10000 --batch-size 256"
  # learning rate (6.5)
  [lr-0.0001]="--num-images 10000 --learning-rate 0.0001"
  [lr-0.0003]="--num-images 10000 --learning-rate 0.0003"
  [lr-0.003]="--num-images 10000 --learning-rate 0.003"
  [lr-0.01]="--num-images 10000 --learning-rate 0.01"
  # backbone (6.6) and global average pooling (6.7)
  [resnet34-10k]="--num-images 10000 --model resnet34"
  [captcha18-10k]="--num-images 10000 --model captcha_resnet18"
  [captcha34-10k]="--num-images 10000 --model captcha_resnet34"
)

declare -A GROUPS_=(
  [final]="captcha100k-resnet18-aug"
  [size]="size-50000 size-20000 baseline-10k size-5000 size-2000 size-1000"
  [ablate]="noaug-10k wd-0 wd-0.01 wd-0.1 lr-0.0001 lr-0.0003 lr-0.003 lr-0.01 resnet34-10k captcha18-10k captcha34-10k"
  [bsx]="bsx16-10k bsx32-10k bsx64-10k bsx128-10k bsx256-10k"
)

list() {
  echo "groups:"
  for group in final size ablate bsx; do echo "  $group: ${GROUPS_[$group]}"; done
}

run_one() {
  local name=$1 seed=$2
  local -a extra
  read -r -a extra <<<"${ARGS[$name]}"

  # --augment is a flag, so "no augmentation" removes it from the baseline instead of overriding it.
  local -a base=()
  local arg
  for arg in "${BASE[@]}"; do
    [[ $name == noaug-10k && $arg == --augment ]] || base+=("$arg")
  done

  echo "== $name (seed $seed) on $(hostname)"
  "$PYTHON" scripts/train.py \
    --output-dir "$OUTPUT_ROOT/$name" \
    --data-dir "$TRAIN_DATA" --val-data-dir "$VAL_DATA" \
    --epochs "$EPOCHS" --workers "$WORKERS" --seed "$seed" --tensorboard \
    "${base[@]}" "${extra[@]}"

  # Evaluate the run that was just created (the newest directory) on the five test sets.
  local run_dir
  run_dir=$(ls -dt "$OUTPUT_ROOT/$name"/*/ | head -n 1)
  "$PYTHON" scripts/evaluate.py --checkpoint "${run_dir}checkpoints/best.pt" \
    --data "$TEST_ROOT"/test0 "$TEST_ROOT"/test1 "$TEST_ROOT"/test2 "$TEST_ROOT"/test3 "$TEST_ROOT"/test4
}

if (($# < 1)) || [[ $1 == -h || $1 == --help || $1 == list ]]; then
  sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'
  list
  exit 0
fi

target=$1
shift
seeds=("$@")
((${#seeds[@]})) || seeds=(42)

if [[ -n ${GROUPS_[$target]:-} ]]; then
  names=(${GROUPS_[$target]})
elif [[ -n ${ARGS[$target]:-} ]]; then
  names=("$target")
else
  echo "error: unknown experiment or group '$target'" >&2
  list >&2
  exit 2
fi

# Seeds are the outer loop so that a group's runs for one seed stay on one GPU.
for seed in "${seeds[@]}"; do
  for name in "${names[@]}"; do
    run_one "$name" "$seed"
  done
done
