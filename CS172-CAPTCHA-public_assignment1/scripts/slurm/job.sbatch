#!/bin/bash -l
# Generic Slurm job: sbatch [slurm options] scripts/slurm/job.sbatch <scripts/train.sh arguments>
# e.g. scripts/slurm/job.sbatch final 42      (use scripts/slurm/submit.sh to pick a GPU automatically)
set -eo pipefail
module load anaconda3/python3.8
source /public/software/anaconda3/etc/profile.d/conda.sh
conda activate cs172
cd "${SLURM_SUBMIT_DIR:?}"
hostname
nvidia-smi -L
scripts/train.sh "$@"
