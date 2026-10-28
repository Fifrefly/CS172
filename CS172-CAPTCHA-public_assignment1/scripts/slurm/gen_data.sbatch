#!/bin/bash -l
# CPU-only job (skipped when both zips exist): regenerate the training and validation sets with the current src/cs172/generate_captcha.py.
#SBATCH -p normal
#SBATCH -A hexm
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
set -eo pipefail
module load anaconda3/python3.8
source /public/software/anaconda3/etc/profile.d/conda.sh
conda activate cs172
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHONPATH=src
mkdir -p data
if [[ -e data/training100k.zip && -e data/validation.zip ]]; then
  echo "data/training100k.zip and data/validation.zip already exist; skipping generation"
  ls -lh data
  exit 0
fi
for f in data/training100k.zip data/training100k.json data/validation.zip data/validation.json; do
  [[ ! -e $f ]] || { echo "error: $f exists but the set is incomplete; move it away first" >&2; exit 1; }
done
python scripts/generate_data.py data/training100k.zip -n 100000 --zip --seed 0
python scripts/generate_data.py data/validation.zip -n 5000 --zip --seed 10000000
ls -lh data
