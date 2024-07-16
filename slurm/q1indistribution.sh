#!/bin/bash
#SBATCH --job-name=q1indistribution
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=4
#SBATCH --output=X%j-%x.out


BASE="."
source ~/miniconda3/bin/activate "$BASE/.venv"

python src/lass/experiments/q1indistribution.py "$@"