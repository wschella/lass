#!/bin/bash
#SBATCH --job-name=q3correlation
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=4
#SBATCH --output=X%j-%x.out


BASE="."
source ~/miniconda3/bin/activate "$BASE/.venv"

python src/lass/experiments/q3correlation.py "$@"