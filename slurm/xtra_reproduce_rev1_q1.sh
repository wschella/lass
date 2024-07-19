#!/bin/bash
#SBATCH --job-name=reproduce_rev1_q1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=4
#SBATCH --output=X%j-%x.out


BASE="."
source ~/miniconda3/bin/activate "$BASE/.venv"

python src/lass/experiments/xtra_reproduce_rev1_q1.py "$@"