#!/bin/bash
#SBATCH --job-name=q6scaling
#SBATCH --exclusive
#SBATCH --ntasks=12
#SBATCH --nodes=2
#SBATCH --mem-per-gpu=40G
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-gpu=2
#SBATCH --output=X%j-%x.out

set -euo pipefail

# n-tasks = 3 assessors * 12 sizes = 36
sizes=("2m" "16m" "53m" "125m" "244m" "422m" "1b" "2b" "4b" "8b" "27b" "128b")
# sizes=("27b" "128b")

# We need to provide the time ourselves, since we do multiple runs
# in python we have this formaT: %m%d%H%M
time=$(date +'%m%d%H%M')

BASE="."
source ~/miniconda3/bin/activate "$BASE/.venv"

for size in "${sizes[@]}"; do
  echo "Running q6small$size"
  srun --output T${time}q6small${size}.out -c 2 --mem 40G --exclusive --nodes 1 --ntasks 1 python src/lass/experiments/q6scaling.py --assessor small --subject $size --time $time "$@" &
done
wait

for size in "${sizes[@]}"; do
  echo "Running q6base$size"
  srun --output T${time}q6base${size}.out -c 2 --mem 40G --exclusive --nodes 1 --ntasks 1 python src/lass/experiments/q6scaling.py --assessor base --subject $size --time $time "$@" &
done
wait

for size in "${sizes[@]}"; do
  echo "Running q6large$size"
  srun --output T${time}q6large${size}.out -c 2 --mem 40G --exclusive --nodes 1 --ntasks 1 python src/lass/experiments/q6scaling.py --assessor large --subject $size --time $time "$@" &
done
wait

# This is essntial to wait for all the jobs to finish
# https://hpc.nmsu.edu/discovery/slurm/tasks/parallel-execution/
wait

echo "Done"