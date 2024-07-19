#!/bin/bash
#SBATCH --job-name=q7shots
#SBATCH --exclusive
#SBATCH --ntasks=4
#SBATCH --nodes=1
#SBATCH --mem-per-gpu=40G
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-gpu=2
#SBATCH --output=X%j-%x.out

set -euo pipefail

shots=("0" "1" "2" "3")
# shots=("0")

# We need to provide the time ourselves, since we do multiple runs
# in python we have this formaT: %m%d%H%M
time=$(date +'%m%d%H%M')

BASE="."
source ~/miniconda3/bin/activate "$BASE/.venv"

for shot in "${shots[@]}"; do
  echo "Running q7left$shot"
  srun --output T${time}q7left${shot}.out -c 2 --mem 40G --exclusive --nodes 1 --ntasks 1 python src/lass/experiments/q7shots.py --shots $shot --truncation-side left --time $time "$@" &
done
wait

for shot in "${shots[@]}"; do
  echo "Running q7right$shot"
  srun --output T${time}q7right${shot}.out -c 2 --mem 40G --exclusive --nodes 1 --ntasks 1 python src/lass/experiments/q7shots.py --shots $shot --truncation-side right --time $time "$@" &
done
wait

# This is essntial to wait for all the jobs to finish
# https://hpc.nmsu.edu/discovery/slurm/tasks/parallel-execution/
wait

echo "Done"