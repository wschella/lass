#!/bin/bash
#SBATCH --job-name=xtra_architecture_selection
#SBATCH --exclusive
#SBATCH --ntasks=5
#SBATCH --nodes=1
#SBATCH --mem-per-gpu=40G
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-gpu=2
#SBATCH --output=X%j-%x.out

set -euo pipefail

models=("albert/albert-base-v2" "google-bert/bert-base-cased" "FacebookAI/roberta-base" "microsoft/deberta-v3-base" "openai-community/gpt2")
# models=("albert/albert-base-v2")

# We need to provide the time ourselves, since we do multiple runs
# in python we have this formaT: %m%d%H%M
time=$(date +'%m%d%H%M')

BASE="."
source ~/miniconda3/bin/activate "$BASE/.venv"

for model in "${models[@]}"; do
  malias=$(echo $model | tr "/" "_")
  echo "Running xtra_architecture_selection/$model"
  srun --output T${time}xtra_archsel_${malias}.out -c 2 --mem 40G --exclusive --nodes 1 --ntasks 1 python src/lass/experiments/xtra_architecture_selection.py --model $model --time $time "$@" &
done
wait


# This is essntial to wait for all the jobs to finish
# https://hpc.nmsu.edu/discovery/slurm/tasks/parallel-execution/
wait

echo "Done"