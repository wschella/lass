#! /bin/bash

# lrzuntar -p 1 -d BIG-bench/bigbench/benchmark_tasks/abstract_narrative_understanding/results/full-logs_BIG-G_BIG-G-sparse.tar.lrz -f -O tmp/
decompress() {
  path=$1
  task=$(echo $path | cut -d'/' -f4)
  echo "Decompressing $task"
  mkdir -p artifacts/logs/$task
  lrzuntar -q -p 1 -O artifacts/logs/$task/ -d $path
}
export -f decompress

parallel \
  -v          `# Print the job to be run on stdout` \
  --lb        `# Output on line basis` \
  --progress  `# Display progress ` \
  decompress \
  ::: $(ls bigbench/bigbench/benchmark_tasks/**/results/full-logs**)