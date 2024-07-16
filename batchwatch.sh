# To be used with sbatch like this:
# sbatch whatever.sh | batchwatch.sh

# Sbatch output looks like this:
# "Submitted batch job 24506"
ID=$(awk '{print $4}')

# We want to watch the output file, which is named after the job number,
# in the format "X24506-{something}.out"

# Need to wait for it tho
files=$(ls -1 X$ID-*.out 2>/dev/null | wc -l)
while [ $files -eq 0 ]; do
    sleep 1
    files=$(ls -1 X$ID-*.out 2>/dev/null | wc -l)
done

echo "Watching X$ID-*.out"
tail -f X$ID-*.out