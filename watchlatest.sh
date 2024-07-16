# There will be a bunch of files like:
# X24515-*.out
# X24516-*.out
# X24517-*.out

# I want to watch the latest file (highest number) for changes
file=$(ls -1 X*.out | sort -n | tail -n 1)

# If there is a parameter, grep for it in the filenames first
if [ -n "$1" ]; then
  file=$(ls -1 X*.out | grep $1 | sort -n | tail -n 1)
fi

# If there is no file, exit
if [ -z "$file" ]; then
  echo "No files found"
  exit 1
fi

echo "Watching $file"
tail -f $file