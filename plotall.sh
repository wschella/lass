echo "Plotting q1indistribution"
python src/lass/plotting/q1indistribution.py --shots 0
python src/lass/plotting/q1indistribution.py --shots 3

echo "Plotting q2outofdistribution"
python src/lass/plotting/q2outofdistribution.py --shots 0
python src/lass/plotting/q2outofdistribution.py --shots 3

echo "Plotting q3correlation"
python src/lass/plotting/q3correlation.py --shots 0
python src/lass/plotting/q3correlation.py --shots 3

echo "Plotting q4multitask"
python src/lass/plotting/q4multitask.py --shots 0
python src/lass/plotting/q4multitask.py --shots 3

echo "Plotting q5population"
python src/lass/plotting/q5population.py --shots 0
python src/lass/plotting/q5population.py --shots 3

echo "Plotting q6scaling"
python src/lass/plotting/q6scaling.py --shots 0
python src/lass/plotting/q6scaling.py --shots 3

echo "Plotting q7shots"
python src/lass/plotting/q7shots.py