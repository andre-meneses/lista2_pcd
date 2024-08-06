#!/bin/bash

processors=(1 2 4 8)
sizes=(2000000 4000000 8000000 16000000)
repeats=10

# Output CSV file
output_file="results.csv"
echo "Processors,Size,Run,Elapsed Time" > "$output_file"

echo "Starting MPI program..."

# Loop through processors and sizes
for p in "${processors[@]}"; do
  for size in "${sizes[@]}"; do
    echo "Running with $p processors for size $size, $repeats times..."

    # Run the configuration multiple times
    for (( i=1; i<=repeats; i++ )); do
      # Run the MPI program
      result=$(mpiexec -n "$p" ./main "$size")

      # Extract the elapsed time
      elapsed_time=$(echo "$result" | grep -oP '=\s*\K[\d.]+')

      # Print each run's result for sanity check
      echo "Run $i: Elapsed Time: $elapsed_time"

      # Save each run's result to the CSV file
      echo "$p,$size,Run $i,$elapsed_time" >> "$output_file"
    done

    echo "Finished runs with $p processors for size $size."
  done
done

echo "Execution completed. Results saved to $output_file."

