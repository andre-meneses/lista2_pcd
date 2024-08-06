#!/bin/bash

# Function to generate a matrix as natural number counts
generate_matrix() {
  local rows=$1
  local cols=$2
  local matrix=""
  local count=1

  for ((i = 0; i < rows; i++)); do
    for ((j = 0; j < cols; j++)); do
      matrix+="$count "
      count=$((count + 1))
    done
    matrix+="\n"
  done
  echo -e "$matrix"
}

# Function to generate a vector as natural number counts
generate_vector() {
  local size=$1
  local vector=""
  for ((i = 1; i <= size; i++)); do
    vector+="$i\n"
  done
  echo -e "$vector"
}

# Arrays for number of processors and matrix sizes
processors=(1 2 4)
sizes=(256 512 1024)

# Output CSV file
output_file="results_original.csv"
echo "Processors,Rows,Columns,Elapsed Time" > "$output_file"

echo "Starting MPI matrix-vector multiplication script..."

# Loop through processors and sizes
for p in "${processors[@]}"; do
  for size in "${sizes[@]}"; do
    rows=$size
    cols=$size

    echo "Running with $p processors for matrix size ${rows}x${cols}..."

    # Generate matrix and vector
    matrix=$(generate_matrix $rows $cols)
    vector=$(generate_vector $cols)

    # Print matrix and vector for sanity check
    #echo "Generated matrix:"
    #echo "$matrix"
    #echo "Generated vector:"
    #echo "$vector"

    # Run the MPI program
    result=$(mpiexec -n $p ./main_original <<EOF
$rows
$cols
$matrix
$vector
EOF
)

    # Extract the elapsed time
    elapsed_time=$(echo "$result" | grep "Elapsed time" | awk '{print $4}')

    # Print result for sanity check
    echo "Elapsed Time: $elapsed_time"

    # Save the result to the CSV file
    echo "$p,$rows,$cols,$elapsed_time" >> "$output_file"

    echo "Finished run with $p processors for matrix size ${rows}x${cols}."
  done
done

echo "Execution completed. Results saved to $output_file."

