#!/bin/bash
# Helper script to run the N-body simulation benchmarks

# Make sure script can execute even if run from another directory
cd "$(dirname "$0")"

# Check Python dependencies
echo "Checking Python dependencies..."
if ! uv pip install -q -r python/requirements.txt; then
    echo "Failed to install Python dependencies. Please check your Python environment."
    exit 1
fi

# Build Go binary
echo "Building Go implementation..."
cd go
go build -o nbody_bin nbody.go
if [ $? -ne 0 ]; then
    echo "Failed to build Go binary. Make sure Go is installed."
    exit 1
fi
cd ..

# Check if Magic is installed for Mojo
echo "Checking for Magic command for Mojo..."
if command -v magic &> /dev/null; then
    echo "Magic command found, Mojo benchmarks will be included."
    
    # Check if the Mojo file exists
    if [ -f "mojo/nbody/nbody.mojo" ]; then
        echo "Mojo implementation found at mojo/nbody/nbody.mojo"
    else
        echo "Warning: Mojo implementation not found at expected location (mojo/nbody/nbody.mojo)."
        echo "Mojo benchmarks may fail."
    fi
else
    echo "Magic command not found. Mojo benchmarks will be skipped."
fi

# Run benchmarks
echo -e "\nRunning benchmarks..."
python benchmark/benchmark.py

echo -e "\nBenchmark completed! Results are saved in the benchmark directory."
echo "You can find visual comparison in benchmark/benchmark_results.png"
echo "Numerical results are also available in benchmark/benchmark_results.csv"

# Run individual implementations for demonstration
echo -e "\nRunning individual implementations with 100 bodies and 10 iterations for demonstration:"

echo -e "\nRunning Python implementation..."
python python/nbody.py --bodies 100 --iterations 10

echo -e "\nRunning Go implementation..."
./go/nbody_bin -bodies 100 -iterations 10

if command -v magic &> /dev/null; then
    echo -e "\nRunning Mojo implementation..."
    magic run mojo/nbody/nbody.mojo --bodies 100 --iterations 10
fi 