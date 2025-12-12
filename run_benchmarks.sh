#!/bin/bash

# Initialize variables
NUM_BODIES=2000
ITERATIONS=1000
DT=0.01
BENCHMARK_CSV=benchmark/data.csv
SEED=42

# Remove CSV file if it exists
if [ -f $BENCHMARK_CSV ]; then
    rm $BENCHMARK_CSV
fi

# Source venv
source .venv/bin/activate

# Python simulation
echo -e "\033[33m[PYTHON] Start simulation\033[0m"
python python/nbody.py \
    --bodies $NUM_BODIES \
    --iterations $ITERATIONS \
    --dt $DT\
    --seed $SEED\
    --benchmark_csv $BENCHMARK_CSV
echo -e "\033[33m[PYTHON] Done!\033[0m"

# Mojo simulation
echo -e "\033[33m[MOJO] Start simulation\033[0m"
mojo mojo/nbody/nbody.mojo $NUM_BODIES $ITERATIONS $DT $BENCHMARK_CSV
echo -e "\033[33m[MOJO] Done!\033[0m"

# Julia simulation
echo -e "\033[33m[Julia] Start simulation\033[0m"
julia --project=julia julia/nbody.jl $NUM_BODIES $ITERATIONS $DT $BENCHMARK_CSV
echo -e "\033[33m[Julia] Done!\033[0m"

# Go simulation
echo -e "\033[33m[Go] Start simulation\033[0m"
go run go/nbody.go \
    --bodies $NUM_BODIES\
    --iterations $ITERATIONS\
    --dt $DT\
    --benchmark_csv $BENCHMARK_CSV
echo -e "\033[33m[Go] Done!\033[0m"
