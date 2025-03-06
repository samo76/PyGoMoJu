#!/usr/bin/env python3
"""
Benchmark script for N-body simulation implementations.
Compares execution times between Mojo/Max, Python, and Go.
"""

import os
import sys
import subprocess
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

# Add the Python implementation directory to the path
sys.path.append(str(Path(__file__).parent.parent / "python"))
from nbody import NBodySimulation as PythonNBodySimulation

# Define paths
REPO_ROOT = Path(__file__).parent.parent
PYTHON_DIR = REPO_ROOT / "python"
GO_DIR = REPO_ROOT / "go"
MOJO_DIR = REPO_ROOT / "mojo" / "nbody"  # Updated path to Mojo directory
BENCHMARK_DIR = REPO_ROOT / "benchmark"

# Define benchmark parameters
NUM_BODIES_OPTIONS = [100, 500, 1000, 2000]  # Different body counts to test
NUM_ITERATIONS = 100  # Fixed iteration count for consistency
DT = 0.01  # Time step
SEED = 42  # Random seed
NUM_RUNS = 3  # Number of runs for averaging


def benchmark_python(num_bodies):
    """Benchmark the Python implementation."""
    print(f"Running Python benchmark with {num_bodies} bodies...")

    total_time = 0
    for run in range(NUM_RUNS):
        sim = PythonNBodySimulation(
            num_bodies=num_bodies, num_iterations=NUM_ITERATIONS, dt=DT, seed=SEED
        )
        execution_time = sim.run_simulation()
        total_time += execution_time
        print(f"  Run {run+1}/{NUM_RUNS}: {execution_time:.4f} seconds")

    avg_time = total_time / NUM_RUNS
    print(f"Python average execution time: {avg_time:.4f} seconds")
    return avg_time


def benchmark_go(num_bodies):
    """Benchmark the Go implementation."""
    print(f"Running Go benchmark with {num_bodies} bodies...")

    # Make sure the Go binary is built
    go_build_cmd = [
        "go",
        "build",
        "-o",
        str(GO_DIR / "nbody_bin"),
        str(GO_DIR / "nbody.go"),
    ]
    subprocess.run(go_build_cmd, check=True)

    # Run the benchmark
    total_time = 0
    for run in range(NUM_RUNS):
        env = os.environ.copy()
        cmd = [
            str(GO_DIR / "nbody_bin"),
            "-bodies",
            str(num_bodies),
            "-iterations",
            str(NUM_ITERATIONS),
            "-dt",
            str(DT),
            "-seed",
            str(SEED),
        ]

        # Time the execution
        start_time = time.time()
        process = subprocess.run(cmd, env=env, capture_output=True, text=True)
        execution_time = time.time() - start_time

        # Extract the time from the output if possible, otherwise use measured time
        output = process.stdout
        if "Execution time:" in output:
            try:
                execution_time = float(
                    output.split("Execution time:")[1].split("seconds")[0].strip()
                )
            except (ValueError, IndexError):
                pass  # Use the measured time if extraction fails

        total_time += execution_time
        print(f"  Run {run+1}/{NUM_RUNS}: {execution_time:.4f} seconds")

    avg_time = total_time / NUM_RUNS
    print(f"Go average execution time: {avg_time:.4f} seconds")
    return avg_time


def benchmark_mojo(num_bodies):
    """Benchmark the Mojo implementation."""
    print(f"Running Mojo benchmark with {num_bodies} bodies...")

    # Check if Magic is available
    try:
        subprocess.run(["which", "magic"], check=True, capture_output=True)
    except (subprocess.SubprocessError, FileNotFoundError):
        print("Magic command not found. Skipping Mojo benchmark.")
        return None

    # Run the benchmark using Magic
    total_time = 0
    for run in range(NUM_RUNS):
        # Construct the command to run the Mojo benchmark
        # We need to use 'magic run' to execute the Mojo file
        cmd = [
            "magic",
            "run",
            str(MOJO_DIR / "nbody.mojo"),
            "--bodies",
            str(num_bodies),
            "--iterations",
            str(NUM_ITERATIONS),
            "--dt",
            str(DT),
            "--seed",
            str(SEED),
        ]

        # Time the execution
        start_time = time.time()
        process = subprocess.run(cmd, capture_output=True, text=True)
        execution_time = time.time() - start_time

        # Extract the time from the output if possible, otherwise use measured time
        output = process.stdout
        if "Execution time:" in output:
            try:
                execution_time = float(
                    output.split("Execution time:")[1].split("seconds")[0].strip()
                )
            except (ValueError, IndexError):
                pass  # Use the measured time if extraction fails

        total_time += execution_time
        print(f"  Run {run+1}/{NUM_RUNS}: {execution_time:.4f} seconds")

    avg_time = total_time / NUM_RUNS
    print(f"Mojo average execution time: {avg_time:.4f} seconds")
    return avg_time


def visualize_results(results):
    """Generate comparative visualization of benchmark results."""
    df = pd.DataFrame(results)

    # Bar chart
    plt.figure(figsize=(12, 8))

    # Plot grouped bars
    bar_width = 0.25
    indices = np.arange(len(NUM_BODIES_OPTIONS))

    # Plot bars for each language (if data exists)
    if "Python" in df.columns:
        plt.bar(
            indices - bar_width, df["Python"], bar_width, label="Python", color="blue"
        )
    if "Go" in df.columns:
        plt.bar(indices, df["Go"], bar_width, label="Go", color="green")
    if "Mojo" in df.columns:
        plt.bar(indices + bar_width, df["Mojo"], bar_width, label="Mojo", color="red")

    plt.xlabel("Number of Bodies")
    plt.ylabel("Execution Time (seconds)")
    plt.title("N-Body Simulation Performance Comparison")
    plt.xticks(indices, [str(n) for n in NUM_BODIES_OPTIONS])
    plt.legend()
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)

    # Add value labels on top of each bar
    for i, col in enumerate(["Python", "Go", "Mojo"]):
        if col in df.columns:
            for j, value in enumerate(df[col]):
                offset = (i - 1) * bar_width
                plt.text(
                    j + offset,
                    value + 0.1,
                    f"{value:.2f}s",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    # Calculate speedup and add a text box with summary
    if "Python" in df.columns and "Mojo" in df.columns:
        max_python = df["Python"].max()
        max_mojo = df["Mojo"].max()
        speedup = max_python / max_mojo if max_mojo > 0 else 0

        textbox = (
            f"Maximum Problem Size: {NUM_BODIES_OPTIONS[-1]} bodies\n"
            f"Python Max Time: {max_python:.2f}s\n"
            f"Mojo Max Time: {max_mojo:.2f}s\n"
            f"Speedup (Mojo vs Python): {speedup:.2f}x"
        )

        plt.figtext(
            0.15,
            0.82,
            textbox,
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.5"),
        )

    # Save the visualization
    plt.tight_layout()
    plt.savefig(BENCHMARK_DIR / "benchmark_results.png", dpi=300)
    plt.close()

    # Also save as CSV for further analysis
    df.to_csv(BENCHMARK_DIR / "benchmark_results.csv", index=False)

    print(
        f"Results saved to {BENCHMARK_DIR / 'benchmark_results.png'} and {BENCHMARK_DIR / 'benchmark_results.csv'}"
    )


def main():
    """Run benchmarks on all implementations with various body counts."""
    results = {"Bodies": NUM_BODIES_OPTIONS}

    # Python benchmarks
    python_times = []
    for num_bodies in NUM_BODIES_OPTIONS:
        python_time = benchmark_python(num_bodies)
        python_times.append(python_time)
    results["Python"] = python_times

    # Go benchmarks
    go_times = []
    for num_bodies in NUM_BODIES_OPTIONS:
        go_time = benchmark_go(num_bodies)
        go_times.append(go_time)
    results["Go"] = go_times

    # Mojo benchmarks (if available)
    mojo_times = []
    for num_bodies in NUM_BODIES_OPTIONS:
        mojo_time = benchmark_mojo(num_bodies)
        mojo_times.append(mojo_time if mojo_time is not None else float("nan"))

    if not all(np.isnan(t) for t in mojo_times):
        results["Mojo"] = mojo_times

    # Visualize and save results
    visualize_results(results)

    # Print summary
    print("\nBenchmark Summary:")
    print(
        f"{'Bodies':<10} {'Python (s)':<15} {'Go (s)':<15} {'Mojo (s)':<15} {'Python/Go':<15} {'Python/Mojo':<15}"
    )
    print("-" * 80)

    for i, num_bodies in enumerate(NUM_BODIES_OPTIONS):
        python_time = python_times[i]
        go_time = go_times[i]
        mojo_time = (
            mojo_times[i]
            if i < len(mojo_times) and not np.isnan(mojo_times[i])
            else float("nan")
        )

        python_go_ratio = python_time / go_time if go_time > 0 else float("nan")
        python_mojo_ratio = (
            python_time / mojo_time
            if mojo_time > 0 and not np.isnan(mojo_time)
            else float("nan")
        )

        print(
            f"{num_bodies:<10} {python_time:<15.4f} {go_time:<15.4f} "
            f"{mojo_time:<15.4f} {python_go_ratio:<15.2f} {python_mojo_ratio:<15.2f}"
        )


if __name__ == "__main__":
    main()
