#!/usr/bin/env python3
"""
Visualization script for N-body simulation benchmark results.
Reads execution times from benchmark_results.csv and generates visualizations.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# Define paths
BENCHMARK_DIR = Path(__file__).parent
CSV_FILE = BENCHMARK_DIR / "benchmark_results.csv"

# Load benchmark results from CSV
df = pd.read_csv(CSV_FILE)

# Convert columns to numeric types explicitly
df["Python"] = pd.to_numeric(df["Python"], errors="coerce")
df["Go"] = pd.to_numeric(df["Go"], errors="coerce")
df["Mojo"] = pd.to_numeric(df["Mojo"], errors="coerce")

# Bar chart
plt.figure(figsize=(12, 8))

bar_width = 0.25
indices = np.arange(len(df["Bodies"]))

# Plot bars for each language
plt.bar(indices - bar_width, df["Python"], bar_width, label="Python", color="blue")
plt.bar(indices, df["Go"], bar_width, label="Go", color="green")
plt.bar(indices + bar_width, df["Mojo"], bar_width, label="Mojo", color="red")

plt.xlabel("Number of Bodies")
plt.ylabel("Execution Time (seconds)")
plt.title("N-Body Simulation Performance Comparison")
plt.xticks(indices, df["Bodies"].astype(str))
plt.legend()
plt.grid(True, axis="y", linestyle="--", alpha=0.7)

# Add value labels on top of each bar
for i, col in enumerate(["Python", "Go", "Mojo"]):
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

# Save the visualization
plt.tight_layout()
plt.savefig("benchmark_results.png", dpi=300)
plt.close()

print("Visualization saved to benchmark_results.png")
