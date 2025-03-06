#!/usr/bin/env python3
"""
Visualization script for N-body simulation benchmark results.
Creates a visually stunning comparison of Python, Go, and Mojo performance.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter
import pandas as pd

# Set the style
plt.style.use("dark_background")
# Try to use Avenir font if available, otherwise fall back to a default sans-serif font
try:
    mpl.rcParams["font.family"] = "Avenir"
except:
    mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.size"] = 12
mpl.rcParams["axes.linewidth"] = 2
mpl.rcParams["axes.edgecolor"] = "#DDDDDD"
mpl.rcParams["xtick.major.size"] = 0
mpl.rcParams["ytick.major.size"] = 0

# Benchmark results
languages = ["Python", "Go", "Mojo"]
execution_times = [178.86, 7.80, 7.21]
colors = ["#4285F4", "#34A853", "#EA4335"]  # Google-inspired colors

# Calculate speedup compared to Python
speedups = [execution_times[0] / t for t in execution_times]

# Create a DataFrame for easy manipulation
df = pd.DataFrame(
    {
        "Language": languages,
        "Execution Time (s)": execution_times,
        "Speedup": speedups,
        "Color": colors,
    }
)

# Sort by execution time (ascending)
df = df.sort_values("Execution Time (s)", ascending=False)

# Create the figure and axes with more space
fig = plt.figure(figsize=(14, 8))
fig.patch.set_facecolor("#121212")

# Create a 1x2 grid with specific width ratios
gs = fig.add_gridspec(1, 2, width_ratios=[2, 1], wspace=0.3)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])


# Function to add value labels
def add_labels(ax, bars, values, format_str="{:.2f}"):
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + (max(values) * 0.03),  # Position labels relative to max value
            format_str.format(value),
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
            color="white",
        )


# Plot 1: Execution Time (Bar Chart)
bars1 = ax1.bar(
    df["Language"],
    df["Execution Time (s)"],
    color=df["Color"],
    width=0.6,
    edgecolor="white",
    linewidth=1.5,
    alpha=0.8,
)

# Add gradient effect to bars (simplified to avoid potential issues)
for i, (bar, color) in enumerate(zip(bars1, df["Color"])):
    bar.set_alpha(0.8)
    bar.set_edgecolor("white")
    bar.set_linewidth(1.5)

# Add value labels
add_labels(ax1, bars1, df["Execution Time (s)"])

# Customize the first plot
ax1.set_title("Execution Time (Lower is Better)", fontsize=18, pad=20, color="white")
ax1.set_ylabel("Seconds", fontsize=14, color="white")
ax1.set_ylim(0, max(execution_times) * 1.15)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.grid(axis="y", linestyle="--", alpha=0.3)

# Plot 2: Speedup (Bar Chart)
bars2 = ax2.bar(
    df["Language"],
    df["Speedup"],
    color=df["Color"],
    width=0.6,
    edgecolor="white",
    linewidth=1.5,
    alpha=0.8,
)

# Add value labels
add_labels(ax2, bars2, df["Speedup"], "{:.1f}x")

# Customize the second plot
ax2.set_title(
    "Speedup vs. Python (Higher is Better)", fontsize=18, pad=20, color="white"
)
ax2.set_ylabel("Times Faster", fontsize=14, color="white")
ax2.set_ylim(0, max(speedups) * 1.15)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.grid(axis="y", linestyle="--", alpha=0.3)

# Add benchmark parameters as text
benchmark_info = (
    "N-Body Simulation Benchmark\n"
    "2000 bodies, 1000 iterations\n"
    "Gravitational physics simulation"
)
fig.text(0.5, 0.95, benchmark_info, ha="center", fontsize=16, color="white")

# Add a footer with implementation details
implementation_details = (
    "Python: NumPy vectorized implementation\n"
    "Go: Parallel implementation with optimized memory layout\n"
    "Mojo: SIMD-optimized implementation with parallel execution"
)
fig.text(0.5, 0.02, implementation_details, ha="center", fontsize=10, color="#BBBBBB")

# Adjust layout and save
plt.subplots_adjust(top=0.85, bottom=0.15, left=0.1, right=0.9)
plt.savefig(
    "benchmark_comparison.png",
    dpi=300,
    bbox_inches="tight",
    facecolor=fig.get_facecolor(),
)
print("Visualization saved as benchmark_comparison.png")

# Don't show the plot automatically to avoid blocking the script
# plt.show()
