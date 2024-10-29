"""
This script visualizes the minimum number of blocks required by various
sequence-to-sequence models (RNN, S4, Mamba, Transformer, and Linear CDE) as a function
of sequence length on the A_5 benchmark.

Each model's number of blocks is plotted against different sequence lengths, with layers
shaded in light grey to delineate the discrete layer thresholds.

Plot Details:
    - The minimum number of blocks required for each model is plotted against sequence
        length.
    - Each model is represented with a unique marker, color, and line style.
    - Light grey regions highlight distinct layer intervals.

Output:
    - Saves the plot as "A5_plot_shaded_regions.pdf" in the "results" directory.
"""

import matplotlib.pyplot as plt
import numpy as np


sequence_lengths = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20]
rnn_layers = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=float)
s4_layers = np.array([1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 4, 4, 4, np.nan], dtype=float)
mamba_layers = np.array(
    [1, 2, 2, 2, 2, 3, 3, 3, 4, 4, np.nan, np.nan, np.nan, np.nan], dtype=float
)
transformer_layers = np.array(
    [1, 1, 1, 1, 2, 3, 3, 3, 4, 4, 4, 4, np.nan, np.nan], dtype=float
)
linear_cde_layers = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=float)

offset = 0.10
transformer_layers += 0 * offset
linear_cde_layers += 1 * offset
s4_layers += 2 * offset
rnn_layers += 3 * offset
mamba_layers += 4 * offset

colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

plt.figure(figsize=(8, 6))

layer_values = [1, 2, 3, 4]
for i, layer in enumerate(layer_values):
    plt.axhspan(layer - offset, layer + 5 * offset, color="lightgrey", alpha=0.3)

plt.plot(
    sequence_lengths,
    mamba_layers,
    marker="s",
    linestyle="-",
    label="Mamba",
    color=colors[3],
    linewidth=2,
)
plt.plot(
    sequence_lengths,
    s4_layers,
    marker="^",
    linestyle="-",
    label="S4",
    color=colors[1],
    linewidth=2,
)
plt.plot(
    sequence_lengths,
    transformer_layers,
    marker="o",
    linestyle="-",
    label="Transformer",
    color=colors[4],
    linewidth=2,
)
plt.plot(
    sequence_lengths,
    rnn_layers,
    marker="v",
    linestyle="-",
    label="RNN",
    color=colors[0],
    linewidth=2,
)
plt.plot(
    sequence_lengths,
    linear_cde_layers,
    marker="x",
    linestyle="-",
    label="Linear CDE",
    color=colors[5],
    linewidth=2,
)

plt.xlabel("Sequence Length", fontsize=20)
plt.ylabel("Minimum # Blocks Required", fontsize=20)
plt.xticks(sequence_lengths, fontsize=14)
plt.yticks([1.2, 2.2, 3.2, 4.2], labels=[1, 2, 3, 4], fontsize=14)
plt.legend(loc="center right", fontsize=18)

plt.savefig(
    "results/A5_plot_shaded_regions.pdf", format="pdf", dpi=300, bbox_inches="tight"
)
plt.show()
