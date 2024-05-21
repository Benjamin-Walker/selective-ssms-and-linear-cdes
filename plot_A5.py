import matplotlib.pyplot as plt
import numpy as np


# Data for the plot (sequence lengths and minimum number of layers)
sequence_lengths = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20]
rnn_layers = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=float)
s4_layers = np.array([1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 4, 4, 4, np.nan], dtype=float)
# ids4_layers = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=float)
mamba_layers = np.array(
    [1, 2, 2, 2, 2, 3, 3, 3, 4, 4, np.nan, np.nan, np.nan, np.nan], dtype=float
)
transformer_layers = np.array(
    [2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, np.nan, np.nan, np.nan], dtype=float
)
linear_cde_layers = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=float)

# Offset each line slightly
offset = 0.10
linear_cde_layers += 0 * offset
s4_layers += 1 * offset
rnn_layers += 2 * offset
mamba_layers += 3 * offset
transformer_layers += 4 * offset

# Define colors for the lines
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

# Plotting
plt.figure(figsize=(8, 6))

# Shade horizontal regions
layer_values = [1, 2, 3, 4]
for i, layer in enumerate(layer_values):
    plt.axhspan(layer - offset, layer + 5 * offset, color="lightgrey", alpha=0.3)

# Plot lines with markers
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
    s4_layers,
    marker="^",
    linestyle="-",
    label="S4",
    color=colors[1],
    linewidth=2,
)
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

# Adding labels and title
plt.xlabel("Sequence Length", fontsize=12)
plt.ylabel("Min. Number of Blocks", fontsize=12)
plt.xticks(sequence_lengths, fontsize=10)
plt.yticks([1.2, 2.2, 3.2, 4.2], labels=[1, 2, 3, 4], fontsize=10)
plt.title(r"$A_5$", fontsize=14)
plt.legend(loc="center right", fontsize=10)

# Save and display the plot
plt.savefig("A5_plot_shaded_regions.pdf", format="pdf", dpi=300, bbox_inches="tight")
plt.show()
