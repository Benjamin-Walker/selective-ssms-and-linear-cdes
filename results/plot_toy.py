"""
This script visualizes the validation RMSE of several models (Mamba, S4, and Linear CDE)
on a toy dataset for both 2D and 3D cases. Each plot shows the RMSE as a function of
training steps, with comparisons made between different model depths and non-linear
configurations.

The RMSE values are computed across multiple runs, and the mean RMSE with ranges (min
to max) is plotted for each model configuration. Horizontal dashed lines represent the
RMSE for predicting zero and the random Linear CDE model as baselines.

Plot Details:
    - Each subplot corresponds to a different task: "Area (2D)" and "Volume (3D)".
    - Models (Mamba and S4) are evaluated with varying depths (1 and 2) and non-linear
        settings.
    - Shaded regions indicate the range of RMSE values across runs.
    - Unique markers and colors differentiate each model configuration.

Output:
    - Saves the plot as "rmse_subplots.pdf" in the "results" directory.
"""

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({"font.size": 14})

fig, axs = plt.subplots(1, 2, figsize=(10.5, 6.5))
lines = []
labels = []

markers = ["o", "s", "D", "^", "p", "v", "P", "*", "X"]
CB_color_cycle = [
    "#377eb8",
    "#ff7f00",
    "#4daf4a",
    "#a65628",
    "#984ea3",
    "#999999",
    "#e41a1c",
    "#dede00",
]

for i, dim in enumerate([2, 3]):
    labels_data = np.load(f"data_dir/toy_data/labels_{dim}.npy")
    rmse_predicting_zero = np.mean(labels_data**2) ** 0.5
    rmse_lcde_list = []
    for run in range(5):
        rmse_lcde_list.append(
            np.load(f"results/outputs_toy/lin_cde_mse_{dim}_run_{run}.npy") ** 0.5
        )
    marker_index = 0
    color_index = 0
    for model_name in ["Mamba", "S4"]:
        for depth in [1, 2]:
            if depth == 2:
                nonlinear_choices = [False, True]
            else:
                nonlinear_choices = [False]
            for nonlinear in nonlinear_choices:
                rmse_list = []
                steps_list = []
                for run in range(5):
                    rmse = np.load(
                        f"results/outputs_toy/rmse_100_{model_name}_{nonlinear}_{depth}_"
                        f"0.0001_256_{dim}_run_{run}.npy"
                    )
                    steps = np.load(
                        f"results/outputs_toy/steps_100_{model_name}_{nonlinear}_{depth}"
                        f"_0.0001_256_{dim}_run_{run}.npy"
                    )
                    rmse_list.append(rmse)
                    steps_list.append(steps)

                rmse_mean = np.mean(rmse_list, axis=0)
                rmse_min = np.min(rmse_list, axis=0)
                rmse_max = np.max(rmse_list, axis=0)
                rmse_range = rmse_max - rmse_min

                label = f"{model_name} Depth {depth}"
                if nonlinear:
                    label += " Nonlinear"
                if model_name == "S4":
                    if depth == 2:
                        if nonlinear:
                            mark_every = [12, -1]
                        else:
                            mark_every = [22, -1]
                    else:
                        mark_every = [12, 90]
                else:
                    if nonlinear:
                        mark_every = [21, -6]
                    else:
                        mark_every = [12, -1]

                (line,) = axs[i].semilogy(
                    steps_list[0],
                    rmse_mean,
                    color=CB_color_cycle[color_index],
                    marker=markers[marker_index],
                    markevery=mark_every,
                    markersize=8,
                )

                axs[i].fill_between(
                    steps_list[0],
                    rmse_min,
                    rmse_max,
                    color=CB_color_cycle[color_index],
                    alpha=0.2,
                )

                if dim == 2:
                    lines.append(line)
                    labels.append(label)
                marker_index = (marker_index + 1) % len(markers)
                color_index = (color_index + 1) % len(CB_color_cycle)

    line = axs[i].hlines(rmse_predicting_zero, 0, 990000, color="black", linestyle="--")
    if dim == 2:
        lines.append(line)
        labels.append("Predicting Zero")
    rmse_lcde_mean = np.mean(rmse_lcde_list)
    rmse_lcde_min = np.min(rmse_lcde_list)
    rmse_lcde_max = np.max(rmse_lcde_list)
    line = axs[i].hlines(rmse_lcde_mean, 0, 990000, color="red", linestyle="--")
    axs[i].fill_between(
        steps_list[0], rmse_lcde_min, rmse_lcde_max, color="red", alpha=0.2
    )
    if dim == 2:
        lines.append(line)
        labels.append("Random Linear CDE")
    axs[i].set_ylim([1e-4, 0.15])
    axs[i].set_xlabel("Training Steps", fontsize=28)
    if dim == 2:
        axs[i].set_ylabel("Validation RMSE", fontsize=28)
        title_text = "Area (2D)"
    else:
        title_text = "Volume (3D)"
    axs[i].set_title(title_text, fontsize=28)

plt.tight_layout()
plt.subplots_adjust(top=0.75)
fig.legend(lines, labels, ncols=3, loc="upper center", fontsize=16)
plt.savefig("results/rmse_subplots.pdf", dpi=300, format="pdf")
plt.clf()
