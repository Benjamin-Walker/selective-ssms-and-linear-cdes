import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({"font.size": 12})

fig, axs = plt.subplots(1, 2, figsize=(10, 6))
lines = []
labels = []

# Define a list of markers
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
    labels_data = np.load(f"data_dir/labels_{dim}.npy")
    rmse_predicting_zero = np.mean(labels_data**2) ** 0.5
    rmse_lcde_list = []
    for run in range(5):
        rmse_lcde_list.append(
            np.load(f"outputs/lin_cde_mse_{dim}_run_{run}.npy") ** 0.5
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
                # Loop through outputs to outputs4 to collect RMSE data
                for run in range(5):
                    rmse = np.load(
                        f"outputs/rmse_100_{model_name}_{nonlinear}_{depth}_0.0001_256_{dim}_run_{run}.npy"
                    )
                    steps = np.load(
                        f"outputs/steps_100_{model_name}_{nonlinear}_{depth}_0.0001_256_{dim}_run_{run}.npy"
                    )
                    rmse_list.append(rmse)
                    steps_list.append(steps)

                # Compute mean and range of RMSE values across folders
                rmse_mean = np.mean(rmse_list, axis=0)
                rmse_min = np.min(rmse_list, axis=0)
                rmse_max = np.max(rmse_list, axis=0)
                rmse_range = rmse_max - rmse_min

                # Plot the mean RMSE
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

                # Plot mean RMSE
                (line,) = axs[i].semilogy(
                    steps_list[0],  # Using steps from the first folder for x-axis
                    rmse_mean,
                    color=CB_color_cycle[color_index],
                    marker=markers[marker_index],
                    markevery=mark_every,
                    markersize=8,
                )

                # Optionally, you can plot the range as a shaded region or error bars:
                axs[i].fill_between(
                    steps_list[0],
                    rmse_min,
                    rmse_max,
                    color=CB_color_cycle[color_index],
                    alpha=0.2,  # Transparency to show range
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
    axs[i].set_xlabel("Training Steps")
    axs[i].set_ylabel("Validation RMSE")
    if dim == 2:
        title_text = "Area (2D)"
    else:
        title_text = "Volume (3D)"
    axs[i].set_title(title_text)

plt.tight_layout()
plt.subplots_adjust(top=0.80)
fig.legend(lines, labels, ncols=3, loc="upper center")
plt.savefig("rmse_subplots.pdf", dpi=300, format="pdf")
plt.clf()
