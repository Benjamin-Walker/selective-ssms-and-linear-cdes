import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({"font.size": 12})

plot_names = {
    "s5": "S5",
    "mamba": "Mamba",
}

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
    labels_data = np.load(f"data/labels_{dim}.npy")
    rmse_predicting_zero = np.mean(labels_data**2) ** 0.5
    rmse_linear_cde = np.load(f"outputs/lin_cde_mse_{dim}.npy") ** 0.5
    marker_index = 0
    color_index = 0
    for model_name in ["mamba", "s5"]:
        if model_name == "mamba":
            c_choices = [True]
        else:
            c_choices = [False]
        for c_dependent in c_choices:
            for depth in [1, 2]:
                if depth == 2:
                    nonlinear_choices = [False, True]
                else:
                    nonlinear_choices = [False]
                for nonlinear in nonlinear_choices:
                    rmse = np.load(
                        f"outputs/rmse_100_{model_name}_{c_dependent}_{nonlinear}_{depth}_0.0001_256_{dim}.npy"
                    )
                    steps = np.load(
                        f"outputs/steps_100_{model_name}_{c_dependent}_{nonlinear}_{depth}_0.0001_256_{dim}.npy"
                    )
                    label = f"{plot_names[model_name]} Depth {depth}"
                    if nonlinear:
                        label += " Nonlinear"
                    if model_name == "s5":
                        if depth == 2:
                            if nonlinear:
                                mark_every = [12, -1]
                            else:
                                mark_every = [22, -1]
                        else:
                            mark_every = [12, 90]
                    else:
                        mark_every = [12, -1]
                    (line,) = axs[i].semilogy(
                        steps,
                        rmse,
                        color=CB_color_cycle[color_index],
                        marker=markers[marker_index],
                        markevery=mark_every,
                        markersize=8,
                    )
                    if dim == 2:
                        lines.append(line)
                        labels.append(label)
                    marker_index = (marker_index + 1) % len(markers)
                    color_index = (color_index + 1) % len(CB_color_cycle)
    line = axs[i].hlines(rmse_predicting_zero, 0, 1e6, color="black", linestyle="--")
    if dim == 2:
        lines.append(line)
        labels.append("Predicting Zero")
    line = axs[i].hlines(rmse_linear_cde, 0, 1e6, color="red", linestyle="--")
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
