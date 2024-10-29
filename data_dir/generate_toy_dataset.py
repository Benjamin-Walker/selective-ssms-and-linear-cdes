"""
Generates synthetic data of random walks in 2D and 3D, computes their path signatures,
and saves both the data and computed labels to files for further use.

This script performs the following steps:
- Generates random walk data for 2D and 3D cases.
- Computes the signatures of these random walks up to a specified depth.
- Extracts specific components from the signatures to serve as labels.
- Normalizes both the data and labels.
- Saves the generated data and labels into `.npy` files in the "data_dir/toy_data"
    directory.

Note:
- The script uses a fixed random seed for reproducibility.
"""

import os

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from signax.signature import signature


key = jr.PRNGKey(1234)
depth = 4

for dim in [2, 3]:

    data = jr.normal(key, shape=(100000, 100, dim))
    data = jnp.round(data)
    data = jnp.cumsum(data, axis=1)
    data = data / jnp.max(jnp.abs(data))

    vmap_calc_sig = jax.vmap(signature, in_axes=(0, None))
    labels = vmap_calc_sig(data, depth)

    if dim == 2:
        labels = labels[1][:, 0, 1]
    elif dim == 3:
        labels = labels[2][:, 0, 1, 2]

    labels = labels / jnp.max(jnp.abs(labels))

    data = np.array(data)
    labels = np.array(labels)

    if not os.path.exists("data_dir/toy_data"):
        os.makedirs("data_dir/toy_data")

    with open(f"data_dir/toy_data/data_{dim}.npy", "wb") as f:
        np.save(f, data)
    with open(f"data_dir/toy_data/labels_{dim}.npy", "wb") as f:
        np.save(f, labels)
