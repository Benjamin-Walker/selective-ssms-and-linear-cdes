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

    with open(f"data/data_{dim}.npy", "wb") as f:
        np.save(f, data)
    with open(f"data/labels_{dim}.npy", "wb") as f:
        np.save(f, labels)
