"""
This module provides data loaders for in-memory datasets, specifically designed for toy
datasets and the A5 dataset used in experiments.

Classes:
    - InMemoryDataloader: A base class for creating data loaders from in-memory
        datasets, supporting batching, shuffling, and looping over data for training
        and validation.
    - ToyDataloader: Inherits from InMemoryDataloader to load and preprocess toy
        datasets from NumPy files, splitting them into training and validation sets.
    - A5Dataloader: Inherits from InMemoryDataloader to load and preprocess the A5
        dataset from CSV files, supporting variable sequence lengths and
        train-validation splits.

Usage:
    - Instantiate ToyDataloader or A5Dataloader with the required parameters.
    - Use the `train_loop` and `val_loop` methods to iterate over the training and
    validation data
      in batches, with options for shuffling and looping over epochs.

Notes:
    - The ToyDataloader expects data files in the directory "data_dir/toy_data" with
        filenames "data_{num}.npy" and "labels_{num}.npy".
    - The A5Dataloader expects CSV files in the directory "data_dir/illusion" with
        filenames formatted as "A5={length}.csv".
    - The `key` parameter is used for random shuffling in JAX to ensure reproducibility.
"""

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pandas as pd


class InMemoryDataloader:
    def __init__(self, num_train, num_val, train_x, train_y, val_x, val_y):
        self.num_train = num_train
        self.num_val = num_val
        self.train_x = train_x
        self.train_y = train_y
        self.val_x = val_x
        self.val_y = val_y

    def loop(self, batch_size, data, labels, *, key):
        size = data.shape[0]
        indices = jnp.arange(size)
        while True:
            perm_key, key = jr.split(key, 2)
            perm = jr.permutation(perm_key, indices)
            for X in self.loop_epoch(batch_size, data[perm], labels[perm]):
                yield X

    def loop_epoch(self, batch_size, data, labels):
        size = data.shape[0]
        indices = jnp.arange(size)
        start = 0
        end = batch_size
        while end < size:
            batch_indices = indices[start:end]
            yield data[batch_indices], labels[batch_indices]
            start = end
            end = start + batch_size
        batch_indices = indices[start:]
        yield data[batch_indices], labels[batch_indices]

    def train_loop(self, batch_size, epoch=False, *, key):
        if epoch:
            return self.loop_epoch(batch_size, self.train_x, self.train_y)
        else:
            return self.loop(batch_size, self.train_x, self.train_y, key=key)

    def val_loop(self, batch_size, epoch=False, *, key):
        if epoch:
            return self.loop_epoch(batch_size, self.val_x, self.val_y)
        else:
            return self.loop(batch_size, self.val_x, self.val_y, key=key)


class ToyDataloader(InMemoryDataloader):
    def __init__(self, num):
        with open(f"data_dir/toy_data/data_{num}.npy", "rb") as f:
            data = jnp.array(np.load(f))
        with open(f"data_dir/toy_data/labels_{num}.npy", "rb") as f:
            labels = jnp.array(np.load(f))
        N = data.shape[0]
        train_x = data[: int(0.8 * N)]
        train_y = labels[: int(0.8 * N)]
        val_x = data[int(0.8 * N) :]
        val_y = labels[int(0.8 * N) :]

        super().__init__(
            train_x.shape[0],
            val_x.shape[0],
            train_x,
            train_y,
            val_x,
            val_y,
        )


class A5Dataloader(InMemoryDataloader):
    def __init__(self, length, train_split, key):
        df = pd.read_csv(f"data_dir/illusion/A5={length}.csv")
        input_array = df["input"].str.split(" ", expand=True).astype(int).to_numpy()
        target_array = df["target"].str.split(" ", expand=True).astype(int).to_numpy()
        data = jnp.array(input_array)
        labels = jnp.array(target_array)
        N = data.shape[0]
        shuffle_idx = jr.permutation(key, jnp.arange(N), independent=True)
        data = data[shuffle_idx]
        labels = labels[shuffle_idx]
        train_x = data[: int(train_split * N)]
        train_y = labels[: int(train_split * N)]
        val_x = data[int(train_split * N) :]
        val_y = labels[int(train_split * N) :]

        super().__init__(
            train_x.shape[0],
            val_x.shape[0],
            train_x,
            train_y,
            val_x,
            val_y,
        )
