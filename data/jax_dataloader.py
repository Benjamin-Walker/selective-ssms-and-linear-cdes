import jax.numpy as jnp
import jax.random as jr
import numpy as np


class InMemoryDataloader:
    def __init__(self, num_train, num_test, train_x, train_y, test_x, test_y):
        self.num_train = num_train
        self.num_test = num_test
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y

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

    def test_loop(self, batch_size, epoch=False, *, key):
        if epoch:
            return self.loop_epoch(batch_size, self.test_x, self.test_y)
        else:
            return self.loop(batch_size, self.test_x, self.test_y, key=key)


class ToyDataloader(InMemoryDataloader):
    def __init__(self, num):
        with open(f"data/data_{num}.npy", "rb") as f:
            data = jnp.array(np.load(f))
        with open(f"data/labels_{num}.npy", "rb") as f:
            labels = jnp.array(np.load(f))
        N = data.shape[0]
        train_x = data[: int(0.8 * N)]
        train_y = labels[: int(0.8 * N)]
        test_x = data[int(0.8 * N) :]
        test_y = labels[int(0.8 * N) :]

        super().__init__(
            train_x.shape[0],
            test_x.shape[0],
            train_x,
            train_y,
            test_x,
            test_y,
        )
