import os

import diffrax as dfx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import sklearn.linear_model

from data.jax_dataloader import ToyDataloader


class LinearCDE:
    hidden_dim: int
    data_dim: int
    omega_dim: int
    xi_dim: int
    label_dim: int
    vf_A: jnp.array
    vf_B: jnp.array

    def __init__(
        self,
        hidden_dim,
        data_dim,
        omega_dim,
        xi_dim,
        label_dim,
        *,
        key,
    ):
        init_matrix_key, init_bias_key, vf_A_key, vf_B_key = jr.split(key, 4)
        self.hidden_dim = hidden_dim
        self.data_dim = data_dim
        self.label_dim = label_dim
        self.init_matrix = jr.normal(init_matrix_key, (hidden_dim, data_dim))
        self.init_bias = jr.normal(init_bias_key, (hidden_dim,))
        self.omega_dim = omega_dim
        self.xi_dim = xi_dim
        self.vf_A = jr.normal(vf_A_key, (hidden_dim, omega_dim, hidden_dim)) / (
            hidden_dim**0.5
        )
        self.vf_B = jr.normal(vf_B_key, (hidden_dim, xi_dim))

    def __call__(self, ts, omega_path, xi_path, x0):
        control_path = jnp.concatenate((omega_path, xi_path), axis=-1)
        control = dfx.LinearInterpolation(ts=ts, ys=control_path)
        y0 = self.init_matrix @ x0 + self.init_bias

        def func(t, y, args):
            return jnp.concatenate((jnp.dot(self.vf_A, y), self.vf_B), axis=-1)

        term = dfx.ControlTerm(func, control).to_ode()
        saveat = dfx.SaveAt(t1=True)
        solution = dfx.diffeqsolve(
            term,
            dfx.Tsit5(),
            0,
            1,
            0.01,
            y0,
            stepsize_controller=dfx.PIDController(atol=1e-2, rtol=1e-2, jump_ts=ts),
            saveat=saveat,
        )
        return solution.ys[-1]


def obtain_features_from_model(model, dataloader, batch_size, num_samples, label_dim):
    features = np.zeros((num_samples, model.hidden_dim))
    labels = np.zeros((num_samples, label_dim))
    start = 0
    end = start
    i = 0
    vmap_model = jax.vmap(model)
    for data in dataloader:
        i += 1
        print(f"Batch {i}")
        X, y = data
        ts = jnp.repeat(jnp.linspace(0.0, 1.0, X.shape[1])[None, :], batch_size, axis=0)
        input = jnp.concatenate((ts[..., None], X), axis=-1)
        out = vmap_model(ts, input, input, X[:, 0, :])
        end += len(out)
        features[start:end] = out
        labels[start:end] = y[:, None]
        start = end
    return features, labels


def train_linear(
    model,
    dataloader,
    num_train,
    num_test,
    label_dim,
    batch_size,
    *,
    key,
):

    featkey_train, featkey_test, key = jr.split(key, 3)

    features_train, labels_train = obtain_features_from_model(
        model,
        dataloader.train_loop(batch_size, epoch=True, key=key),
        batch_size,
        num_train,
        label_dim,
    )
    features_test, labels_test = obtain_features_from_model(
        model,
        dataloader.test_loop(batch_size, epoch=True, key=key),
        batch_size,
        num_test,
        label_dim,
    )

    clf = sklearn.linear_model.LinearRegression()
    clf.fit(features_train, labels_train)
    predictions = clf.predict(features_test)
    mse = jnp.mean((labels_test - predictions) ** 2)
    return mse


if __name__ == "__main__":
    key = jr.PRNGKey(2345)
    if not os.path.isdir("outputs"):
        os.mkdir("outputs")
    model_key, train_key = jr.split(key)
    hidden_dim = 256
    label_dim = 1
    batch_size = 4000
    for data_dim in [2, 3]:
        omega_dim = data_dim + 1
        xi_dim = data_dim + 1
        model = LinearCDE(
            hidden_dim, data_dim, omega_dim, xi_dim, label_dim, key=model_key
        )
        dataset = ToyDataloader(num=data_dim)
        mse = train_linear(
            model,
            dataset,
            num_train=dataset.num_train,
            num_test=dataset.num_test,
            label_dim=label_dim,
            batch_size=batch_size,
            key=train_key,
        )
        mse = np.array(mse)
        np.save(f"outputs/lin_cde_mse_{data_dim}.npy", mse)
        print(f"Data dim: {data_dim}, MSE: {mse}")
