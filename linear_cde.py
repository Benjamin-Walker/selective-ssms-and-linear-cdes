import os
import time

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import sklearn.linear_model

from data_dir.jax_dataloader import A5Dataloader, ToyDataloader


class Embedding(eqx.Module):
    weights: jnp.array

    def __init__(self, num_embeddings: int, embedding_dim: int, *, key):
        self.weights = jax.random.normal(key, (num_embeddings, embedding_dim))

    def __call__(self, x):
        return self.weights[x]


def adaptive_ode_solve(func, control_path, y0, ts):
    control = dfx.LinearInterpolation(ts=ts, ys=control_path)
    term = dfx.ControlTerm(func, control).to_ode()
    saveat = dfx.SaveAt(ts=ts)
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
    return solution.ys


def scan_solve(func, control_path, y0):
    length = control_path.shape[0]  # Check same

    def scan_fn(y, i):
        vec = func(None, y, None)
        y = y + (vec @ control_path[i]) * (1.0 / length)
        return y, y

    _, ys = jax.lax.scan(scan_fn, y0, jnp.arange(1, length))
    ys = jnp.concatenate((y0[None, :], ys), axis=0)
    return ys


class LinearCDE(eqx.Module):
    hidden_dim: int
    data_dim: int
    omega_dim: int
    xi_dim: int
    label_dim: int
    init_matrix: jnp.array
    init_bias: jnp.array
    vf_A: jnp.array
    vf_B: jnp.array
    adaptive_ode: bool
    continuous_output: bool

    def __init__(
        self,
        hidden_dim,
        data_dim,
        omega_dim,
        xi_dim,
        label_dim,
        continuous_output=False,
        adaptive_ode=True,
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
        self.adaptive_ode = adaptive_ode
        self.continuous_output = continuous_output

    def __call__(self, X):
        ts = jnp.linspace(0, 1, X.shape[0])
        inp = jnp.concatenate((ts[..., None], X), axis=-1)
        x0 = X[0, :]
        control_path = jnp.concatenate((inp, inp), axis=-1)
        y0 = self.init_matrix @ x0 + self.init_bias

        def func(t, y, args):
            return jnp.concatenate((self.vf_A @ y, self.vf_B), axis=-1)

        if self.adaptive_ode:
            ys = adaptive_ode_solve(func, control_path, y0, ts)
        else:
            ys = scan_solve(func, control_path, y0)

        if self.continuous_output:
            return ys

        return ys[-1]


class A5LinearCDE(eqx.Module):
    embedding: Embedding
    LCDE: LinearCDE
    norm: eqx.nn.LayerNorm
    drop: eqx.nn.Dropout
    linear: eqx.nn.Linear
    label_dim: int

    def __init__(self, hidden_dim, data_dim, omega_dim, xi_dim, label_dim, *, key):
        self.embedding = Embedding(label_dim, data_dim, key=key)
        self.LCDE = LinearCDE(
            hidden_dim,
            data_dim,
            omega_dim,
            xi_dim,
            label_dim,
            continuous_output=True,
            adaptive_ode=False,
            key=key,
        )
        self.norm = eqx.nn.LayerNorm(hidden_dim)
        self.drop = eqx.nn.Dropout(p=0.1)
        self.linear = eqx.nn.Linear(hidden_dim, label_dim, key=key)
        self.label_dim = label_dim

    def __call__(self, X, enable_dropout, key):
        X = jax.vmap(self.embedding)(X)
        ys = self.LCDE(X)
        ys = jax.vmap(self.norm)(ys)
        ys = self.drop(ys, inference=not enable_dropout, key=key)
        return jax.vmap(lambda x: jax.nn.softmax(self.linear(x)))(ys)


def obtain_features_from_model(model, dataloader, num_samples, label_dim):
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
        out = vmap_model(X)
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
    *,
    key,
):
    featkey_train, featkey_test, key = jr.split(key, 3)

    features_train, labels_train = obtain_features_from_model(
        model,
        dataloader.train_loop(batch_size, epoch=True, key=key),
        num_train,
        label_dim,
    )
    features_test, labels_test = obtain_features_from_model(
        model,
        dataloader.test_loop(batch_size, epoch=True, key=key),
        num_test,
        label_dim,
    )

    clf = sklearn.linear_model.LinearRegression()
    clf.fit(features_train, labels_train)
    predictions = clf.predict(features_test)
    mse = jnp.mean((labels_test - predictions) ** 2)
    return mse


def train_model(
    model,
    dataloader_length2,
    dataloader,
    num_steps,
    print_steps,
    learning_rate,
    batch_size,
    *,
    key,
):
    optimizer = optax.adamw(learning_rate, weight_decay=0.01)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_value_and_grad
    def loss_fn(model, X, y, keys):
        preds = jax.vmap(model, in_axes=(0, None, 0))(X, True, keys)
        loss = -jnp.sum(y * jnp.log(preds + 1e-10), axis=-1)
        return jnp.mean(loss)

    @eqx.filter_jit
    def train_step(model, opt_state, X, y, keys):
        loss, grads = loss_fn(model, X, y, keys)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    step = 0
    total_loss = 0
    steps = []
    test_accs = []
    num_classes = model.label_dim
    start = time.time()
    for data, length2_data in zip(
        dataloader.train_loop(batch_size, key=key),
        dataloader_length2.train_loop(batch_size // 10, key=key),
    ):
        X, y = data
        y = jax.vmap(lambda y: jax.nn.one_hot(y, num_classes))(y)
        key, *subkeys = jr.split(key, num=batch_size + 1)
        subkeys = jnp.array(subkeys)
        model, opt_state, loss = train_step(model, opt_state, X, y, subkeys)
        total_loss += loss
        X, y = length2_data
        y = jax.vmap(lambda y: jax.nn.one_hot(y, num_classes))(y)
        key, *subkeys = jr.split(key, num=batch_size // 10 + 1)
        subkeys = jnp.array(subkeys)
        model, opt_state, loss = train_step(model, opt_state, X, y, subkeys)
        if step % print_steps == 0:
            end = time.time()
            train_acc = 0
            train_num = 0
            # Test on a subset of the train data
            for _, data in zip(
                range(5), dataloader.train_loop(400, epoch=True, key=key)
            ):
                X, y = data
                key, *subkeys = jr.split(key, num=400 + 1)
                subkeys = jnp.array(subkeys)
                pred = jax.vmap(model, in_axes=(0, None, 0))(X, False, subkeys)
                train_acc += jnp.sum(jnp.argmax(pred, axis=-1) == y)
                train_num += y.shape[0] * y.shape[1]
            test_acc = 0
            test_num = 0
            for data in dataloader.test_loop(400, epoch=True, key=key):
                X, y = data
                key, *subkeys = jr.split(key, num=400 + 1)
                subkeys = jnp.array(subkeys)
                pred = jax.vmap(model, in_axes=(0, None, 0))(X, False, subkeys)
                test_acc += jnp.sum(jnp.argmax(pred, axis=-1) == y)
                test_num += y.shape[0] * y.shape[1]
            if step == 0:
                total_loss *= print_steps
            print(
                f"Step {step}, Loss: {total_loss / print_steps:.4f}, "
                f"Train Acc: {train_acc / train_num:.4f}, "
                f"Test Acc: {test_acc / test_num:.4f}, Time: {end - start:.2f}s"
            )
            steps.append(step)
            test_accs.append(test_acc / test_num)
            start = time.time()
            total_loss = 0
        if step >= num_steps:
            break
        step += 1

    return model, steps, test_accs


if __name__ == "__main__":
    key = jr.PRNGKey(2345)
    experiment = "toy"

    if experiment == "toy":
        if not os.path.isdir("outputs_toy"):
            os.mkdir("outputs_toy")
        model_key, train_key = jr.split(key)
        hidden_dim = 256
        label_dim = 1
        batch_size = 4000
        for run in range(5):
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
                    key=train_key,
                )
                mse = np.array(mse)
                np.save(f"outputs_toy/lin_cde_mse_{data_dim}_run_{run}.npy", mse)
                print(f"Data dim: {data_dim}, MSE: {mse}")
    elif experiment == "A5":
        if not os.path.isdir("outputs_A5"):
            os.mkdir("outputs_A5")
        model_key, train_key, length_2_data_key, data_key, key = jr.split(key, 5)
        hidden_dim = 110
        label_dim = 60
        batch_size = 32
        data_dim = 255
        omega_dim = data_dim + 1
        xi_dim = data_dim + 1
        length = 20
        model = A5LinearCDE(
            hidden_dim, data_dim, omega_dim, xi_dim, label_dim, key=model_key
        )
        dataset_length2 = A5Dataloader(length=2, train_split=1.0, key=length_2_data_key)
        dataset = A5Dataloader(length=length, train_split=0.8, key=data_key)
        _, steps, test_accs = train_model(
            model,
            dataset_length2,
            dataset,
            num_steps=1000000,
            print_steps=10000,
            learning_rate=1e-4,
            batch_size=batch_size,
            key=train_key,
        )
        np.save("outputs_A5/linear_cde_steps.npy", steps)
        np.save("outputs_A5/linear_cde_test_accs.npy", test_accs)
