"""
This module provides implementations and utilities for running experiments with Linear
Controlled Differential Equations (CDEs) using JAX and Equinox.

It includes classes and functions for:

- Defining embedding layers for discrete inputs.
- Implementing Linear CDE models.
- Training models on the toy dataset and the A5 dataset.
- Utility functions for solving CDEs, extracting features, and training models.

Classes:
    - Embedding: Embedding layer for mapping discrete indices to dense vectors.
    - LinearCDE: Implements a Linear CDE model.
    - A5LinearCDE: Sequence-to-sequence model for the A5 dataset using a Linear CDE.

Functions:
    - adaptive_cde_solve: Approximates a CDE adaptively using the Tsit5 ODE solver.
    - scan_cde_solve: Approximates a CDE using Euler discretisation.
    - obtain_features_from_model: Extracts features and labels from a model using
        batched data.
    - train_linear: Trains a linear regression model on features extracted from a model.
    - train_model: Trains a model using batched data and stochastic gradient descent.
    - run_lcde_toy_experiment: Runs the toy experiment with a Linear CDE model.
    - run_lcde_A5_experiment: Runs the A5 experiment with a Linear CDE model.
"""

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
    """
    Embedding layer for mapping discrete indices to dense vectors.

    Args:
        num_embeddings (int): Number of unique embeddings (vocabulary sise).
        embedding_dim (int): Dimensionality of each embedding vector.
        key (jax.random.PRNGKey): PRNG key for initialising weights.

    Returns:
        Embedding vectors corresponding to the input indices.
    """

    weights: jnp.ndarray

    def __init__(self, num_embeddings: int, embedding_dim: int, *, key):
        self.weights = jax.random.normal(key, (num_embeddings, embedding_dim))

    def __call__(self, x):
        return self.weights[x]


def adaptive_cde_solve(func, control_path, y0, ts):
    """
    Approximates a CDE adaptively using Tsit5.

    Args:
        func: The function defining the CDE dynamics.
        control_path: The control path (values for the input function) over time.
        y0: Initial state of the system.
        ts: Time steps at which to evaluate the solution.

    Returns:
        Solution of the ODE evaluated at each time step in ts.
    """
    control = dfx.LinearInterpolation(ts=ts, ys=control_path)
    term = dfx.ControlTerm(func, control).to_ode()
    saveat = dfx.SaveAt(ts=ts)
    solution = dfx.diffeqsolve(
        term,
        dfx.Tsit5(),
        ts[0],
        ts[-1],
        0.01,
        y0,
        stepsize_controller=dfx.PIDController(atol=1e-2, rtol=1e-2, jump_ts=ts),
        saveat=saveat,
    )
    return solution.ys


def scan_cde_solve(func, control_path, y0):
    """
    Approximates a CDE by scanning over an Euler discretisation.

    Args:
        func: The function defining the CDE dynamics.
        control_path: The control path as a sequence of control values over discrete
                        steps.
        y0: Initial state of the system.

    Returns:
        Array of states over the time steps of control_path.
    """
    length = control_path.shape[0]

    def scan_fn(y, i):
        vec = func(None, y, None)
        y = y + (vec @ control_path[i]) * (1.0 / length)
        return y, y

    _, ys = jax.lax.scan(scan_fn, y0, jnp.arange(1, length))
    ys = jnp.concatenate((y0[None, :], ys), axis=0)
    return ys


class LinearCDE(eqx.Module):
    """
    Implements a Linear Controlled Differential Equation (CDE) model.

    This model defines a linear CDE with initial parameters and provides
    an option for adaptive or fixed-step ODE solving. Given an input path,
    it outputs either a continuous solution over the entire path or just the
    final state.

    A general form of a Linear CDE is given by:
        y_t = y_0 + ∫_0^t A y_s dω_s + ∫_0^t B dξ_s
    where:
        - A is a 3D tensor of shape (hidden_dim, omega_dim, hidden_dim),
        - B is a 2D tensor of shape (hidden_dim, xi_dim),
        - y_t is the hidden state at time t,
        - y_0 is the initial hidden state,
        - ω_t is a path embedding of the input path X at time t,
        - ξ_t is a path embedding of the input path X at time t.

    In this implementation, we fix ξ_t = ω_t = [t, X_t].

    Args:
        hidden_dim (int): Dimension of the hidden state.
        input_dim (int): Dimension of the input data.
        output_dim (int): Dimension of the output data.
        continuous_output (bool): If True, returns the entire output path;
            otherwise, returns only the final state. Default is False.
        adaptive_ode (bool): If True, uses an adaptive ODE solver;
            otherwise, uses a fixed-step scan solver. Default is True.
        key (jax.random.PRNGKey): PRNG key for parameter initialisation.

    Returns:
        If `continuous_output` is True, returns the complete output trajectory.
        Otherwise, returns the final output state.

    """

    hidden_dim: int
    input_dim: int
    omega_dim: int
    xi_dim: int
    output_dim: int
    init_matrix: jnp.array
    init_bias: jnp.array
    vf_A: jnp.array
    vf_B: jnp.array
    adaptive_ode: bool
    continuous_output: bool

    def __init__(
        self,
        hidden_dim,
        input_dim,
        output_dim,
        continuous_output=False,
        adaptive_ode=True,
        *,
        key,
    ):
        init_matrix_key, init_bias_key, vf_A_key, vf_B_key = jr.split(key, 4)
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.init_matrix = jr.normal(init_matrix_key, (hidden_dim, input_dim))
        self.init_bias = jr.normal(init_bias_key, (hidden_dim,))
        self.omega_dim = input_dim + 1
        self.xi_dim = input_dim + 1
        self.vf_A = jr.normal(vf_A_key, (hidden_dim, self.omega_dim, hidden_dim)) / (
            hidden_dim**0.5
        )
        self.vf_B = jr.normal(vf_B_key, (hidden_dim, self.xi_dim))
        self.adaptive_ode = adaptive_ode
        self.continuous_output = continuous_output

    def __call__(self, X):
        ts = jnp.linspace(0, 1, X.shape[0])
        x0 = X[0, :]
        y0 = self.init_matrix @ x0 + self.init_bias

        # Construct ξ_t and ω_t and stack them together
        inp = jnp.concatenate((ts[..., None], X), axis=-1)
        control_path = jnp.concatenate((inp, inp), axis=-1)

        def func(t, y, args):
            return jnp.concatenate((self.vf_A @ y, self.vf_B), axis=-1)

        if self.adaptive_ode:
            ys = adaptive_cde_solve(func, control_path, y0, ts)
        else:
            ys = scan_cde_solve(func, control_path, y0)

        if self.continuous_output:
            return ys

        return ys[-1]


class A5LinearCDE(eqx.Module):
    """
    Implements a sequence-to-sequence model for the A5 dataset using a Linear CDE.

    Args:
        hidden_dim (int): Dimension of the hidden state and embedding vectors.
        input_dim (int): Dimension of the input data.
        omega_dim (int): Dimension of the omega path.
        xi_dim (int): Dimension of the xi path.
        output_dim (int): Dimension of the final output data.
        key (jax.random.PRNGKey): PRNG key for initialising the parameters.

    Call Args:
        X (jnp.array): Input data array, to be embedded and processed by the CDE.
        enable_dropout (bool): If True, applies dropout during forward pass.
        key (jax.random.PRNGKey): PRNG key for dropout.

    Returns:
        jnp.array: Softmax-transformed output vector of shape `(output_dim,)`.
    """

    embedding: Embedding
    hidden_dim: int
    LCDE: LinearCDE
    norm: eqx.nn.LayerNorm
    drop: eqx.nn.Dropout
    linear_mix: eqx.nn.Linear
    linear_out: eqx.nn.Linear
    output_dim: int

    def __init__(self, hidden_dim, input_dim, output_dim, *, key):
        self.embedding = Embedding(input_dim, hidden_dim, key=key)
        self.hidden_dim = hidden_dim
        self.LCDE = LinearCDE(
            hidden_dim,
            hidden_dim,
            output_dim,
            continuous_output=True,
            adaptive_ode=False,
            key=key,
        )
        self.norm = eqx.nn.LayerNorm(hidden_dim)
        self.drop = eqx.nn.Dropout(p=0.1)
        self.linear_mix = eqx.nn.Linear(hidden_dim, hidden_dim, key=key)
        self.linear_out = eqx.nn.Linear(hidden_dim, output_dim, key=key)
        self.output_dim = output_dim

    def __call__(self, X, enable_dropout, key):
        drop1, drop2, key = jr.split(key, 3)
        X = jax.vmap(self.embedding)(X)
        residual = X
        ys = self.LCDE(X)
        ys = self.drop(ys, inference=not enable_dropout, key=drop1)
        ys = jax.vmap(lambda x: jax.nn.relu(self.linear_mix(x)))(ys)
        ys = self.drop(ys, inference=not enable_dropout, key=drop2)
        ys = ys + residual
        ys = jax.vmap(self.norm)(ys)
        return jax.vmap(lambda x: jax.nn.softmax(self.linear_out(x)))(ys)


def obtain_features_from_model(model, dataloader, num_samples, output_dim):
    """
    Extracts features and labels from a model using batched data from a dataloader.

    Args:
        model: The model to obtain features from. Assumes the model's `__call__`
               method outputs feature representations.
        dataloader: An iterable that provides batches of (input, label) pairs.
        num_samples (int): Total number of samples to process.
        output_dim (int): Dimension of the output labels.

    Returns:
        Tuple[np.array, np.array]: A tuple containing:
            - features (np.array): Array of extracted features of shape
              (num_samples, model.hidden_dim).
            - labels (np.array): Array of labels of shape (num_samples, output_dim).
    """
    features = np.zeros((num_samples, model.hidden_dim))
    labels = np.zeros((num_samples, output_dim))
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
    num_val,
    output_dim,
    batch_size,
    *,
    key,
):
    """
    Trains a linear regression model on features extracted from a model.

    This function extracts features from a given model for both training and
    validation data, trains a linear regression model on the training features,
    and evaluates it on validation features using mean squared error (MSE).

    Args:
        model: The model from which to extract features.
        dataloader: Dataloader with `train_loop` and `val_loop` methods for
                    providing batched training and validation data.
        num_train (int): Number of training samples.
        num_val (int): Number of validation samples.
        output_dim (int): Dimension of the output labels.
        batch_size (int): Size of each batch during feature extraction.
        key (jax.random.PRNGKey): PRNG key for randomness, split internally for feature
                                    extraction.

    Returns:
        float: Mean squared error of the linear model's predictions on validation data.
    """
    featkey_train, featkey_val, key = jr.split(key, 3)

    features_train, labels_train = obtain_features_from_model(
        model,
        dataloader.train_loop(batch_size, epoch=True, key=key),
        num_train,
        output_dim,
    )
    features_val, labels_val = obtain_features_from_model(
        model,
        dataloader.val_loop(batch_size, epoch=True, key=key),
        num_val,
        output_dim,
    )

    clf = sklearn.linear_model.LinearRegression()
    clf.fit(features_train, labels_train)
    predictions = clf.predict(features_val)
    mse = jnp.mean((labels_val - predictions) ** 2)
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
    """
    Trains a model using batched data, logging progress at regular intervals.

    This function trains the provided model using an AdamW optimiser, performing
    updates with a custom loss function and recording accuracy on training and
    validation sets at specified intervals. Training stops when the validation
    accuracy exceeds 90% or the maximum number of steps is reached.

    Args:
        model: The model to be trained.
        dataloader_length2: An auxiliary dataloader for an additional dataset
                            (e.g. length 2 examples from the A5 dataset).
        dataloader: Main dataloader for obtaining training and validation data.
        num_steps (int): Maximum number of training steps.
        print_steps (int): Interval (in steps) at which to print training progress and
                           evaluate accuracy.
        learning_rate (float): Learning rate for the optimiser.
        batch_size (int): Number of samples in each training batch.
        key (jax.random.PRNGKey): PRNG key for randomness in dropout and data splitting.

    Returns:
        tuple: A tuple containing:
            - model: The trained model.
            - steps (list): List of steps at which training progress was logged.
            - val_accs (list): List of validation accuracies at each logged step.

    Logs:
        Prints training loss, training accuracy, validation accuracy, and
        elapsed time at each `print_steps` interval.
    """
    params = eqx.filter(model, eqx.is_inexact_array)
    optimizer = optax.adamw(learning_rate, weight_decay=0.01)
    opt_state = optimizer.init(params)

    @eqx.filter_value_and_grad
    def loss_fn(model, X, y, keys):
        preds = jax.vmap(model, in_axes=(0, None, 0))(X, True, keys)
        loss = -jnp.sum(y * jnp.log(preds + 1e-10), axis=-1)
        return jnp.mean(loss)

    @eqx.filter_jit
    def train_step(model, opt_state, X, y, keys):
        loss, grads = loss_fn(model, X, y, keys)
        params = eqx.filter(model, eqx.is_inexact_array)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    step = 0
    total_loss = 0
    steps = []
    val_accs = []
    num_classes = model.output_dim
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
            val_acc = 0
            val_num = 0
            for data in dataloader.val_loop(400, epoch=True, key=key):
                X, y = data
                key, *subkeys = jr.split(key, num=400 + 1)
                subkeys = jnp.array(subkeys)
                pred = jax.vmap(model, in_axes=(0, None, 0))(X, False, subkeys)
                val_acc += jnp.sum(jnp.argmax(pred, axis=-1) == y)
                val_num += y.shape[0] * y.shape[1]
            if step == 0:
                total_loss *= print_steps
            print(
                f"Step {step}, Loss: {total_loss / print_steps:.4f}, "
                f"Train Acc: {train_acc / train_num:.4f}, "
                f"Test Acc: {val_acc / val_num:.4f}, Time: {end - start:.2f}s"
            )
            steps.append(step)
            val_accs.append(val_acc / val_num)
            if val_accs[-1] > 0.9:
                break
            start = time.time()
            total_loss = 0
        if step >= num_steps:
            break
        step += 1

    return model, steps, val_accs


def run_lcde_toy_experiment(runs, input_dims, hidden_dim, output_dim, batch_size, seed):
    """
    Runs the toy experiment with a Linear Controlled Differential Equation (CDE) model.

    For each specified input dimension, this function initialises a LinearCDE model,
    trains it using a toy dataset, and computes the mean squared error (MSE) on the
    validation set. Results for each configuration are saved.

    Args:
        runs (int): Number of times to repeat the experiment for each input dimension.
        input_dims (list[int]): List of input dimensions to test with.
        hidden_dim (int): Dimension of the hidden layer in the LinearCDE model.
        output_dim (int): Dimension of the model output.
        batch_size (int): Batch size for training.
        seed (int): Random seed for reproducibility.

    Saves:
        A .npy file containing the MSE for each run and input dimension in the
        "outputs_toy" directory. Files are named as
                                    `lin_cde_mse_{input_dim}_run_{run}.npy`.

    Prints:
        For each configuration, prints the input dimension and the computed MSE.
    """

    key = jr.PRNGKey(seed)

    if not os.path.isdir("results/outputs_toy"):
        os.mkdir("results/outputs_toy")

    for run in range(runs):
        model_key, train_key, key = jr.split(key, 3)
        for input_dim in input_dims:
            model = LinearCDE(hidden_dim, input_dim, output_dim, key=model_key)
            dataset = ToyDataloader(num=input_dim)
            mse = train_linear(
                model,
                dataset,
                num_train=dataset.num_train,
                num_val=dataset.num_val,
                output_dim=output_dim,
                batch_size=batch_size,
                key=train_key,
            )
            mse = np.array(mse)
            np.save(f"results/outputs_toy/lin_cde_mse_{input_dim}_run_{run}.npy", mse)
            print(f"Data dim: {input_dim}, MSE: {mse}")


def run_lcde_A5_experiment(
    input_dim,
    hidden_dim,
    output_dim,
    batch_size,
    num_steps,
    print_steps,
    learning_rate,
    lengths,
    train_split,
    seed,
):
    """
    Runs the A5 experiment with a Linear Controlled Differential Equation (CDE) model.

    This experiment trains an A5LinearCDE model with various input sequence lengths.
    For each length, a new model and dataset are initialised, and training is conducted
    with the specified parameters. The steps and validation accuracies are saved for
    analysis.

    Args:
        input_dim (int): Dimension of the input data.
        hidden_dim (int): Dimension of the hidden layer in the A5LinearCDE model.
        output_dim (int): Dimension of the model output.
        batch_size (int): Batch size for training.
        num_steps (int): Number of training steps.
        print_steps (int): Interval at which to log training progress.
        learning_rate (float): Learning rate for the optimiser.
        lengths (list[int]): Sequence lengths to evaluate in the experiment.
        train_split (float): Fraction of data used for training.
        seed (int): Random seed for reproducibility.

    Saves:
        In the "outputs_A5" directory, saves:
            - Steps at which training progress was logged
                (`linear_cde_length_{length}_steps.npy`)
            - Validation accuracies at each logged step
                (`linear_cde_length_{length}_val_accs.npy`)

    Prints:
        The current sequence length being processed, for progress tracking.
    """

    key = jr.PRNGKey(seed)

    if not os.path.isdir("results/outputs_A5"):
        os.mkdir("results/outputs_A5")

    model_key, train_key, length_2_data_key, data_key, key = jr.split(key, 5)

    for length in lengths:
        print(f"Length: {length}")
        model = A5LinearCDE(hidden_dim, input_dim, output_dim, key=model_key)
        dataset_length2 = A5Dataloader(length=2, train_split=1.0, key=length_2_data_key)
        dataset = A5Dataloader(length=length, train_split=train_split, key=data_key)
        _, steps, val_accs = train_model(
            model,
            dataset_length2,
            dataset,
            num_steps=num_steps,
            print_steps=print_steps,
            learning_rate=learning_rate,
            batch_size=batch_size,
            key=train_key,
        )
        np.save(f"results/outputs_A5/linear_cde_length_{length}_steps.npy", steps)
        np.save(f"results/outputs_A5/linear_cde_length_{length}_val_accs.npy", val_accs)
