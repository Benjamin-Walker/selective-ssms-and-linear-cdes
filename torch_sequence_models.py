"""
This module provides implementations and utilities for running experiments with
Sequence Models (SMs) using PyTorch.

It includes classes and functions for:

- Implementing various SM layers such as S4, Mamba, RNN, and Transformer.
- Defining Stacked SMs that use these SM layers.
- Training models on the toy dataset and the A5 dataset.
- Dataset classes for loading and preprocessing data.
- Utility functions for running experiments.

References:
- S4 and Mamba recurrence adapted from https://github.com/state-spaces/mamba
"""

import math
import os
from itertools import cycle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from einops import rearrange, repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn


class Embedding(nn.Module):
    """
    Embedding layer for mapping discrete indices to dense vectors.

    Args:
        input_dim (int): Number of unique embeddings (vocabulary size).
        output_dim (int): Dimensionality of each embedding vector.

    Forward Args:
        x (Tensor): Input tensor of shape (batch_size, sequence_length).

    Returns:
        Tensor: Embedding vectors corresponding to the input indices.
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, output_dim)

    def forward(self, x):
        return self.embedding(x)


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding.

    This module adds positional information to the input embeddings to allow the model
    to make use of the order of the sequence.

    Args:
        d_model (int): The embedding dimension.
        dropout (float): The dropout probability. Defaults to 0.1.
        max_len (int): The maximum length of the sequences. Defaults to 5000.

    Forward Args:
        x (Tensor): Input tensor of shape (batch_size, sequence_length, embedding_dim).

    Returns:
            Tensor: Output tensor with positional encoding added, of the same shape as
                input.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(1)].transpose(0, 1)
        return self.dropout(x)


class TransformerLayer(nn.Module):
    """
    A Transformer encoder layer with a causal mask for sequence-to-sequence modelling.

    Args:
        hidden_dim (int): The dimension of the hidden state.
        n_heads (int): The number of attention heads.

    Call Args:
        x (torch.Tensor): Input tensor of shape (batch_size, sequence_length,
                            hidden_dim).

    Returns:
        torch.Tensor: Output tensor of the same shape as input, with transformed
                        features.
    """

    def __init__(self, hidden_dim, n_heads):
        super().__init__()
        self.transformer_layer = nn.TransformerEncoderLayer(
            hidden_dim, n_heads, batch_first=True
        )

    def __call__(self, x):
        mask = torch.nn.Transformer.generate_square_subsequent_mask(
            x.size(1), device=x.device
        )
        x = self.transformer_layer(x, src_mask=mask, is_causal=True)
        return x


class S4Recurrence(nn.Module):
    """
    Implements the S4 recurrence layer from the Structured State Spaces for Sequence
    Modeling.

    Args:
        d_model (int): Dimension of the model (number of features).
        d_state (int): Dimension of the state space. Defaults to 16.
        dt_rank (int or str): Rank for the time-step parameterization. Defaults to
            "auto".
        dt_min (float): Minimum value for time-steps. Defaults to 0.001.
        dt_max (float): Maximum value for time-steps. Defaults to 0.1.
        dt_init_floor (float): Floor value for initializing time-steps. Defaults to
            1e-4.
        device (torch.device, optional): Device to run the computations on. If None,
            uses CUDA if available.

    Forward Args:
        hidden_states (Tensor): Input tensor of shape (batch_size, sequence_length,
            d_model).

    Returns:
        Tensor: Output tensor of the same shape as input.
    """

    def __init__(
        self,
        d_model,
        d_state=16,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        device=None,
    ):
        super().__init__()

        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device

        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        dt = torch.exp(
            torch.rand(self.d_model, device=self.device)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        dt_log = torch.log(dt)
        self.log_dt = nn.Parameter(dt_log)

        # S4D-Lin initialization
        A_real = (
            0.5
            * torch.ones(self.d_model, self.d_state, device=self.device).contiguous()
        )
        A_imag = repeat(
            torch.pi * torch.arange(1, self.d_state + 1, device=self.device),
            "n -> d n",
            d=self.d_model,
        ).contiguous()
        A_real_log = torch.log(A_real)
        self.A_real_log = nn.Parameter(A_real_log)
        self.A_imag = nn.Parameter(A_imag)
        B = nn.init.xavier_normal_(
            torch.empty(
                self.d_model, self.d_state, device=self.device, dtype=torch.complex64
            )
        )
        C = nn.init.xavier_normal_(
            torch.empty(
                self.d_model, self.d_state, device=self.device, dtype=torch.complex64
            )
        )
        self.B = nn.Parameter(B)
        self.C = nn.Parameter(C)
        self.D = nn.Parameter(torch.ones(self.d_model, device=self.device))

    def forward(self, hidden_states):
        A_real = -torch.exp(self.A_real_log.float())
        A_imag = self.A_imag.float()
        A = A_real + 1j * A_imag
        x = rearrange(hidden_states, "b l d -> b d l")
        dt = self.log_dt.exp()[None, :, None].repeat(x.shape[0], 1, x.shape[2])
        y = selective_scan_fn(
            x,
            dt,
            A,
            self.B,
            self.C,
            self.D,
            z=None,
            delta_softplus=True,
            return_last_state=False,
        )
        return rearrange(y, "b d l -> b l d")


class MambaRecurrence(nn.Module):
    """
    Implements the Mamba recurrence layer for sequence modeling.

    Args:
        d_model (int): Dimension of the model (number of features).
        d_state (int): Dimension of the state space. Defaults to 16.
        dt_rank (int or str): Rank for the time-step parameterization. Defaults to
            "auto".
        dt_min (float): Minimum value for time-steps. Defaults to 0.001.
        dt_max (float): Maximum value for time-steps. Defaults to 0.1.
        dt_init (str): Initialization method for dt. Options are "constant" or "random".
            Defaults to "random".
        dt_scale (float): Scale factor for dt initialization. Defaults to 1.0.
        dt_init_floor (float): Floor value for initializing time-steps. Defaults to
            1e-4.
        device (torch.device, optional): Device to run the computations on. If None,
            uses CUDA if available.

    Forward Args:
        hidden_states (Tensor): Input tensor of shape (batch_size, sequence_length,
            d_model).

    Returns:
        Tensor: Output tensor of the same shape as input.
    """

    def __init__(
        self,
        d_model,
        d_state=16,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        device=None,
    ):
        super().__init__()

        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device
        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.x_proj = nn.Linear(
            self.d_model,
            self.dt_rank + self.d_state * 2,
            bias=False,
            device=self.device,
        )
        self.dt_proj = nn.Linear(
            self.dt_rank, self.d_model, bias=True, device=self.device
        )

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_model, device=self.device)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this
        # one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=self.device),
            "n -> d n",
            d=self.d_model,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)

        # D "skip" parameter
        self.D = nn.Parameter(
            torch.ones(self.d_model, device=self.device)
        )  # Keep in fp32
        self.D._no_weight_decay = True

    def forward(self, hidden_states):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        x = rearrange(hidden_states, "b l d -> b d l")
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
        dt, B, C = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = self.dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        y = selective_scan_fn(
            x,
            dt,
            A,
            B,
            C,
            self.D.float(),
            z=None,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
            return_last_state=False,
        )
        return rearrange(y, "b d l -> b l d")


class RNN(nn.Module):
    """
    A simple RNN layer wrapper.

    Args:
        input_size (int): The number of expected features in the input.
        hidden_size (int): The number of features in the hidden state.

    Forward Args:
        x (Tensor): Input tensor of shape (batch_size, sequence_length, input_size).

    Returns:
        Tensor: Output tensor from the RNN, of shape (batch_size, sequence_length,
            hidden_size).
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)

    def forward(self, x):
        return self.rnn(x)[0]


class SequenceModel(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim,
        state_dim,
        depth,
        model_name,
        nonlinear,
        activation="GLU",
        continuous_output=False,
        embedding=False,
        use_layernorm=False,
        dropout=0.0,
    ):
        """
        A general sequence model that can use different recurrent layers like S4, Mamba,
         RNN, or Transformer.

        Args:
            input_dim (int): Dimension of the input features.
            output_dim (int): Dimension of the output features.
            hidden_dim (int): Dimension of the hidden state.
            state_dim (int): Dimension of the state space (for SSM layers).
            depth (int): Number of recurrent layers.
            model_name (str): Name of the model to use ('S4', 'Mamba', 'RNN',
                'Transformer', or 'S5').
            nonlinear (bool): Whether to apply a non-linear activation function.
            continuous_output (bool): If True, outputs the full sequence; otherwise,
                outputs the mean over the sequence.
            embedding (bool): If True, applies an embedding layer to the input.
            use_layernorm (bool): If True, applies LayerNorm after each layer.
            dropout (float): Dropout probability. Defaults to 0.0.

        Forward Args:
            x (Tensor): Input tensor of shape (batch_size, sequence_length, input_dim).

        Returns:
            Tensor: Output tensor of shape (batch_size, output_dim) or (batch_size,
                sequence_length, output_dim) depending on `continuous_output`.
        """
        super().__init__()
        if embedding:
            self.init_layer = Embedding(input_dim, hidden_dim)
        else:
            self.init_layer = torch.nn.Linear(input_dim, hidden_dim)
        self.nonlinear = nonlinear
        if activation == "GLU":
            self.activation = nn.GLU()
        elif activation == "GELU":
            self.activation = nn.GELU()
        else:
            raise ValueError("Invalid activation function")
        self.act_name = activation

        self.ssms = nn.ModuleList()
        self.linear_mixes = nn.ModuleList()

        self.pos_encode = None

        if model_name == "Mamba":
            sequence_layer = MambaRecurrence
        elif model_name == "S4":
            sequence_layer = S4Recurrence
        elif model_name == "RNN":
            sequence_layer = lambda x, y: RNN(input_size=x, hidden_size=x)
        elif model_name == "Transformer":
            sequence_layer = TransformerLayer
            self.pos_encode = SinusoidalPositionalEncoding(hidden_dim)
        else:
            raise ValueError("Invalid model name")

        for _ in range(depth):
            self.ssms.append(sequence_layer(hidden_dim, state_dim))
            if self.nonlinear and activation == "GLU":
                self.linear_mixes.append(nn.Linear(hidden_dim, 2 * hidden_dim))
            else:
                self.linear_mixes.append(nn.Linear(hidden_dim, hidden_dim))

        self.use_layernorm = use_layernorm
        if use_layernorm:
            self.layernorms = nn.ModuleList(
                [nn.LayerNorm(hidden_dim) for _ in range(depth)]
            )

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

        self.linear_out = torch.nn.Linear(hidden_dim, output_dim)
        self.continuous_output = continuous_output

    def forward(self, x):
        x = self.init_layer(x)
        if self.pos_encode is not None:
            x = self.pos_encode(x)

        residual = x

        for i, (ssm, linear_mix) in enumerate(zip(self.ssms, self.linear_mixes)):
            x = ssm(x)
            if self.dropout is not None:
                x = self.dropout(x)
            x = linear_mix(x)
            if self.nonlinear:
                x = self.activation(x)
            if self.dropout is not None:
                x = self.dropout(x)
            x = x + residual
            if self.use_layernorm:
                x = self.layernorms[i](x)
            residual = x

        if not self.continuous_output:
            x = x.mean(dim=1)
        x = self.linear_out(x)
        return x


class ToyDataset(torch.utils.data.Dataset):
    """
    Dataset class for the toy dataset.

    Args:
        train (bool): If True, loads the training split; otherwise, loads the validation
            split.
        num (int): Dimensionality of the toy dataset (2 or 3).

    Attributes:
        data (Tensor): The input data tensors.
        labels (Tensor): The target labels.
        length (int): Number of samples in the dataset.
    """

    def __init__(self, train, num):
        super().__init__()
        numpy_data = np.load(f"data_dir/toy_data/data_{num}.npy")
        data = torch.tensor(numpy_data, dtype=torch.float32)
        N = data.shape[0]
        numpy_labels = np.load(f"data_dir/toy_data/labels_{num}.npy")
        labels = torch.tensor(numpy_labels, dtype=torch.float32)
        if train:
            self.data = data[: int(0.8 * N)]
            self.labels = labels[: int(0.8 * N), None]
        else:
            self.data = data[int(0.8 * N) :]
            self.labels = labels[int(0.8 * N) :, None]

        self.length = len(self.data)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class A5Dataset(torch.utils.data.Dataset):
    """
    Dataset class for the A5 dataset.

    Args:
        length (int): Length of the sequences in the dataset.
        train (bool): If True, loads the training split; otherwise, loads the validation
            split.
        train_split (float): Proportion of data to use for training. Defaults to 0.8.
        seed (int, optional): Seed for random shuffling of the data.

    Attributes:
        data (Tensor): The input data tensors.
        labels (Tensor): The target labels.
        length (int): Number of samples in the dataset.
    """

    def __init__(self, length, train=True, train_split=0.8, seed=None):
        super().__init__()
        df = pd.read_csv(f"data_dir/illusion/A5={length}.csv")
        input_array = df["input"].str.split(" ", expand=True).astype(int).to_numpy()
        target_array = df["target"].str.split(" ", expand=True).astype(int).to_numpy()

        data = torch.tensor(input_array, dtype=torch.long)
        labels = torch.tensor(target_array, dtype=torch.long)

        N = data.shape[0]
        if seed is not None:
            torch.manual_seed(seed)
        shuffle_idx = torch.randperm(N)
        data = data[shuffle_idx]
        labels = labels[shuffle_idx]

        split_idx = int(train_split * N)
        if train:
            self.data = data[:split_idx]
            self.labels = labels[:split_idx]
        else:
            self.data = data[split_idx:]
            self.labels = labels[split_idx:]

        self.length = len(self.data)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def run_sm_toy_experiment(
    runs,
    model_names,
    input_dims,
    depth_nonlinears,
    output_dim,
    hidden_dim,
    state_dim,
    batch_size,
    learning_rate,
    num_epochs=400,
    seed=1234,
):
    """
    Runs experiments on the toy dataset using different sequence models.

    Args:
        runs (int): Number of times to repeat the experiment.
        model_names (list[str]): List of model names to test ('S4', 'Mamba', 'RNN').
        input_dims (list[int]): List of input dimensions to test with.
        depth_nonlinears (list[tuple]): List of tuples containing depth and nonlinear
            flag.
        output_dim (int): Dimension of the model output.
        hidden_dim (int): Dimension of the hidden state.
        state_dim (int): Dimension of the state space (for SSM layers).
        batch_size (int): Batch size for training.
        learning_rate (float): Learning rate for the optimizer.
        num_epochs (int): Number of epochs to train. Defaults to 400.
        seed (int): Random seed for reproducibility.

    Saves:
        - Steps and RMSE values are saved in the "results/outputs_toy" directory with
            filenames containing model parameters.

    Prints:
        - Training progress and validation RMSE at intervals.
    """
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.isdir("results/outputs_toy"):
        os.mkdir("results/outputs_toy")

    for model_name in model_names:
        for run in range(runs):
            for input_dim in input_dims:
                for depth_nonlinear in depth_nonlinears:
                    depth, nonlinear = depth_nonlinear

                    print(
                        "Running",
                        model_name,
                        "on toy dataset with input dimension",
                        input_dim,
                        "depth",
                        depth,
                        "nonlinear",
                        nonlinear,
                    )

                    train_dataset = ToyDataset(train=True, num=input_dim)
                    val_dataset = ToyDataset(train=False, num=input_dim)

                    dataloader = torch.utils.data.DataLoader(
                        train_dataset, batch_size=batch_size, shuffle=True
                    )
                    val_dataloader = torch.utils.data.DataLoader(
                        val_dataset, batch_size=batch_size, shuffle=True
                    )

                    model = SequenceModel(
                        input_dim,
                        output_dim,
                        hidden_dim,
                        state_dim,
                        depth,
                        model_name,
                        nonlinear,
                        activation="GELU",
                    ).to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

                    running_loss = 0.0
                    all_rmse = []
                    steps = []
                    step = 0

                    # Training loop
                    for _ in range(num_epochs):
                        for X, y in dataloader:
                            optimizer.zero_grad()

                            X = X.to(device)
                            y = y.to(device)
                            y_hat = model(X)
                            loss = torch.nn.functional.mse_loss(y_hat, y)
                            running_loss += loss.item()
                            loss.backward()
                            optimizer.step()

                            if step % 10000 == 0:
                                total_mse = 0.0
                                model.eval()
                                with torch.no_grad():
                                    for X, y in val_dataloader:
                                        X = X.to(device)
                                        y = y.to(device)
                                        y_hat = model(X)
                                        mse = torch.nn.functional.mse_loss(y_hat, y)
                                        total_mse += mse.item()

                                running_loss = (
                                    running_loss * 10000 if step == 0 else running_loss
                                )
                                print(
                                    f"Step: {step}, Loss: {running_loss / 10000}, "
                                    f"RMSE: {(total_mse / len(val_dataloader)) ** 0.5}"
                                )
                                all_rmse.append(
                                    (total_mse / len(val_dataloader)) ** 0.5
                                )
                                steps.append(step)
                                running_loss = 0.0
                                model.train()

                            step += 1

                    steps = np.array(steps)
                    all_rmse = np.array(all_rmse)

                    np.save(
                        f"results/outputs_toy/steps_100_{model_name}_{nonlinear}"
                        f"_{depth}_{learning_rate}_{hidden_dim}_{input_dim}"
                        f"_run_{run}.npy",
                        steps,
                    )
                    np.save(
                        f"results/outputs_toy/rmse_100_{model_name}_{nonlinear}"
                        f"_{depth}_{learning_rate}_{hidden_dim}_{input_dim}"
                        f"_run_{run}.npy",
                        all_rmse,
                    )


def run_sm_A5_experiment(
    model_names,
    lengths,
    depths,
    input_dim,
    output_dim,
    hidden_dim,
    state_dim,
    batch_size,
    learning_rate,
    num_epochs=400,
    seed=1234,
):
    """
    Runs experiments on the A5 dataset using different sequence models.

    Args:
        model_names (list[str]): List of model names to test ('S4', 'Mamba', 'RNN').
        lengths (list[int]): List of sequence lengths to evaluate.
        depths (list[int]): List of depths for the models.
        input_dim (int): Dimension of the input data.
        output_dim (int): Dimension of the model output.
        hidden_dim (int): Dimension of the hidden state.
        state_dim (int): Dimension of the state space (for SSM layers).
        batch_size (int): Batch size for training.
        learning_rate (float): Learning rate for the optimizer.
        num_epochs (int): Number of epochs to train. Defaults to 400.
        seed (int): Random seed for reproducibility.

    Saves:
        - Steps and accuracies are saved in the "results/outputs_A5" directory with
            filenames containing model parameters.

    Prints:
        - Training progress and validation accuracy at intervals.
    """
    torch.manual_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.isdir("results/outputs_A5"):
        os.mkdir("results/outputs_A5")

    for model_name in model_names:
        depth_recurrence = depths.copy()
        for length in lengths:
            depths_to_remove = []
            for depth in depth_recurrence:
                print(
                    f"Running {model_name} on A5 dataset with length {length} and "
                    f"depth {depth}"
                )
                model = SequenceModel(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    hidden_dim=hidden_dim,
                    state_dim=state_dim,
                    depth=depth,
                    model_name=model_name,
                    nonlinear=True,
                    embedding=True,
                    use_layernorm=True,
                    dropout=0.1,
                    continuous_output=True,
                ).to(device)

                train_dataset = A5Dataset(length, train=True, seed=seed + 1)
                train_dataset_length_2 = A5Dataset(2, train=True, seed=seed + 2)
                val_dataset = A5Dataset(length, train=False, seed=seed + 1)

                train_dataloader = torch.utils.data.DataLoader(
                    train_dataset, batch_size=batch_size, shuffle=True
                )
                train_dataloader_length_2 = torch.utils.data.DataLoader(
                    train_dataset_length_2, batch_size=batch_size, shuffle=True
                )
                val_dataloader = torch.utils.data.DataLoader(
                    val_dataset, batch_size=batch_size, shuffle=True
                )

                optimizer = torch.optim.AdamW(
                    model.parameters(), lr=learning_rate, weight_decay=0.01
                )

                running_loss = 0.0
                all_acc = []
                steps = []
                step = 0

                early_stop = False

                cycled_train_dataloader_length_2 = cycle(train_dataloader_length_2)

                for _ in range(num_epochs):
                    for (X, y), (X_2, y_2) in zip(
                        train_dataloader, cycled_train_dataloader_length_2
                    ):
                        optimizer.zero_grad()

                        X = X.to(device)
                        y = y.to(device)
                        y_hat = model(X)
                        loss = torch.nn.functional.cross_entropy(
                            y_hat.reshape(-1, output_dim),
                            y.reshape(
                                -1,
                            ),
                        )
                        running_loss += loss.item()
                        loss.backward()
                        X_2 = X_2.to(device)
                        y_2 = y_2.to(device)
                        y_hat_2 = model(X_2)
                        loss_2 = torch.nn.functional.cross_entropy(
                            y_hat_2.reshape(-1, output_dim),
                            y_2.reshape(
                                -1,
                            ),
                        )
                        loss_2.backward()
                        optimizer.step()

                        if step % 10000 == 0:
                            model.eval()
                            total_acc = 0.0
                            total_num = 0
                            with torch.no_grad():
                                for X, y in val_dataloader:
                                    X = X.to(device)
                                    y = y.to(device)
                                    y_hat = model(X)
                                    acc = (y_hat.argmax(dim=-1) == y).sum().item()
                                    total_acc += acc
                                    total_num += y.shape[0] * y.shape[1]

                            running_loss = (
                                running_loss * 10000 if step == 0 else running_loss
                            )
                            print(
                                f"Step: {step}, Loss: {running_loss / 10000}, "
                                f"Acc: {total_acc / total_num}"
                            )
                            all_acc.append(total_acc / total_num)
                            steps.append(step)
                            if all_acc[-1] > 0.90:
                                early_stop = True
                                break
                            running_loss = 0.0
                            model.train()

                        step += 1

                    if early_stop:
                        break

                steps = np.array(steps)
                all_acc = np.array(all_acc)
                np.save(
                    f"results/outputs_A5/{model_name}_{length}_{depth}_{learning_rate}_{hidden_dim}_steps_run_{seed}.npy",
                    steps,
                )
                np.save(
                    f"results/outputs_A5/{model_name}_{length}_{depth}_{learning_rate}_{hidden_dim}_acc_run_{seed}.npy",
                    all_acc,
                )

                if early_stop:
                    break
                else:
                    depths_to_remove.append(depth)

            for depth in depths_to_remove:
                depth_recurrence.remove(depth)
