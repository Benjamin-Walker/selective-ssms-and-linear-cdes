# S4 and Mamba recurrence adapted from https://github.com/state-spaces/mamba

import math
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from einops import rearrange, repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from s5 import S5


class S4Recurrence(nn.Module):
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

        self.activation = "silu"
        self.act = nn.SiLU()

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
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        A_real = -torch.exp(self.A_real_log.float())
        A_imag = self.A_imag.float()
        A = A_real + 1j * A_imag
        x = rearrange(hidden_states, "b l d -> b d l")
        dt = self.log_dt.exp()[None, :, None].repeat(x.shape[0], 1, x.shape[2])
        assert self.activation in ["silu", "swish"]
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
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_model, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(
            self.dt_rank, self.d_model, bias=True, **factory_kwargs
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
            torch.rand(self.d_model, **factory_kwargs)
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
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_model,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_model, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device

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
        assert self.activation in ["silu", "swish"]
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


class SSM(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim,
        state_dim,
        depth,
        model_name,
        nonlinear,
    ):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, hidden_dim)
        self.depth = depth
        self.nonlinear = nonlinear
        self.activation = nn.GELU()
        if model_name == "Mamba":
            self.ssm1 = MambaRecurrence(hidden_dim, state_dim)
            if self.depth == 2:
                self.ssm2 = MambaRecurrence(hidden_dim, state_dim)
        elif model_name == "S5":
            self.ssm1 = S5(hidden_dim, state_dim)
            if self.depth == 2:
                self.ssm2 = S5(hidden_dim, state_dim)
        elif model_name == "S4":
            self.ssm1 = S4Recurrence(hidden_dim, state_dim)
            if self.depth == 2:
                self.ssm2 = S4Recurrence(hidden_dim, state_dim)
        else:
            raise ValueError("Invalid model name")
        self.linear_mix1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear_mix2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear_out = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.linear(x)
        y = x
        x = self.ssm1(x)
        if self.nonlinear:
            x = self.activation(x)
        x = self.linear_mix1(x)
        x = x + y
        if self.depth == 2:
            y = x
            x = self.ssm2(x)
            if self.nonlinear:
                x = self.activation(x)
            x = self.linear_mix2(x)
            x = x + y

        x = x.mean(dim=1)
        x = self.linear_out(x)
        return x


class ToyDataset(torch.utils.data.Dataset):
    def __init__(self, train, num):
        super().__init__()
        numpy_data = np.load(f"data_dir/data_{num}.npy")
        data = torch.tensor(numpy_data, dtype=torch.float32)
        N = data.shape[0]
        numpy_labels = np.load(f"data_dir/labels_{num}.npy")
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
    def __init__(self, length, train=True, train_split=0.8, seed=None):
        super().__init__()
        # Load the data
        df = pd.read_csv(f"data_dir/illusion/A5={length}.csv")
        input_array = df["input"].str.split(" ", expand=True).astype(int).to_numpy()
        target_array = df["target"].str.split(" ", expand=True).astype(int).to_numpy()

        # Convert to tensors
        data = torch.tensor(input_array, dtype=torch.float32)
        labels = torch.tensor(target_array, dtype=torch.float32)

        # Shuffle the data
        N = data.shape[0]
        if seed is not None:
            torch.manual_seed(seed)
        shuffle_idx = torch.randperm(N)
        data = data[shuffle_idx]
        labels = labels[shuffle_idx]

        # Split the data into training and test sets
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


if __name__ == "__main__":

    batch = 32
    hidden_dim = 256
    state_dim = 256
    output_dim = 1
    lr = 1e-4
    for run in range(5):
        for input_dim in [2, 3]:
            for model_name in ["S4", "S5", "Mamba"]:
                for depth_nonlinear in [(2, True), (2, False), (1, False)]:
                    depth, nonlinear = depth_nonlinear
                    train_dataset = ToyDataset(train=True, num=input_dim)
                    val_dataset = ToyDataset(train=False, num=input_dim)
                    dataloader = torch.utils.data.DataLoader(
                        train_dataset, batch_size=batch, shuffle=True
                    )
                    val_dataloader = torch.utils.data.DataLoader(
                        val_dataset, batch_size=batch, shuffle=True
                    )
                    model = SSM(
                        input_dim,
                        output_dim,
                        hidden_dim,
                        state_dim,
                        depth,
                        model_name,
                        nonlinear,
                    ).to("cuda")
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                    running_loss = 0.0
                    all_rmse = []
                    steps = []
                    step = 0
                    for _ in range(400):
                        for X, y in dataloader:
                            optimizer.zero_grad()

                            X = X.to("cuda")
                            y = y.to("cuda")
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
                                        X = X.to("cuda")
                                        y = y.to("cuda")
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
                    if not os.path.isdir("outputs"):
                        os.mkdir("outputs")
                    np.save(
                        f"outputs/steps_100_{model_name}_{nonlinear}_{depth}_{lr}_{hidden_dim}_{input_dim}_run_{run}.npy",
                        steps,
                    )
                    np.save(
                        f"outputs/rmse_100_{model_name}_{nonlinear}_{depth}_{lr}_{hidden_dim}_{input_dim}_run_{run}.npy",
                        all_rmse,
                    )
