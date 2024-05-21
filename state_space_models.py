# Copyright (c) 2023, Tri Dao, Albert Gu.

import math

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from s5 import S5


class MambaRecurrence(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        c_dependent=True,
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
        self.c_dependent = c_dependent
        if not self.c_dependent:
            self.c = nn.Parameter(
                torch.randn(
                    self.d_state,
                )
            )

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
        if self.c_dependent:
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        else:
            C = self.C[None, :, None].repeat(batch, 1, seqlen)
        D = torch.zeros(
            self.d_model,
        ).to(self.device)
        assert self.activation in ["silu", "swish"]
        y = selective_scan_fn(
            x,
            dt,
            A,
            B,
            C,
            D,
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
        C_dependent,
        depth,
        model_name,
        nonlinear,
    ):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, hidden_dim)
        self.depth = depth
        self.nonlinear = nonlinear
        if model_name == "mamba":
            self.ssm1 = MambaRecurrence(hidden_dim, state_dim, C_dependent)
            if self.depth == 2:
                self.ssm2 = MambaRecurrence(hidden_dim, state_dim, C_dependent)
        elif model_name == "s5":
            self.ssm1 = S5(hidden_dim, state_dim)
            if self.depth == 2:
                self.ssm2 = S5(hidden_dim, state_dim)
        else:
            raise ValueError("Invalid model name")
        self.linear_mix1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear_out = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.linear(x)
        x = self.ssm1(x)
        if self.depth == 2:
            if self.nonlinear:
                x = torch.relu(x)
            x = self.linear_mix1(x)
            x = self.ssm2(x)
        x = x[:, -1, :]
        x = self.linear_out(x)
        return x


class ToyDataset(torch.utils.data.Dataset):
    def __init__(self, train, num):
        super().__init__()
        numpy_data = np.load(f"data/data_{num}.npy")
        data = torch.tensor(numpy_data, dtype=torch.float32)
        N = data.shape[0]
        numpy_labels = np.load(f"data/labels_{num}.npy")
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


if __name__ == "__main__":

    batch = 32
    hidden_dim = 256
    state_dim = 256
    output_dim = 1
    lr = 1e-4
    for input_dim in [2, 3]:
        for model_name_C_dependence in [
            ("mamba", True),
            ("mamba", False),
            ("s5", False),
        ]:
            model_name, C_dependent = model_name_C_dependence
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
                    C_dependent,
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
                            all_rmse.append((total_mse / len(val_dataloader)) ** 0.5)
                            steps.append(step)
                            running_loss = 0.0
                            model.train()

                        step += 1

                steps = np.array(steps)
                all_rmse = np.array(all_rmse)
                np.save(
                    f"outputs/steps_100_{model_name}_{C_dependent}_{nonlinear}_{depth}_{lr}_{hidden_dim}_{input_dim}.npy",
                    steps,
                )
                np.save(
                    f"outputs/rmse_100_{model_name}_{C_dependent}_{nonlinear}_{depth}_{lr}_{hidden_dim}_{input_dim}.npy",
                    all_rmse,
                )
