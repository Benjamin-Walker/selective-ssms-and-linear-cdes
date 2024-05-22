# selective-ssms-and-linear-cdes
Experiments for Theoretical Foundations of Deep Selective State-Space Models

## Data Generation

The two toy datasets can be generated via `data/generate_toy_dataset.py`. The datasets consist of 2D and 3D paths respectively, both with $100$ 
regularly spaced samples between $t = 0$ and $t = 1$. The change in each channel at each time point is an independent sample 
from a standard Normal distribution rounded to the nearest integer. The labels are specific terms in the anti-symmetric parts of the signature,
given by the integrals
$$\int_0^1 \int_0^v \text{d}X^1_u \text{d}X^2_v$$
and 
$$\int_0^1 \int_0^w \int_0^v \text{d}X^1_u \text{d}X^2_v \text{d}X^3_w$$
for the 2D and 3D datasets respectively. The data and labels are saved in `data/data_n.npy` 
and `data/labels_n.npy` where `n` is the number of dimensions.

## Experiments

### Linear CDE

The linear CDE experiments can be run via `linear_cde.py`. For each path, a feature set is calculated as the solution 
$y_1\in\mathbb{R}^{256}$ to the linear CDE 
$$y_t = y_0 + \int_0^t A y_s \text{d}\omega_s + \int_0^t B \text{d}\xi_s$$
where $y_0=Cx_0+d$, $A\sim\mathcal{N}(0, 1/N)$, $B,C,d\sim\mathcal{N}(0, 1)$, and $\omega_s=\xi_s=(s,x_s)$. These features
are then used to train an ordinary least squares linear regression to predict the label.

### State-Space Models

The state-space model experiments can be run via `state_space_models.py`. The first models considered are a single S5 or
Mamba (S6) recurrence with a linear input layer and linear output layer applied to the final value from the recurrence. 
We then consider stacking two recurrences, with either a linear mixing layer or a linear mixing layer + ReLU inbetween 
the recurrent layers. Both state-space models use a hidden dimension of 256 and a state dimension of 256. They are trained
using the Adam optimizer with a learning rate of 3e-5 and a batch size of 32.


## Plotting


## Python Environments

The linear CDE and dataset generation are implemented using Jax, Diffrax, and Signax, as these libraries are currently
supported, unlike their PyTorch counterparts. The state-space models are implemented using PyTorch, as the selective SSM
layer is implemented in PyTorch.

It is possible to install cuda versions of Jax and Pytorch in the same environment using cuda 11.8. However,
we recommend using separate environments for Jax and PyTorch with cuda 12. 

### The Jax Environment

```angular2html
conda create -n jax_cde python=3.11
conda activate jax_cde
conda install pre-commit numpy scikit-learn matplotlib
pip install -U "jax[cuda12]"
pip install diffrax signax==0.1.1
pre-commit install
```


### The PyTorch Environment

```angular2html
conda create -n pytorch_mamba python=3.11
conda activate pytorch_mamba
conda install pytorch=2.2 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install packaging -c conda-forge
pip install causal-conv1d>=1.2.0 mamba-ssm s5-pytorch einops
```