<h1 align='center'> Theoretical Foundations of Deep Selective State-Space Models (NeurIPS 2024)<br>
    [<a href="https://arxiv.org/abs/2402.19047">arXiv</a>] </h1>

Our paper derives a theoretical framework for deep selective state-space models (SSMs) by recasting them as Linear CDEs. 
This reformulation allows us to fully characterise their expressive power and identify the gating mechanism as the crucial 
architectural choice. This repository contains the code to recreate the empirical results in the paper.

# Table of Contents
- [Introduction](#introduction)
  - [Controlled Differential Equations](#controlled-differential-equations)
  - [Linear CDEs](#linear-cdes)
  - [Selective State-Space Models are Linear CDEs](#selective-state-space-models-are-linear-cdes)
  - [This Repository](#this-repository)
- [Datasets](#datasets)
  - [The Toy Datasets](#the-toy-datasets)
  - [The A5 Benchmark](#the-a5-benchmark)
- [Models](#models)
  - [Linear CDE](#linear-cde)
  - [Sequence Models](#sequence-models)
- [Experiments](#experiments)
  - [Running Experiments](#running-experiments)
- [Results](#results)
  - [Visualising Results](#visualising-results)
- [Python Environments](#python-environments)
  - [The Jax Environment](#the-jax-environment)
  - [The PyTorch Environment](#the-pytorch-environment)
- [Citation](#citation)

# Introduction

## Controlled Differential Equations

A controlled differential equation (CDE) is a differential equation of the form,

$$
dy_t = F(y_t)\text{d}X_t,
$$

where $X_t : [0,T] \rightarrow \mathbb{R}^d$ is called the *control path*, $y_t :[0,T] \rightarrow \mathbb{R}^e$ is the *solution path*, and $F$ is a function from $R^e$ to $R^{e \times d}$.
Neural Controlled Differential Equations (NCDEs) are a special case of CDEs where $F$ is parameterised by a neural network. See [here](https://github.com/Benjamin-Walker/log-neural-cdes) for more details.

## Linear CDEs

The general form for a Linear CDE is a CDE where $F$ is a linear function of $y_t$, 

$$
\text{d}y_t = A y_t\text{d}X_t
$$

where $A$ is a tensor of shape $e\times d \times e$, or equivalently, 

$$
\text{d}y_t = \sum_{i=1}^{d} A_iy_t \text{d}X^i_t
$$

where $A_i$ are matrices of shape $e \times e$ and $X^i_t$ are the channels of the control path.


## Selective State-Space Models are Linear CDEs

As we show in the paper, it is possible to reformulate selective state-space models (SSMs) as Linear CDEs. The general form for a selective SSM is,

$$
\text{d}Z_t = \sum_{i=1}^{d_\omega} A_iZ_t \text{d}\omega^i_t + B\text{d}\xi_t,
$$

where:

- $\omega$ and $\xi$ are paths derived from the input data $X$,
- $A_i$ and $B$ are trainable parameters.

Importantly, the difference between SSM variants can be characterised by different choices of $\omega$ and $\xi$.

## This Repository

This repository contains the code to reproduce the experiments from our paper, which provide empirical evidence for many of our theoretical results.

# Datasets

## The Toy Datasets

The two toy datasets can be generated via `data_dir/generate_toy_dataset.py`. The datasets consist of 2D and 3D paths respectively, both with $100$ 
regularly spaced samples between $t = 0$ and $t = 1$. The change in each channel at each time point is an independent sample 
from a standard Normal distribution rounded to the nearest integer. The labels are specific terms in the anti-symmetric parts of the signature,
given by the integrals

$$ \int_0^1 \int_0^v \text{d}X^1_u \text{d}X^2_v $$

and 

$$ \int_0^1 \int_0^w \int_0^v \text{d}X^1_u \text{d}X^2_v \text{d}X^3_w $$

for the 2D and 3D datasets respectively. The data and labels are saved in `data_dir/toy_data/data_n.npy` 
and `data_dir/toy_data/labels_n.npy` where `n` is the number of dimensions.

## The A5 Benchmark

The A5 benchmark was introduced in ["The Illusion of State in State-Space Models"](https://arxiv.org/abs/2404.08819) by Merrill et al. The dataset tests models
on their state tracking ability. Each path in the dataset is a random sequence of elements from the group $A_5$, the even 
permutations of five elements. The label is the cumulative product of the elements in the path. The benchmark is split into a number of different
tasks by the length of the path, ranging from $3$ to $20$. The dataset can be downloaded in csv format from their [github repository](https://github.com/jopetty/word-problem). 

# Models

## Linear CDE

The Linear CDE processes sequential data by modeling the evolution of a hidden state over time, influenced by both the current state and control inputs derived from the data. It is defined by the equation:

$$ 
y_t = y_0 + \int_0^t A y_s \text{d}\omega_s + \int_0^t B \text{d}\xi_s 
$$

where:
- $y_t$ is the hidden state at time $t$,
- $y_0$  is the initial hidden state,
- $A$ and $B$ are trainable parameters,
- $\omega_s$ and $\xi_s$ are control paths derived from the input data.

We implement this model using Jax and take $\omega_s = \xi_s = [s, X_s]$, where $X_s$ is the input at time $s$.

The `linear_cde.py` file contains the following:

- **Embedding**: An embedding layer that maps discrete input indices to dense vectors.
- **LinearCDE Class**: Implements the Linear CDE model, providing options for adaptive or fixed-step ODE solvers.
- **A5LinearCDE Class**: A model tailored for the A5 dataset, incorporating the Linear CDE with additional layers such as normalization and dropout.
- **ODE Solvers**: Functions `adaptive_cde_solve` and `scan_cde_solve` for solving the CDE using adaptive and fixed-step methods, respectively.
- **Training Utilities**: Functions for training the models (`train_model`, `train_linear`) and extracting features (`obtain_features_from_model`).
- **Experiment Scripts**: Functions to run experiments on the toy dataset (`run_lcde_toy_experiment`) and the A5 dataset (`run_lcde_A5_experiment`).

## Sequence Models

We implement various sequence-to-sequence models using PyTorch, including a Recurrent Neural Network (RNN), Transformer, S4, and Mamba.

The `torch_sequence_models.py` file contains the following:

- **Embedding**: An `Embedding` layer that maps discrete input indices to dense vectors.

- **SequenceModel Class**: A flexible class (`SequenceModel`) that instantiates different model architectures based on the `model_name`. Key components of `SequenceModel` include:
  - **Embedding Layer**: Converts input tokens into dense embeddings.
  - **Positional Encoding**: Adds positional information for Transformers using a sinusoidal encoding.
  - **Sequence Layers**: 
    - **RNN**: Utilises recurrent connections to process sequences.
    - **TransformerLayer**: A Transformer with multi-head self-attention.
    - **S4Recurrence**: Structured State Space (S4) layer, using state-space models to capture long-term dependencies.
    - **MambaRecurrence**: An extension of S4 with a selectivity mechanism.
  - **Linear Mixing and Non-Linear Activation**: Applies a linear mixing layer with a non-linear activation. 
  - **Layer Normalisation**: Normalises each layer to improve training stability.
  - **Residual Connections**: Adds residual connections to each layer.
  - **Dropout**: Optional dropout layers prevent overfitting.
  - **Output Layer**: A final linear layer maps the processed sequences to the output dimension.

- **Training Utilities**: Functions to train the models, including:
  - **run_sm_toy_experiment**: Runs experiments on a toy dataset to compare different models, configurations, and depths.
  - **run_sm_A5_experiment**: Runs experiments on the $A_5$ dataset, assessing model performance across sequence lengths.

# Experiments

We provide a set of experiments to evaluate the performance of different sequence models on two datasets:

- **Toy Dataset**: A synthetic dataset designed to test the models' ability to capture the signature of a path.
- **$A_5$ Dataset**: A synthetic dataset designed to test the models' ability to state-track.

## Running Experiments

The experiments can be launched with the `run_experiment.py` script, which accepts command-line arguments to specify the model, dataset, and random seed. Configuration details, such as model depth, learning rate, and batch size, are set in YAML files within the `experiment_configs` directory.

```bash 
python run_experiment.py -m [MODEL] -e [EXPERIMENT] -s [SEED]
```
where `[MODEL]` specifies the model type (`LCDE` or `SequenceModel`), `[EXPERIMENT]` selects the dataset (`toy` or `A5`), and `[SEED]` sets a random seed (optional).

Configuration files for the paper's experiments are provided as lcde_toy.yaml, lcde_a5.yaml, ssm_toy.yaml, and ssm_a5.yaml, each containing recommended settings for their respective experiments. Adjust these files to customise hyperparameters and model configurations as needed.

# Results

After running the experiments, results are saved in the `results` directory. Each experiment produces numerical output files (e.g. RMSE values or accuracies over training steps) which can be used to evaluate model performance and analyse training dynamics.

## Visualising Results

We provide scripts to visualise the results and compare model performance across various settings:

- **RMSE Comparison Plot**: The `plot_toy.py` script compares RMSE performance on the toy dataset across different models, depths, and configurations. This script generates `rmse_subplots.pdf`, which is saved in the `results` directory. This is figure 1 in the paper.
- **Layer Requirements Plot**: The `plot_A5.py` script visualises the minimum number of blocks required by different models across sequence lengths on the $ A_5 $ benchmark. Results are saved as `A5_plot_shaded_regions.pdf` in the `results` directory. This is figure 2 in the paper.

These plots help summarise model efficiency and accuracy, highlighting differences in model capacity and suitability for various sequence lengths.

# Python Environments

The linear CDE and dataset generation are implemented using Jax, Diffrax, and Signax, as these libraries are currently
supported, unlike their PyTorch counterparts. The state-space models are implemented using PyTorch, as the selective SSM 
layer is implemented in PyTorch.

It is possible to install cuda versions of Jax and Pytorch in the same environment using cuda 11.8. However,
we recommend using separate environments for Jax and PyTorch with cuda 12. 

## The Jax Environment

```angular2html
conda create -n jax_cde python=3.11
conda activate jax_cde
conda install pre-commit numpy scikit-learn matplotlib pandas pyyaml
pip install -U "jax[cuda12]"
pip install diffrax optax signax==0.1.1
pre-commit install
```

## The PyTorch Environment

```angular2html
conda create -n pytorch_mamba python=3.11
conda activate pytorch_mamba
conda install pytorch=2.2 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install packaging pyyaml -c conda-forge
pip install causal-conv1d>=1.2.0 mamba-ssm s5-pytorch einops
```

# Citation

If you find this repository useful, please consider citing our paper:

```
@inproceedings{cirone2024deepSSM,
  title     = {Theoretical Foundations of Deep Selective State-Space Models},
  author    = {Nicola Muca Cirone and Antonio Orvieto and Benjamin Walker and Cristopher Salvi and Terry Lyons},
  booktitle = {Proceedings of the 38th Conference on Neural Information Processing Systems (NeurIPS)},
  year      = {2024},
  organization = {NeurIPS},
}
```
