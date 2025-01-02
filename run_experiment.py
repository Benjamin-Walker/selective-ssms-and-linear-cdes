"""
This script facilitates the running of model experiments by loading configurations,
parsing command-line arguments, and executing the appropriate experiment function based
on the user’s input.

The available options for models and experiments are:
- **Models**: `LCDE` (Linear CDE), `SequenceModel` (supports RNN, Transformer, S4,
    Mamba).
- **Experiments**: `toy` (toy dataset), `A5` ($A_5$ dataset).

Usage:
    Run the script from the command line with arguments for model type, experiment type,
    and an optional random seed:
    ```
    python run_experiment.py -m <model_type> -e <experiment_type> -s <seed>
    ```

Functions:
    - `load_config`: Loads YAML configuration files from the `experiment_configs/`
        directory.
    - `parse_arguments`: Parses command-line arguments specifying model, experiment,
        and seed.
    - `main`: Executes the appropriate experiment function with the specified
        configuration and seed.

Experiment Configuration Files:
    - **Linear CDE Configs**: `lcde_toy.yaml`, `lcde_a5.yaml`
    - **Sequence Model Configs**: `ssm_toy.yaml`, `ssm_a5.yaml`
"""

import argparse

import yaml


def load_config(config_file):
    """Loads the configuration file."""
    with open("experiment_configs/" + config_file, "r") as file:
        config = yaml.safe_load(file)
    return config


def str2bool(value):
    """Convert string to boolean."""
    if isinstance(value, bool):
        return value
    if value.lower() in ("true", "yes", "1"):
        return True
    elif value.lower() in ("false", "no", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run model experiments with different configurations."
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        choices=["LCDE", "SequenceModel"],
        help="Model type. Choose between 'LCDE' and 'SequenceModel'.",
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        choices=["Transformer", "Mamba", "RNN", "S4"],
    )
    parser.add_argument(
        "-l",
        "--length",
        type=int,
        choices=[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20],
    )
    parser.add_argument(
        "-d",
        "--depth",
        type=int,
        choices=[1, 2, 3, 4],
    )
    parser.add_argument(
        "-i",
        "--input_dim",
        type=int,
        choices=[2, 3],
    )
    parser.add_argument(
        "-nl",
        "--nonlinear",
        type=str2bool,
    )
    parser.add_argument(
        "-e",
        "--experiment",
        type=str,
        required=True,
        choices=["toy", "A5"],
        help="Experiment type. Choose between 'toy' and 'A5'.",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=2345,
        help="Random seed for the experiment. Default is 2345.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    model = args.model
    experiment = args.experiment
    seed = args.seed

    if model == "LCDE":
        from linear_cde import run_lcde_A5_experiment, run_lcde_toy_experiment

        if experiment == "toy":
            config_file = "lcde_toy.yaml"
            run_fn = run_lcde_toy_experiment
        elif experiment == "A5":
            config_file = "lcde_a5.yaml"
            run_fn = run_lcde_A5_experiment
        else:
            raise ValueError("Invalid experiment type!")
    elif model == "SequenceModel":
        from torch_sequence_models import run_sm_A5_experiment, run_sm_toy_experiment

        if experiment == "toy":
            config_file = "ssm_toy.yaml"
            run_fn = run_sm_toy_experiment
        elif experiment == "A5":
            config_file = "ssm_a5.yaml"
            run_fn = run_sm_A5_experiment
        else:
            raise ValueError("Invalid experiment type!")

    config = load_config(config_file)
    config = load_config(config_file)
    if args.name:
        config["model_names"] = [args.name]
    if args.depth:
        config["depth_nonlinears"] = [[args.depth, args.nonlinear]]
    if args.length:
        config["lengths"] = [args.length]
    if args.input_dim:
        config["input_dims"] = [args.input_dim]
    run_fn(**config, seed=seed)
