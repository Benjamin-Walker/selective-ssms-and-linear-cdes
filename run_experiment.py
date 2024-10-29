import argparse

import yaml


def load_config(config_file):
    """Loads the configuration file."""
    with open("experiment_configs/" + config_file, "r") as file:
        config = yaml.safe_load(file)
    return config


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
        from state_space_models import run_sm_A5_experiment, run_sm_toy_experiment

        if experiment == "toy":
            config_file = "ssm_toy.yaml"
            run_fn = run_sm_toy_experiment
        elif experiment == "A5":
            config_file = "ssm_a5.yaml"
            run_fn = run_sm_A5_experiment
        else:
            raise ValueError("Invalid experiment type!")

    config = load_config(config_file)
    run_fn(**config, seed=seed)
