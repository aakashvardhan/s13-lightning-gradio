import os
import torch


def get_config():
    """
    Returns the configuration dictionary for the model.

    Returns:
        config (dict): A dictionary containing the configuration parameters.
    """

    config = {
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "num_workers": 2,
        "batch_size": 128,
        "num_epochs": 20,
        "precision": 16,
        "progress_bar_refresh_rate": 10,
        "lr": 0.01,
    }

    return config


def get_lrfinder_config():
    """
    Returns the configuration dictionary for the LR Finder.

    Returns:
        config (dict): A dictionary containing the configuration parameters.
    """

    config = {
        "end_lr": 10,
        "num_iter": 100,
        "best_lr": 0.01,
    }

    return config
