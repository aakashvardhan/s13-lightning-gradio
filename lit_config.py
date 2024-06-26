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
        "num_epochs": 20,
        "num_workers": 2,
        "batch_size": 128,
        "precision": "16-mixed",
        "accelerator": "cuda",
        "progress_bar_refresh_rate": 10,
        "model_folder": "weights",
        "model_basename": "model_",
        "preload": True,
        "experiment_name": "runs/model",
        "end_lr": 10,
        "num_iter": 100,
        "ckpt_path": "",
        "classes": (
            "plane",
            "car",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        )
    }

    return config

