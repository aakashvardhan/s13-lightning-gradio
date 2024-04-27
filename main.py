import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, RichModelSummary,EarlyStopping

from lit_datamodule import CIFAR10DataModule
from lit_resnet import LitResNet
from s11_gradcam import get_config


def main(cfg):
    """
    Main function for training and evaluating the ResNet model.
    """

    # Set the seed for reproducibility
    L.seed_everything(1, workers=True)
    print("Seed set for reproducibility...")

    # Initialize the data module
    data_module = CIFAR10DataModule(cfg)
    data_module.prepare_data()
    data_module.setup()
    print("Data prepared and setup completed...")

    # Tensorboard logger
    tb_logger = TensorBoardLogger(save_dir="logs/", name="model")
    # Initialize the Lightning Trainer

    trainer = L.Trainer(precision="16-mixed",max_epochs=cfg["num_epochs"], logger=tb_logger,
                        accelerator="cuda",
                        devices="auto",
                        callbacks=[
                        ModelCheckpoint(dirpath=cfg['model_folder'], save_top_k=3, monitor="train_loss",mode="min",filename="model-{epoch:02d}-{train_loss:4f}",save_last=False),
                        LearningRateMonitor(logging_interval="step", log_momentum=True),
                        RichModelSummary(),EarlyStopping(monitor="train_loss", mode="min", stopping_threshold=1.5)],
                        gradient_clip_val=0.5,
                        num_sanity_val_steps=10,
                        log_every_n_steps=1,
                        check_val_every_n_epoch=2,
                        limit_val_batches=1000)
