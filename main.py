import os
from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    RichModelSummary,
    EarlyStopping,
    TQDMProgressBar
)

from lit_datamodule import CIFAR10DataModule
from lit_resnet import LitResNet
from config import get_config


def main(cfg, arg):
    """
    Main function for training and evaluating the ResNet model.
    """

    # Set the seed for reproducibility
    L.seed_everything(42, workers=True)
    print("Seed set for reproducibility...")

    # Initialize the data module
    data_module = CIFAR10DataModule(cfg)
    data_module.prepare_data()
    data_module.setup()
    print("Data prepared and setup completed...")

    # Tensorboard logger
    tb_logger = TensorBoardLogger(save_dir=os.getcwd(), version=1, name="lightning_logs")
    # Initialize the Lightning Trainer

    trainer = L.Trainer(
        precision=cfg["precision"],
        max_epochs=cfg["num_epochs"],
        logger=tb_logger,
        accelerator=cfg["accelerator"],
        devices=arg.devices,
        callbacks=[
            ModelCheckpoint(
                dirpath=cfg["model_folder"],
                save_top_k=3,
                monitor="train_loss",
                mode="min",
                filename="model-{epoch:02d}-{train_loss:4f}",
                save_last=True,
            ),
            LearningRateMonitor(logging_interval="step", log_momentum=True),
            EarlyStopping(monitor="train_loss", mode="min", stopping_threshold=0.15),
            TQDMProgressBar(refresh_rate=10)
        ],
        gradient_clip_val=0.5,
        deterministic=True,
        num_sanity_val_steps=5,
        overfit_batches=1000,
        sync_batchnorm=True,
        enable_progress_bar=True,
        log_every_n_steps=5,
        check_val_every_n_epoch=2,
        limit_val_batches=1000,
    )

    tuner = L.pytorch.tuner.Tuner(trainer)

    model = LitResNet(cfg)

    lr_finder = tuner.lr_find(
        model, datamodule=data_module, num_training=trainer.max_epochs
    )
    print(lr_finder)

    # Check if the lr_finder has completed successfully
    if lr_finder:
        # Plot with suggest=True to find the suggested learning rate
        fig = lr_finder.plot(suggest=True)
        fig.show()

        # Get the suggested learning rate
        suggested_lr = lr_finder.suggestion()
        print(f"Suggested learning rate: {suggested_lr}")
    else:
        print("Learning rate finding did not complete successfully.")

    # Train the model

    model.one_cycle_best_lr = suggested_lr

    if cfg["ckpt"]:
        trainer.fit(model, datamodule=data_module, ckpt_path=cfg["ckpt_path"])

    else:
        trainer.fit(model, datamodule=data_module)

    # Evaluate the model
    trainer.test(model, datamodule=data_module)
    print("Model evaluation completed...")

    # Save the model
    torch.save(
        model.state_dict(),
        "saved_resnet18_model.pth",
    )
    print("Model saved...")
    print("Training and evaluation completed...")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--devices", default=None)

    args = parser.parse_args()
    config = get_config()
    main(config, args)
