import lightning as L

import torch
import torch.nn.functional as F

from s11_gradcam.models.model_utils import model_summary
from s11_gradcam.models.resnet import ResNet18

class LitResNet(L.LightningModule):
    """
    LightningModule implementation of ResNet for training and evaluation.

    Args:
        resnet: The ResNet model.
        config: Configuration parameters for the model.
    """

    def __init__(self, 
                 config,
                 one_cycle_best_lr=0.01,
                 learning_rate=0.01):
        super().__init__()
        self.resnet = ResNet18()
        self.config = config
        self.learning_rate = learning_rate
        self.one_cycle_best_lr = one_cycle_best_lr
        
        self.save_hyperparameters()

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x: The input data.

        Returns:
            The output of the model.
        """
        return self.resnet(x)

    def training_step(self, batch, batch_idx):
        """
        Performs a single training step on the given batch of data.

        Args:
            batch: A tuple containing the input data and corresponding target labels.
            batch_idx: The index of the current batch.

        Returns:
            The computed loss value for the training step.
        """
        data, target = batch
        output = self.resnet(data)
        loss = F.cross_entropy(output, target)
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        processed = len(data)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        acc = 100 * correct / processed
        self.log(
            "train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        return loss

    def test_step(self, batch, batch_idx):
        """
        Performs a single test step on the given batch of data.

        Args:
            batch: A tuple containing the input data and corresponding target labels.
            batch_idx: The index of the current batch.

        Returns:
            The computed loss value for the test step.
        """
        data, target = batch
        output = self.resnet(data)
        loss = F.cross_entropy(output, target)
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        processed = len(data)
        self.log(
            "test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        acc = 100 * correct / processed
        self.log(
            "test_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        return loss

    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler for training.

        Returns:
            A tuple containing the optimizer and learning rate scheduler.
        """
        optimizer = torch.optim.Adam(
            self.resnet.parameters(), lr=self.learning_rate, weight_decay=1e-4
        )

        self.trainer.fit_loop.setup_data()
        dataloader = self.trainer.train_dataloader
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.one_cycle_best_lr,  # need to manually set this after using torch-lr-finder
            steps_per_epoch=len(dataloader),
            epochs=self.config["num_epochs"],
            pct_start=5 / 24,
            div_factor=100,
            three_phase=False,
            final_div_factor=100,
            anneal_strategy="linear",
        )

        lr_scheduler = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
            "monitor": "train_loss",
        }

        return [optimizer], [lr_scheduler]

    def on_train_epoch_start(self):
        """
        Prints the model summary when training starts.
        """
        model_summary(self.resnet, input_size=(3, 32, 32))
