# Pytorch Lightning & Gradio Tutorial using CIFAR-10 dataset

## Table of Contents
- [Introduction](#introduction)
- [Pytorch Lightning](#pytorch-lightning)
- [Gradio](#gradio)
- [Training Logs](#training-logs)


## Introduction
This tutorial will guide you through the process of creating a simple image classifier using Pytorch Lightning and Gradio. This is following up from [s11-gradcam](https://github.com/aakashvardhan/s11_gradcam/tree/1d099dc9771bd0ebda318c561d50beab2a803128).

## Pytorch Lightning

PyTorch Lightning offers several advantages for deep learning development:

### Simplified Training Loop
- Abstracts boilerplate code, letting you focus on model architecture and hyperparameters.

### Readability and Maintainability
- Organizes code for easier reading and maintenance, beneficial for large projects.

### Reproducibility
- Standardized training loop for consistent and reproducible results.

### Automatic Hardware Acceleration
- Auto-handles GPU or TPU acceleration.

### Distributed Training Support
- Easy integration with frameworks like PyTorch Distributed Data Parallel (DDP).

### Experiment Logging
- Integrates with platforms like TensorBoard for metrics tracking.

### Multi-Framework Support
- Compatible with PyTorch, PyTorch Lightning Bolts, and more.

### Active Community
- Large, contributing community and a range of pre-built components.

## Gradio

Gradio is a Python library that allows you to quickly create UIs for your machine learning models. It is simple to use and supports a wide range of input types, making it easy to create interactive demos for your models. This tutorial will show you how to use Gradio to create a UI for your PyTorch Lightning model which can be added to huggingface spaces.

## Training Logs

The training logs for the model are as follows:

Showing raw output from the training logs:
```
INFO: Seed set to 42
INFO:lightning.fabric.utilities.seed:Seed set to 42
Seed set for reproducibility...
Downloading https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to ./data/cifar-10-python.tar.gz
100%|██████████| 170498071/170498071 [00:04<00:00, 40562199.33it/s]
Extracting ./data/cifar-10-python.tar.gz to ./data
Files already downloaded and verified
Files already downloaded and verified
Files already downloaded and verified
Data prepared and setup completed...
INFO: Using 16bit Automatic Mixed Precision (AMP)
INFO:lightning.pytorch.utilities.rank_zero:Using 16bit Automatic Mixed Precision (AMP)
INFO: GPU available: True (cuda), used: True
INFO:lightning.pytorch.utilities.rank_zero:GPU available: True (cuda), used: True
INFO: TPU available: False, using: 0 TPU cores
INFO:lightning.pytorch.utilities.rank_zero:TPU available: False, using: 0 TPU cores
INFO: IPU available: False, using: 0 IPUs
INFO:lightning.pytorch.utilities.rank_zero:IPU available: False, using: 0 IPUs
INFO: HPU available: False, using: 0 HPUs
INFO:lightning.pytorch.utilities.rank_zero:HPU available: False, using: 0 HPUs
Files already downloaded and verified
Files already downloaded and verified
Files already downloaded and verified
Files already downloaded and verified
INFO: LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
INFO:lightning.pytorch.accelerators.cuda:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
/usr/lib/python3.10/multiprocessing/popen_fork.py:66: RuntimeWarning: os.fork() was called. os.fork() is incompatible with multithreaded code, and JAX is multithreaded, so this will likely lead to a deadlock.
  self.pid = os.fork()
Finding best initial lr: 100%
 20/20 [00:21<00:00, 15.96it/s]
INFO: `Trainer.fit` stopped: `max_steps=20` reached.
INFO:lightning.pytorch.utilities.rank_zero:`Trainer.fit` stopped: `max_steps=20` reached.
INFO: Learning rate set to 0.0002511886431509582
INFO:lightning.pytorch.tuner.lr_finder:Learning rate set to 0.0002511886431509582
INFO: Restoring states from the checkpoint path at /content/.lr_find_8a65e97f-c807-4b0f-91be-0483f5f0ae16.ckpt
INFO:lightning.pytorch.utilities.rank_zero:Restoring states from the checkpoint path at /content/.lr_find_8a65e97f-c807-4b0f-91be-0483f5f0ae16.ckpt
INFO: Restored all states from the checkpoint at /content/.lr_find_8a65e97f-c807-4b0f-91be-0483f5f0ae16.ckpt
INFO:lightning.pytorch.utilities.rank_zero:Restored all states from the checkpoint at /content/.lr_find_8a65e97f-c807-4b0f-91be-0483f5f0ae16.ckpt
<lightning.pytorch.tuner.lr_finder._LRFinder object at 0x7d4ce6557bb0>
Suggested learning rate: 0.0002511886431509582
Files already downloaded and verified
Files already downloaded and verified
Files already downloaded and verified
Files already downloaded and verified
INFO: LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
INFO:lightning.pytorch.accelerators.cuda:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
INFO: 
  | Name   | Type   | Params
----------------------------------
0 | resnet | ResNet | 11.2 M
----------------------------------
11.2 M    Trainable params
0         Non-trainable params
11.2 M    Total params
44.696    Total estimated model params size (MB)
INFO:lightning.pytorch.callbacks.model_summary:
  | Name   | Type   | Params
----------------------------------
0 | resnet | ResNet | 11.2 M
----------------------------------
11.2 M    Trainable params
0         Non-trainable params
11.2 M    Total params
44.696    Total estimated model params size (MB)
Epoch 19: 100%
 352/352 [00:22<00:00, 15.71it/s, v_num=1, train_loss_step=0.149, train_acc_step=95.80, train_loss_epoch=0.229, train_acc_epoch=92.30, val_loss_step=0.0444, val_acc_step=100.0, val_loss_epoch=0.391, val_acc_epoch=86.70]
INFO: `Trainer.fit` stopped: `max_epochs=20` reached.
INFO:lightning.pytorch.utilities.rank_zero:`Trainer.fit` stopped: `max_epochs=20` reached.
Files already downloaded and verified
Files already downloaded and verified
INFO: LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
INFO:lightning.pytorch.accelerators.cuda:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Testing DataLoader 0: 100%
 79/79 [00:01<00:00, 46.97it/s]
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│      test_acc_epoch       │     90.31999969482422     │
│      test_loss_epoch      │    0.3008700907230377     │
└───────────────────────────┴───────────────────────────┘
Model evaluation completed...
Model saved...
Training and evaluation completed...

```

### Training & Val Loss/Accuracy Graphs

<table>
  <tr>
    <td><img src="https://github.com/aakashvardhan/s13-lightning-gradio/blob/main/asset/train_acc_step.png" alt="Plot 1" style="width: 100%;"/></td>
    <td><img src="https://github.com/aakashvardhan/s13-lightning-gradio/blob/main/asset/train_loss_step.png" alt="Plot 2" style="width: 100%;"/></td>
  </tr>
  <tr>
    <td align="center">train_acc_step</td>
    <td align="center">train_loss_step</td>
  </tr>
  <tr>
    <td><img src="https://github.com/aakashvardhan/s13-lightning-gradio/blob/main/asset/test_acc_step.png" alt="Plot 4" style="width: 100%;"/></td>
    <td><img src="https://github.com/aakashvardhan/s13-lightning-gradio/blob/main/asset/test_loss_step.png" alt="Plot 5" style="width: 100%;"/></td>
  </tr>
  <tr>
    <td align="center">test_acc_step</td>
    <td align="center">test_loss_step</td>
  </tr>
</table>


