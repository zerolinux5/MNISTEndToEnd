# MNISTEndToEnd

Train a convolutional model on MNIST using train/val/test splits.
Logs metrics and artifacts to Weights & Biases.

## Training
```
uv sync
wandb login
uv run python train.py
```