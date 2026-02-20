# AI Starter Kit – MNIST (PyTorch + W&B)

This repository contains a small, reproducible baseline project to train an MNIST digit classifier in **PyTorch**, track experiments with **Weights & 
Biases (W&B)**, and tune hyperparameters using **W&B Sweeps**.

## Dataset
- **MNIST** handwritten digits (0–9), grayscale images, 28×28 pixels  
- Standard split: ~60k train / ~10k test

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

