# Assignment 1: Multi-Layer Perceptron for Image Classification

## Overview

Implementation of a multi-layer perceptron (MLP) neural network from scratch using only NumPy for DA6401 Deep Learning course. The network supports training on MNIST and Fashion-MNIST datasets with configurable architectures, optimizers, activations, and loss functions.

## Features

- **Layers**: Fully connected (Dense) layers with forward/backward pass
- **Activations**: ReLU, Sigmoid, Tanh
- **Optimizers**: SGD, Momentum, NAG (Nesterov), RMSProp
- **Loss Functions**: Cross-Entropy, Mean Squared Error (MSE)
- **Weight Initialization**: Random, Xavier, Zero
- **Experiment Tracking**: Weights & Biases (W&B) integration

## Usage

### Training
```bash
python src/train.py
```

### Inference
```bash
python src/inference.py --model_path src/best_model.npy
```

## Project Structure

```
src/
├── ann/
│   ├── activations.py        # ReLU, Sigmoid, Tanh, Softmax
│   ├── neural_layer.py       # Dense layer (forward/backward)
│   ├── neural_network.py     # Main NeuralNetwork class
│   ├── objective_functions.py # CrossEntropy, MSE losses
│   └── optimizers.py         # SGD, Momentum, NAG, RMSProp
├── utils/
│   └── data_loader.py        # Data loading utilities
├── train.py                  # Training script with CLI
├── inference.py              # Inference script with CLI
├── best_model.npy            # Best model weights (after training)
└── best_config.json          # Best hyperparameter config (after training)
```

## Links

- **W&B Report**: [TODO: Add your W&B report link here]
- **GitHub Repository**: [https://github.com/usnaveen/da6401_assignment_1](https://github.com/usnaveen/da6401_assignment_1)

## Contact

For questions or issues, please contact the teaching staff or post on the course forum.
