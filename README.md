# PyTorch Tutorial: Classification and Segmentation with JIT and TorchScript

Welcome to our PyTorch tutorial, where we'll guide you through the process of solving classification and segmentation problems. Here, we'll cover the transformation of your trained PyTorch models into JIT (Just-In-Time) and TorchScript formats. This will make them compatible with C++ environments, facilitating seamless integration and execution.

## Overview

This tutorial focuses on addressing two conventional challenges: classification and segmentation using PyTorch. The highlight of our guide is the conversion of trained models into JIT and TorchScript formats. This conversion ensures easy integration and execution in C++ environments.

## Introduction to TorchScript

TorchScript, a statically-typed, high-performance subset of Python, is designed for representing PyTorch models. Explore its advantages, including improved execution speed and compatibility with C++.

## Usage

To begin, navigate to the respective directories for classification and segmentation tutorials. Follow the step-by-step instructions to create models, train them, and convert them into JIT and TorchScript formats. We also do some compare between two result, include the errors and fps calculation.

[Segmentation Tutorial](./tutorials/segmentation.ipynb)
<a target="_blank" href="https://colab.research.google.com/github/nhtlongcs/torch-realtime/blob/master/tutorials/segmentation.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

[Image Classification Tutorial](./tutorials/classification.ipynb) 
<a target="_blank" href="https://colab.research.google.com/github/nhtlongcs/torch-realtime/blob/master/tutorials/classification.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## Key Steps

In both tutorials, you'll follow a series of steps with variations based on the specific problem and input data.

### Step 1: Prepare Your PyTorch Model

Ensure that your PyTorch model is well-defined and trained. Encapsulate the forward pass logic cleanly for a smooth conversion process.

### Step 2: Convert PyTorch to TorchScript

Use the `torch.jit.trace` function to convert your PyTorch model into TorchScript. This takes two main parameters, includes a callable (e.g., a function or method) that you want to convert to TorchScript and an example input that the function will be traced with. This example input helps TorchScript to trace the operations and build the computational graph.

```python
import torch

# Define your PyTorch model
class MyModel(torch.nn.Module):
    # ... (define your model architecture)

# Create an instance of your model
model = MyModel()

# Create an example input
input = torch.rand(C, W, H) 

# Convert to TorchScript
script_model = torch.jit.trace(model, input)
with torch.no_grad():
    output = script_model(input)

# Save to file
script_model.save('model.pth') 
```

### Step 3: Integrating with C++ for Edge Deployment

Integrate the saved TorchScript model into your C++ application using the PyTorch C++ API. This enables inference in your C++ code. Refer to the official PyTorch documentation for detailed information on integrating TorchScript with C++. Make sure to have the necessary C++ compiler and dependencies installed, such as libtorch.

## Conclusion

By following these steps, you'll be equipped to integrate your PyTorch models into C++ environments, allowing for efficient deployment on edge devices. Experiment with your own models and datasets, and explore the possibilities of extending this integration for real-world applications. Happy coding.
