# PyTorch Tutorial: Classification and Segmentation with JIT and TorchScript

## Overview

This repository provides a tutorial on solving two traditional problems - classification and segmentation - using PyTorch. The main emphasis is on converting the trained models into JIT (Just-In-Time) and TorchScript formats, enabling them to be seamlessly integrated and executed in C++ environments.

## Table of Contents

1. [Introduction](#introduction)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Structure of the Repository](#repository-structure)
6. [Classification Tutorial](#classification-tutorial)
7. [Segmentation Tutorial](#segmentation-tutorial)
8. [Converting to JIT and TorchScript](#converting-to-jit-and-torchscript)
9. [Running in C++](#running-in-c++)
10. [Contributing](#contributing)
11. [License](#license)

## Introduction

PyTorch is a powerful deep learning framework, and this tutorial aims to guide users through implementing classification and segmentation tasks using PyTorch. Additionally, it provides instructions on converting these models into JIT and TorchScript formats for deployment in C++ applications.

## Requirements

Ensure you have the following dependencies installed:

- Python (>=3.6)
- PyTorch (>=1.6)
- torchvision
- C++ compiler (for running JIT and TorchScript in C++)

## Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/nhtlongcs/torch-realtime/
cd PyTorch-JIT-Tutorial
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Usage

Follow the tutorials in the respective directories for classification and segmentation. The tutorials provide step-by-step guidance on creating models, training, and converting them into JIT and TorchScript formats.

```bash
cd classification_tutorial
python train_classification.py
python convert_to_jit_and_torchscript.py
```

```bash
cd segmentation_tutorial
python train_segmentation.py
python convert_to_jit_and_torchscript.py
```

## Repository Structure

- `classification_tutorial/`: Contains the classification tutorial.
- `segmentation_tutorial/`: Contains the segmentation tutorial.
- `utils/`: Utility functions and scripts.
- `requirements.txt`: Python dependencies.

## Classification Tutorial

In this tutorial, we cover the implementation of a classification model using PyTorch. Follow the steps outlined in the README within the `classification_tutorial` directory.

## Segmentation Tutorial

The segmentation tutorial focuses on implementing a segmentation model using PyTorch. Follow the steps outlined in the README within the `segmentation_tutorial` directory.

## Converting to JIT and TorchScript

Learn how to convert your trained PyTorch models into JIT and TorchScript formats using the provided scripts.

## Running in C++

Detailed instructions on running the converted models in a C++ environment are provided in this section. Ensure you have the necessary C++ compiler and dependencies installed.
