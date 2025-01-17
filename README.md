# MSACN: Multi-Scale Adaptive Convolutional Network

## Overview

MSACN (Multi-Scale Adaptive Convolutional Network) is a deep learning-based model designed for accurate classification of crop diseases. This project aims to improve classification performance by leveraging multi-scale feature extraction and adaptive convolutional mechanisms.

## Features

- **Multi-Scale Adaptive Convolution**: Enhances feature extraction across different spatial scales.
- **Improved Data Augmentation**: Handles class imbalances and improves generalization.
- **Optimized Regularization Techniques**: Prevents overfitting while maintaining high classification accuracy.
- **Comparative Analysis**: Benchmarks against other deep learning models such as MLP ANN, DMCNN, MConvExt, and MSCPNet.

## Dataset

This project utilizes publicly available datasets for crop disease classification:

- **[PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)** (Corn, Strawberry, Tomato)
- **[Wheat Leaf Dataset](https://www.kaggle.com/datasets/olyadgetch/wheat-leaf-dataset)** (Wheat)
- **[Rice Disease Dataset](https://www.kaggle.com/datasets/anshulm257/rice-disease-dataset)** (Rice)

## Installation

```bash
git clone https://github.com/your-repo/MSACN.git
cd MSACN
pip install -r requirements.txt

## Model Architecture

The MSACN model consists of the following components:

- **Multi-Scale Convolutional Blocks**
- **Adaptive Feature Selection**
- **Batch Normalization and Dropout**
- **Fully Connected Layers with Softmax Output**

## Training

To train the model, use the following command:

```bash
python train.py --epochs 50 --batch_size 32 --lr 0.001
```

### Training Configuration

- Optimizer: Adam with cosine annealing learning rate scheduler
- Loss Function: Categorical Crossentropy
- Metrics: Precision, Recall, F1-score

## Evaluation

```bash
python evaluate.py --model best_model.pth
```

The results are compared using classification reports and confusion matrices for each class.

## Results

| Model   | Precision | Recall | F1-score |
|---------|----------|--------|----------|
| MSACN   | XX.XX%   | XX.XX% | XX.XX%   |
| MLP ANN | XX.XX%   | XX.XX% | XX.XX%   |
| DMCNN   | XX.XX%   | XX.XX% | XX.XX%   |
| MConvExt| XX.XX%   | XX.XX% | XX.XX%   |
| MSCPNet | XX.XX%   | XX.XX% | XX.XX%   |

## Usage

To classify new images:

```bash
python predict.py --image path/to/image.jpg --model best_model.pth
```

## Future Work

- Improve model interpretability.
- Expand dataset coverage.
- Deploy as a web-based classification tool.
